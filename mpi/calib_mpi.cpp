// Algorithm 1, P2 (MPI). Every rank reads its own slice of the images, then the ranks
// combine with three allreduces: the max, the histogram and the two winners. Run from
// the project folder:
//   mpirun -np P ./cmake-build-release/calib_mpi data/layer01_act.bin [--radius R]
//                                                [--first K] [--images N] [--runs R]
// --radius: smoothing, 0 = off (default). --first, --images: calibrate on images K .. K + N - 1.
// --runs: timed runs after one warm-up (default 5).
//
// Steps per layer, with the matching CUDA kernel in brackets:
//   1. read my slice of the images          MPI_File_read_at, no communication
//   2. local max            [absmax]       -> MPI_Allreduce MAX: every rank gets m
//   3. local histogram      [hist]         -> MPI_Allreduce SUM: every rank gets the full histogram
//   4. smooth, percentile   [smooth, percentile]  every rank on its own copy, no communication
//   5. search my candidates [mse, kl]      cyclic: candidate c belongs to rank c mod p
//   6. winners              [argmin]       -> MPI_Allreduce MINLOC, both methods in one message
// The math in every step is the one-thread version from calib_common.h, the same code
// that checks P1, so P2 gives identical thresholds for every p.
#include <mpi.h>

#include <cstring>
#include <string>

#include "calib_common.h"

// This rank's share of the chosen images, read straight from the file.
struct Slice {
    std::vector<float> x;  // my values
    size_t file_images = 0; // images in the file
    size_t images = 0;      // images all ranks use together
    size_t per_image = 0;   // values per image
    size_t my_first = 0;    // first image of mine
    size_t my_images = 0;   // how many images are mine
};

// Rank r owns images r * N / p .. (r + 1) * N / p - 1 of the N chosen, so shares differ
// by at most one image. Images are first in the file, so each share is one contiguous read.
static Slice read_slice(const char* path, size_t first, size_t images, int rank, int p) {
    MPI_File f;
    if (MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &f) != MPI_SUCCESS) {
        if (rank == 0) std::fprintf(stderr, "cannot open %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Header: int32 ndim, then ndim int32 dims
    int32_t ndim = 0;
    MPI_File_read_at(f, 0, &ndim, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    std::vector<int32_t> dims(ndim > 0 ? ndim : 1);
    MPI_File_read_at(f, sizeof(int32_t), dims.data(), ndim, MPI_INT32_T, MPI_STATUS_IGNORE);

    Slice s;
    s.file_images = dims[0];
    s.per_image = 1;
    for (int i = 1; i < ndim; ++i) s.per_image *= dims[i];
    if (images == 0 && first < s.file_images) images = s.file_images - first;
    if (ndim < 1 || images == 0 || first + images > s.file_images) {
        if (rank == 0) std::fprintf(stderr, "%s has %zu images\n", path, s.file_images);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.images = images;
    s.my_first = first + rank * images / p;
    s.my_images = first + (rank + 1) * images / p - s.my_first;

    // Skip the header and the images before mine
    s.x.resize(s.my_images * s.per_image);
    const MPI_Offset offset = sizeof(int32_t) * (1 + ndim) + s.my_first * s.per_image * sizeof(float);
    MPI_File_read_at(f, offset, s.x.data(), static_cast<int>(s.x.size()), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&f);
    return s;
}

// The phases timed separately. The report splits time into computation and communication.
enum Phase { LOCAL_MAX, MAX_ALLREDUCE, LOCAL_HIST, HIST_ALLREDUCE, SMOOTH_PERCENTILE, SEARCH, MINLOC_ALLREDUCE, NPHASES };
static const char* PHASE_NAMES[NPHASES] = {"local max", "max allreduce", "local histogram",
                                           "hist allreduce", "smooth+percentile", "search",
                                           "minloc allreduce"};
static const bool IS_COMM[NPHASES] = {false, true, false, true, false, false, true};

// A (score, candidate) pair laid out the way MPI_FLOAT_INT expects
struct ScoreIndex {
    float score;
    int index;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, p = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    const char* usage = "usage: calib_mpi FILE [--radius R] [--first K] [--images N] [--runs R]\n";
    int radius = 0;               // smoothing, 0 = off
    int runs = 5;                 // timed runs, after one warm-up run
    size_t first = 0, images = 0; // images = 0: every image from first on
    bool ok = argc >= 2;
    for (int a = 2; ok && a < argc; ++a) {
        if (std::strcmp(argv[a], "--radius") == 0 && a + 1 < argc) radius = std::atoi(argv[++a]);
        else if (std::strcmp(argv[a], "--runs") == 0 && a + 1 < argc) runs = std::atoi(argv[++a]);
        else if (std::strcmp(argv[a], "--first") == 0 && a + 1 < argc) first = std::strtoull(argv[++a], nullptr, 10);
        else if (std::strcmp(argv[a], "--images") == 0 && a + 1 < argc) images = std::strtoull(argv[++a], nullptr, 10);
        else ok = false;
    }
    if (!ok || runs < 1) {
        if (rank == 0) std::fprintf(stderr, "%s", usage);
        MPI_Finalize();
        return 2;
    }

    // 1. Read my slice. I/O is timed separately from the algorithm.
    MPI_Barrier(MPI_COMM_WORLD);
    double t_read = MPI_Wtime();
    const Slice slice = read_slice(argv[1], first, images, rank, p);
    t_read = MPI_Wtime() - t_read;
    MPI_Allreduce(MPI_IN_PLACE, &t_read, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const size_t n = slice.images * slice.per_image; // values across all ranks
    const std::vector<float> weights = gaussian_weights(radius);
    if (rank == 0)
        std::printf("%s: %d ranks, images %zu .. %zu of %zu, %zu values (%zu to %zu images per rank), "
                    "smoothing radius %d\n", argv[1], p, first, first + slice.images - 1, slice.file_images, n,
                    slice.images / p, (slice.images + p - 1) / p, radius);

    // Per phase: the slowest rank's time and the fastest rank's, averaged over the timed runs
    double phase_max[NPHASES] = {}, phase_min[NPHASES] = {};
    float m = 0.0f, width = 0.0f;
    std::vector<unsigned> ghist(NBINS);
    std::vector<int> pbins;
    ScoreIndex best[2] = {}; // [0] = MSE, [1] = KL

    for (int run = 0; run <= runs; ++run) { // run 0 is the warm-up and isn't counted
        double t[NPHASES] = {};
        // Starts a phase: wait for every rank, so the clock measures this phase only and a
        // slow rank's lateness doesn't show up as communication time in the next one
        double start = 0;
        auto begin = [&]() { MPI_Barrier(MPI_COMM_WORLD); start = MPI_Wtime(); };
        auto end = [&](Phase ph) { t[ph] = MPI_Wtime() - start; };

        // 2. Global max
        begin();
        float local_max = cpu_absmax(slice.x.data(), slice.x.size());
        end(LOCAL_MAX);
        begin();
        MPI_Allreduce(&local_max, &m, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        end(MAX_ALLREDUCE);
        // Every rank computes the same floats from the same m, so bins match across ranks
        const float bins_per_unit = NBINS / m;
        width = m / NBINS;

        // 3. Global histogram. Each rank counts into its own private array: no contention.
        begin();
        const std::vector<unsigned> hist = cpu_histogram(slice.x.data(), slice.x.size(), bins_per_unit);
        end(LOCAL_HIST);
        begin();
        MPI_Allreduce(hist.data(), ghist.data(), NBINS, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        end(HIST_ALLREDUCE);

        // 4. Smoothing and percentile on every rank. 2,048 bins cost less than a message.
        begin();
        const std::vector<float> smooth = cpu_smooth(ghist, weights);
        const std::vector<float> h = kl_hist(smooth);
        pbins = cpu_percentile(ghist, n, PERCENTILES);
        end(SMOOTH_PERCENTILE);

        // 5. My candidates, cyclic. KL's cost grows with i, so a block split would give
        //    the last rank the most expensive ones. c only grows, so "<" keeps the first on a tie.
        begin();
        best[0] = best[1] = {INFINITY, NCANDIDATES};
        for (int c = rank; c < NCANDIDATES; c += p) {
            const float mse = mse_score(smooth, width, n, 128 + c);
            if (mse < best[0].score) best[0] = {mse, c};
            const float kl = kl_score(h, 128 + c);
            if (kl < best[1].score) best[1] = {kl, c};
        }
        end(SEARCH);

        // 6. Winners. MINLOC keeps the lowest score and, on a tie, the lowest index, the
        //    same first-index rule as argmin_kernel. A count of 2 sends both methods at once.
        begin();
        MPI_Allreduce(MPI_IN_PLACE, best, 2, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
        end(MINLOC_ALLREDUCE);

        // Slowest and fastest rank per phase, outside the timed part
        double tmax[NPHASES], tmin[NPHASES];
        MPI_Reduce(t, tmax, NPHASES, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(t, tmin, NPHASES, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (run == 0) continue;
        for (int ph = 0; ph < NPHASES; ++ph) {
            phase_max[ph] += tmax[ph] * 1e3 / runs;
            phase_min[ph] += tmin[ph] * 1e3 / runs;
        }
    }

    if (rank == 0) {
        // The histogram must count every value exactly once
        unsigned long long counted = 0;
        for (const unsigned c : ghist) counted += c;
        std::printf("check: histogram counts %llu of %zu values\n", counted, n);

        std::printf("\ncutoffs:\n");
        std::printf("  max: T = %f\n", m);
        std::printf("  MSE: bucket %d, T = %f\n", 128 + best[0].index, static_cast<float>(128 + best[0].index) * width);
        std::printf("  KL: bucket %d, T = %f\n", 128 + best[1].index,
                    (static_cast<float>(128 + best[1].index) + 0.5f) * width);
        for (size_t k = 0; k < pbins.size(); ++k)
            std::printf("  percentile %.4f%%: bucket %d, T = %f\n", PERCENTILES[k] * 100, pbins[k],
                        static_cast<float>(pbins[k] + 1) * width);
        print_result(radius, first, slice.images, m, width, 128 + best[0].index, 128 + best[1].index, pbins);

        std::printf("\ntiming: average of %d runs after 1 warm-up; read %.3f ms (I/O, not counted)\n",
                    runs, t_read * 1e3);
        std::printf("  %-18s %-6s %11s %11s\n", "phase", "kind", "slowest ms", "fastest ms");
        double comp = 0, comm = 0;
        for (int ph = 0; ph < NPHASES; ++ph) {
            std::printf("  %-18s %-6s %11.4f %11.4f\n", PHASE_NAMES[ph], IS_COMM[ph] ? "comm" : "comp",
                        phase_max[ph], phase_min[ph]);
            (IS_COMM[ph] ? comm : comp) += phase_max[ph];
        }
        std::printf("  computation %.4f ms, communication %.4f ms, total %.4f ms\n", comp, comm, comp + comm);
        // One line for scripts: ranks, then the slowest rank's ms per phase
        std::printf("mpi_ms: p %d", p);
        for (int ph = 0; ph < NPHASES; ++ph) {
            std::string name = PHASE_NAMES[ph];
            for (char& ch : name) if (ch == ' ' || ch == '+') ch = '_';
            std::printf(" %s %.5f", name.c_str(), phase_max[ph]);
        }
        std::printf(" search_fastest %.5f comp %.5f comm %.5f total %.5f\n", phase_min[SEARCH], comp, comm, comp + comm);
    }
    MPI_Finalize();
    return 0;
}
