// Algorithm 1. Run: ./calib ~/ptq/layer01_act.bin
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include <chrono>
#include <cmath>
#include <complex.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// File format: int32 ndim, int32 dims, then raw float32.
static std::vector<float> read_tensor(const char* path) {
    // Read activations from file
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    int32_t ndim = 0, dim = 0;
    size_t count = 1;
    bool ok = std::fread(&ndim, sizeof(ndim), 1, f) == 1;
    for (int i = 0; ok && i < ndim; ++i) {
        ok = std::fread(&dim, sizeof(dim), 1, f) == 1;
        count *= dim;
    }
    if (!ok) { std::fprintf(stderr, "bad header: %s\n", path); std::exit(1); }
    std::vector<float> data(count);
    if (std::fread(data.data(), sizeof(float), count, f) != count) {
        std::fprintf(stderr, "short read: %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return data;
}

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

#define NBINS 2048
#define LEVELS 128 // Using int8 quantization
#define NCANDIDATES 1921 // i = 128 .. 2048

// Order for one layer:
//   absmax -> histogram -> smooth (optional) -> KL, MSE -> argmin for each
//                       -> percentile (raw histogram)
// Zero d_max_bits and d_hist on the host before launching.

// Sums value across a block using warp shuffles, then s_red[32 warps].
// KL and MSE need this to total one number from every thread.
//   1. Add the warp's 32 values with shuffles. Only lane 0 gets the right sum.
//   2. Lane 0 of each warp writes its sum to s_red.
//   3. __syncthreads.
//   4. Every thread adds up s_red and returns the block total.
__device__ float block_reduce_sum(float value, float* s_red) {
    // Sum by warp using shuffling
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    // One thread per warp sums
    const unsigned warp_t_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;
    // Oth thread per ward writes to shared memory
    if (warp_t_id == 0) s_red[warp_id] = value;
    __syncthreads();
    // Add up all warp sums
    float total = 0.0f;
    for (int t = 0; t < blockDim.x/32; ++t) total += s_red[t];
    __syncthreads();
    return total;
}

// max |x| over one layer, stored as float bits. Launch: <<<grid, 256>>>
// The histogram covers 0 .. m, so m is needed first.
//   1. Walk x by grid-stride, keeping the largest |x|.
//   2. Reduce the warp to one maximum with shuffles.
//   3. Lane 0 does atomicMax on d_max_bits. Comparing float bits as ints is safe
//      because m >= 0.
// Then on the host: read m and set inv_width = NBINS / m.
__global__ void absmax_kernel(const float* x, size_t n, int* d_max_bits) {
    float m = 0.0f;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n_workers = blockDim.x * gridDim.x;

    // Get max of nums assigned to worker. Coalesce by getting ever n_workers number
    for (size_t i = idx; i < n; i += n_workers) {
        m = fmaxf(m, fabsf(x[i]));
    }
    // Get warp max with shuffles
    for (int offset = 16; offset > 0; offset /= 2) {
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
    }

    // Set max atomically from the accumulated maxes per warp
    if (threadIdx.x%32 == 0) {
        atomicMax(d_max_bits, __float_as_int(m));
    }
}

// Baseline: every increment is a global atomic. Launch: <<<grid, 256>>>
// Only for timing, to show what contention costs.
//   1. Walk x by grid-stride.
//   2. bin = min(int(|x| * inv_width), NBINS - 1).
//   3. atomicAdd on d_hist[bin].
__global__ void hist_global_kernel(const float* x, size_t n, float inv_width, unsigned* d_hist) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n_workers = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += n_workers) {
        int bin = min((int)fabsf(x[i] * inv_width), NBINS - 1);
        atomicAdd(&d_hist[bin], 1u);
    }
}

// Per-block shared histogram, then merge. Launch: <<<grid, 256>>>
// A private copy per block avoids most of the fighting over global memory.
//   1. Zero s_hist.
//   2. __syncthreads.
//   3. Walk x by grid-stride and compute the bin.
//   4. atomicAdd on s_hist[bin].
//      AGGREGATE_ZEROS: a warp gets its zero lanes from
//      __ballot_sync(__activemask(), a == 0), and one zero lane adds the whole
//      count. This replaces up to 32 atomics on bin 0 with one.
//   5. __syncthreads.
//   6. atomicAdd each non-empty s_hist bin into d_hist.
template <bool AGGREGATE_ZEROS>
__global__ void hist_kernel(const float* tensor, size_t n, float bins_per_unit, unsigned* d_hist) {
    // Per warp private counters. Lower contention at global memory
    __shared__ unsigned s_hist[NBINS];

    // Every thread clear memory of every 256th bin
    for (unsigned b = threadIdx.x; b < NBINS; b += blockDim.x) {
        s_hist[b] = 0u;
    }
    __syncthreads();

    // Count into shared memory to create histogram
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n_workers = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += n_workers) {
        const int bin = min((int)fabsf(tensor[i] * bins_per_unit), NBINS - 1);
        if (AGGREGATE_ZEROS) {
            // Count how many threads value falls into 0 bin
            const unsigned zeros = __ballot_sync(__activemask(), bin == 0);
            if (bin != 0) {
                // Threads that don't have zero do regular atomic add
                atomicAdd(&s_hist[bin], 1u);
                // First thread who got zero adds all zeros into histogram for all zero value threads
            } else if (threadIdx.x%32 == __ffs(static_cast<int>(zeros)) - 1) {
                atomicAdd(&s_hist[0], static_cast<unsigned>(__popc(zeros)));
            }
        } else {
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // Sum warp histograms together into global histogram
    for (unsigned t = threadIdx.x; t < NBINS; t += blockDim.x) {
        if (s_hist[t] != 0u) {
            atomicAdd(&d_hist[t], s_hist[t]);
        }
    }
}

// Optional, 2 * radius + 1 symmetric weights summing to 1, over bins 1 .. 2047 only.
// Bin 0 is copied. An index j off the end reads bin 1 - j or 4095 - j, which keeps
// the total. radius = 0 with weight 1 is the unsmoothed run. Launch: <<<2, 1024>>>
// Turns random spikes and gaps in the sparse tail into a slope.
//   1. One thread per bin: b = blockIdx.x * 1024 + threadIdx.x.
//   2. Bin 0: copy it and stop.
//   3. Otherwise take the weighted sum of bins b - radius .. b + radius, reflecting
//      at the ends as above.
//   4. Write d_smooth[b].
__global__ void smooth_kernel(const unsigned* d_hist, const float* d_weights, int radius, float* d_smooth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NBINS) {
        return;
    }
    // Copy bucket 0 dont modify
    if (idx == 0) {
        d_smooth[0] = (float)d_hist[0];
        return;
    }
    float sum = 0.0f;
    // Sum all weighted sums to get smoothed bin values
    for (int k = -radius; k <=radius; ++k) {
        int j = idx + k;
        if (j < 1) j= 1- j;
        if (j >= NBINS) j = 2 * NBINS - 1 - j;
        sum += d_weights[k+radius] * static_cast<float>(d_hist[j]);
    }
    d_smooth[idx] = sum;
}

// One block per candidate; treat bin 0 as bin 1 and only q == 0 as eps.
// total = outliers + in-range sum after that fix. Launch: <<<NCANDIDATES, 256>>>
// Measures the information lost by clipping at bin i and using 128 levels.
// Candidate: i = 128 + blockIdx.x.
//   1. outliers = sum of bins i .. 2047, using block_reduce_sum.
//   2. Threads 0 .. 127 each own one group. Bin j is in group j * 128 / i. Sum the
//      group's bins into s_gsum and count its non-empty bins into s_gnz.
//   3. __syncthreads.
//   4. q_total = block sum of s_gsum, then total = outliers + q_total.
//   5. For each bin j < i:
//        p = bin j / total, and bin i - 1 also gets the outliers. Skip if p == 0.
//        q = 0 for an empty bin, else s_gsum[g] / s_gnz[g] / q_total.
//        If q == 0, use eps. Add p * log(p / q).
//   6. block_reduce_sum, then thread 0 writes d_score[blockIdx.x].
// Threads 128 .. 255 have no group in step 2 but must still reach every barrier.

// Bucket j for KL: bucket 0 reads bucket 1's count (the zero fix).
__device__ float kl_bin(const float* h, int j) { return h[j == 0 ? 1 : j]; }

__global__ void kl_kernel(const float* d_smooth, float* d_score) {
    __shared__ float s_gsum[128];
    __shared__ int s_gnz[128];
    __shared__ float s_red[32];
    const int i = 128 + blockIdx.x; // histogram bucket to cut off at, 128 .. 2048

    // 1. Values past the cut
    float part = 0.0f;
    for (int j = i + threadIdx.x; j < NBINS; j += blockDim.x) part += d_smooth[j];
    const float outliers = block_reduce_sum(part, s_red);

    // 2. Thread g sums group g. The + 127 rounds up, so bucket j lands in group j * 128 / i
    if (threadIdx.x < 128) {
        const int g = threadIdx.x;
        const int lo = (g * i + 127) / 128;       // first bucket in group g
        const int hi = ((g + 1) * i + 127) / 128; // one past the last
        float sum = 0.0f;
        int nz = 0;
        for (int j = lo; j < hi; ++j) {
            const float h = kl_bin(d_smooth, j);
            sum += h;
            nz += (h != 0.0f);
        }
        s_gsum[g] = sum;
        s_gnz[g] = nz;
    }
    // 3. Wait for every group before reading s_gsum
    __syncthreads();

    // 4. Totals. Threads without a group add 0
    const float q_total = block_reduce_sum(threadIdx.x < 128 ? s_gsum[threadIdx.x] : 0.0f, s_red);
    const float total = outliers + q_total;

    // 5. KL terms for my share of buckets 0 .. i - 1
    const float eps = 1e-4f;
    float kl = 0.0f;
    for (int j = threadIdx.x; j < i; j += blockDim.x) {
        const float h = kl_bin(d_smooth, j);
        const float p = (h + (j == i - 1 ? outliers : 0.0f)) / total; // last bucket gets the clipped values
        if (p == 0.0f) continue;                                      // empty in P adds nothing
        const int g = j * 128 / i;
        float q = (h != 0.0f) ? s_gsum[g] / s_gnz[g] / q_total : 0.0f; // group count spread evenly
        if (q == 0.0f) q = eps;
        kl += p * logf(p / q);
    }

    // 6. Block total
    kl = block_reduce_sum(kl, s_red);
    if (threadIdx.x == 0) d_score[blockIdx.x] = kl;
}

// One block per candidate, T = i * width. Launch: <<<NCANDIDATES, 256>>>
// Measures the average squared error from rounding and clipping at T.
//   1. s = T / 127.
//   2. Each thread loops over its share of all NBINS bins:
//        c = (j + 0.5) * width, the bin center.
//        qv = s * min(round(c / s), 127).
//        err += bin j * (c - qv)^2.
//   3. block_reduce_sum, then divide by total.
//   4. Thread 0 writes d_score[blockIdx.x].
__global__ void mse_kernel(const float* d_smooth, const float width, const float total, float* d_score) {
    __shared__ float s_red[32];
    // Each block does one cutoff candidate
    const unsigned i = 128 + blockIdx.x; // histogram bucket num to cutoff at from 128-2048
    const float T = static_cast<float>(i) * width; // The cutoff value
    const float s = T/127.0f; // Gap between quantize levels

    float err = 0.0f; // The error for thread
    // Split 2048 histogram buckets amongst threads per block
    for (unsigned bucket = threadIdx.x; bucket < NBINS; bucket += blockDim.x) {
        const float c = (static_cast<float>(bucket)+0.5f) * width; // middle of bucket
        const float qv = s * fminf(roundf(c/s), 127.f); // nearest quant level
        const float d = c - qv; // error bucket - nearest quant level
        err += d_smooth[bucket] * d * d; // Count per histogram bucket * error^2 for total squared error for this candidate
    }
    // Error for full block aka the entire candidate
    err = block_reduce_sum(err, s_red);
    // Write out error per candidate to global memory
    if (threadIdx.x == 0) d_score[blockIdx.x] = err / total; // sq error / total = MSE
}

// Prefix sum, then the first bin reaching each q[]. Launch: <<<1, 1024>>>
// Finds the bin where a fixed fraction of values has been counted.
//   1. Load the bins. Thread t owns bins 2t and 2t + 1.
//   2. Build the running total with cub::BlockScan.
//   3. __syncthreads.
//   4. For each q, target = q * total.
//   5. The thread holding the first bin with cum[b] >= target and
//      cum[b - 1] < target writes d_bin. Only one bin can match.
// Then on the host: T = (bin + 1) * width.
// Uses the raw histogram, and needs exactly 1024 threads (2 bins each).
__global__ void percentile_kernel(const unsigned* d_hist, unsigned total, const double* q, int nq, int* d_bin) {
    using BlockScan = cub::BlockScan<unsigned, 1024>;
    __shared__ typename BlockScan::TempStorage s_scan; // CUB's scratch space
    __shared__ unsigned s_cum[NBINS];                  // running total up to and including each bin

    // 1. My two bins
    const int b0 = 2 * threadIdx.x;
    const unsigned h0 = d_hist[b0];
    const unsigned h1 = d_hist[b0 + 1];

    // 2. before = sum of every bin before b0, from all the threads with a lower number
    unsigned before = 0;
    BlockScan(s_scan).ExclusiveSum(h0 + h1, before);
    s_cum[b0] = before + h0;
    s_cum[b0 + 1] = before + h0 + h1;
    // 3. Wait for the whole running total, since step 5 reads a neighbor's bin
    __syncthreads();

    // 4, 5. The bin where the running total first reaches the target
    for (int k = 0; k < nq; ++k) {
        const double target = q[k] * total;
        for (int b = threadIdx.x; b < NBINS; b += blockDim.x) {
            const unsigned prev = (b == 0) ? 0u : s_cum[b - 1];
            if (s_cum[b] >= target && prev < target) d_bin[k] = b;
        }
    }
}

// Index of the smallest score, first index on a tie. Launch: <<<1, 1024>>>
// Run once for KL and once for MSE.
//   1. Each thread scans its share, keeping the best score and index.
//   2. Reduce each warp to one pair with shuffles, keeping the lower index on a tie.
//   3. Lane 0 writes the warp's pair to shared memory.
//   4. __syncthreads.
//   5. One thread picks the best pair and writes d_best.
// Then on the host: i = 128 + d_best, T = (i + 0.5) * width for KL, i * width for MSE.
__global__ void argmin_kernel(const float* d_score, int count, int* d_best) {
    __shared__ float s_score[32];
    __shared__ int s_pos[32];

    float min_err = INFINITY;
    int min_pos = count;
    // Each thread gets min error for errors assigned to it
    for (unsigned c = threadIdx.x; c < count; c += blockDim.x) {
        if (d_score[c] < min_err) {
            min_err = d_score[c];
            min_pos = static_cast<int>(c);
        }
    }

    // Reduce find min MSE and position
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other_err = __shfl_down_sync(0xffffffff, min_err, offset);
        const int other_pos = __shfl_down_sync(0xffffffff, min_pos, offset);
        // if other thread has lower MSE or has same but lower position
        if (other_err < min_err || (other_err == min_err && other_pos < min_pos)) {
            min_err = other_err;
            min_pos = other_pos;
        }
    }

    const unsigned warp_th_idx = threadIdx.x % 32;
    const unsigned warp_idx = threadIdx.x / 32;
    // Thread 0 per warp writes to shared memory for each warp
    if (warp_th_idx == 0) {
        s_score[warp_idx] = min_err;
        s_pos[warp_idx] = min_pos;
    }
    __syncthreads();

    // Thread 0 get min from shared scores
    if (threadIdx.x == 0) {
        for (int t = 1; t < blockDim.x/32; ++t) {
            if (s_score[t] < min_err || (s_score[t] == min_err && s_pos[t] < min_pos)) {
                min_err = s_score[t];
                min_pos = s_pos[t];
            }
        }
        *d_best = min_pos;
    }

}

static void cuda_check(const char* what) {
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Smoothing weights: a bell curve with sigma = radius / 2, adding up to 1. radius 0 gives {1}.
static std::vector<float> gaussian_weights(int radius) {
    const float sigma = static_cast<float>(radius) / 2.0f;
    std::vector<float> weights(2 * radius + 1);
    float weight_sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const float weight = (radius == 0) ? 1.0f :
            std::exp(-0.5f * (static_cast<float>(k) / sigma) * (static_cast<float>(k) / sigma));
        weights[k + radius] = weight;
        weight_sum += weight;
    }
    for (float& w : weights) w /= weight_sum; // normalize
    return weights;
}

// CPU version: the same steps on one thread. It checks every kernel and is the
// one-worker time for speedup.

static float cpu_absmax(const std::vector<float>& x) {
    float m = 0.0f;
    for (const float v : x) m = std::fmax(m, std::fabs(v));
    return m;
}

static std::vector<unsigned> cpu_histogram(const std::vector<float>& x, float bins_per_unit) {
    std::vector<unsigned> hist(NBINS, 0);
    for (const float v : x) {
        int bin = static_cast<int>(std::fabs(v * bins_per_unit));
        if (bin > NBINS - 1) bin = NBINS - 1;
        hist[bin]++;
    }
    return hist;
}

static std::vector<float> cpu_smooth(const std::vector<unsigned>& hist, const std::vector<float>& weights) {
    const int radius = static_cast<int>(weights.size() - 1) / 2;
    std::vector<float> smooth(NBINS);
    smooth[0] = static_cast<float>(hist[0]);
    for (int b = 1; b < NBINS; ++b) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int j = b + k;
            if (j < 1) j = 1 - j;
            if (j > NBINS - 1) j = 2 * NBINS - 1 - j;
            sum += weights[k + radius] * static_cast<float>(hist[j]);
        }
        smooth[b] = sum;
    }
    return smooth;
}

static std::vector<float> cpu_mse(const std::vector<float>& smooth, float width, size_t n) {
    std::vector<float> score(NCANDIDATES);
    for (int cand = 0; cand < NCANDIDATES; ++cand) {
        const float s = static_cast<float>(128 + cand) * width / 127.0f;
        double err = 0;
        for (int j = 0; j < NBINS; ++j) {
            const float c = (static_cast<float>(j) + 0.5f) * width;
            const float d = c - s * std::fmin(std::round(c / s), 127.0f);
            err += smooth[j] * static_cast<double>(d) * d;
        }
        score[cand] = static_cast<float>(err / static_cast<double>(n));
    }
    return score;
}

// Groups come from j * 128 / i here, which checks the kernel's lo/hi bounds.
static std::vector<float> cpu_kl(const std::vector<float>& smooth) {
    std::vector<float> h = smooth;
    h[0] = h[1]; // the zero fix
    std::vector<float> score(NCANDIDATES);
    for (int cand = 0; cand < NCANDIDATES; ++cand) {
        const int i = 128 + cand;
        double outliers = 0;
        for (int j = i; j < NBINS; ++j) outliers += h[j];
        double gsum[128] = {};
        int gnz[128] = {};
        for (int j = 0; j < i; ++j) {
            gsum[j * 128 / i] += h[j];
            gnz[j * 128 / i] += (h[j] != 0);
        }
        double q_total = 0;
        for (int g = 0; g < 128; ++g) q_total += gsum[g];
        const double total = outliers + q_total;
        double sum = 0;
        for (int j = 0; j < i; ++j) {
            const double p = (h[j] + (j == i - 1 ? outliers : 0)) / total;
            if (p == 0) continue;
            double q = h[j] != 0 ? gsum[j * 128 / i] / gnz[j * 128 / i] / q_total : 0;
            if (q == 0) q = 1e-4;
            sum += p * std::log(p / q);
        }
        score[cand] = static_cast<float>(sum);
    }
    return score;
}

// Lowest score, first one on a tie.
static int cpu_argmin(const std::vector<float>& score) {
    int best = 0;
    for (int c = 1; c < static_cast<int>(score.size()); ++c) if (score[c] < score[best]) best = c;
    return best;
}

// Walk the running total until it reaches each target.
static std::vector<int> cpu_percentile(const std::vector<unsigned>& hist, size_t n, const std::vector<double>& qs) {
    std::vector<int> bins;
    for (const double q : qs) {
        const double target = q * static_cast<double>(n);
        int b = 0;
        unsigned long long cum = hist[0];
        while (cum < target) cum += hist[++b];
        bins.push_back(b);
    }
    return bins;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: calib FILE [--pageable]\n"); return 2; }
    const bool pinned = !(argc > 2 && std::strcmp(argv[2], "--pageable") == 0);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("%s: %d SMs, shared memory %zu KB (opt-in %zu KB)\n", prop.name,
                prop.multiProcessorCount, prop.sharedMemPerBlock / 1024,
                prop.sharedMemPerBlockOptin / 1024);

    // Read activations tensors from file
    std::vector<float> act = read_tensor(argv[1]);
    const size_t n = act.size();
    const size_t bytes = n * sizeof(float);

    // Settings
    const int grid = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / 256); // fills every SM once, 660 here
    constexpr int radius = 2; // smoothing, 0 = off
    const std::vector<float> weights = gaussian_weights(radius);
    const std::vector<double> qs = {0.999, 0.9999, 0.99999, 0.999999};
    const int nq = static_cast<int>(qs.size());
    constexpr int GPU_RUNS = 10; // timed runs, after one warm-up run
    constexpr int CPU_RUNS = 3;

    // init GPU memory is made before timing
    float* d_tensor = nullptr;
    int* d_max = nullptr;
    unsigned* d_hist = nullptr;        // the pipeline's histogram (shared + zero trick)
    unsigned* d_hist_global = nullptr; // variant: global atomics only
    unsigned* d_hist_plain = nullptr;  // variant: shared, no zero trick
    float* d_weights = nullptr;
    float* d_smooth = nullptr;
    float* d_mse = nullptr;
    float* d_kl = nullptr;
    int* d_best = nullptr;
    double* d_q = nullptr;
    int* d_pbins = nullptr;
    cudaMalloc(&d_tensor, bytes);
    cudaMalloc(&d_max, sizeof(int));
    cudaMalloc(&d_hist, NBINS * sizeof(unsigned));
    cudaMalloc(&d_hist_global, NBINS * sizeof(unsigned));
    cudaMalloc(&d_hist_plain, NBINS * sizeof(unsigned));
    cudaMalloc(&d_weights, weights.size() * sizeof(float));
    cudaMalloc(&d_smooth, NBINS * sizeof(float));
    cudaMalloc(&d_mse, NCANDIDATES * sizeof(float));
    cudaMalloc(&d_kl, NCANDIDATES * sizeof(float));
    cudaMalloc(&d_best, 2 * sizeof(int));
    cudaMalloc(&d_q, nq * sizeof(double));
    cudaMalloc(&d_pbins, nq * sizeof(int));
    // Writing a buffer twice the L2 size before each run pushes the tensor out of the
    // cache. Without it, runs after the first read x from L2 and look faster than memory allows.
    const size_t flush_bytes = 2 * static_cast<size_t>(prop.l2CacheSize);
    void* d_flush = nullptr;
    cudaMalloc(&d_flush, flush_bytes);
    cuda_check("cudaMalloc");
    std::printf("L2 cache %d MB\n", prop.l2CacheSize >> 20);

    // Pin act in place so the GPU copies straight from it. Pinning costs time once;
    // compare pin + copy against a --pageable run to see whether it pays off.
    double start = now();
    if (pinned) cudaHostRegister(act.data(), bytes, cudaHostRegisterDefault);
    const double pin_time = now() - start;

    // Copy tensor to gpu
    start = now();
    cudaMemcpy(d_tensor, act.data(), bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const double copy_time = now() - start;
    cuda_check("move tensors");
    std::printf("upload %.1f MB %s: pin %.6f s, copy %.6f s (%.1f GB/s)\n", bytes / 1e6,
                pinned ? "pinned" : "pageable", pin_time, copy_time, bytes / copy_time / 1e9);
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, qs.data(), nq * sizeof(double), cudaMemcpyHostToDevice);

    // GPU pipeline. Events are timestamps the GPU records itself, so ev[s] to ev[s + 1]
    // is the GPU time of step s with no CPU overhead mixed in.
    constexpr int NSTEPS = 8;
    const char* step_names[NSTEPS] = {"absmax", "histogram", "smooth", "mse", "argmin mse",
                                      "kl", "argmin kl", "percentile"};
    cudaEvent_t ev[NSTEPS + 1];
    for (cudaEvent_t& e : ev) cudaEventCreate(&e);
    double gpu_ms[NSTEPS] = {};
    float maxVal = 0.0f, bins_per_unit = 0.0f, width = 0.0f;

    for (int run = 0; run <= GPU_RUNS; ++run) { // run 0 is for warmup only
        cudaMemset(d_flush, run, flush_bytes);  // start cold, outside the timed part
        cudaEventRecord(ev[0]);
        cudaMemset(d_max, 0, sizeof(int));
        absmax_kernel<<<grid, 256>>>(d_tensor, n, d_max);
        // The CPU needs m for the bucket math, so this copy is part of the algorithm
        cudaMemcpy(&maxVal, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        bins_per_unit = NBINS / maxVal;
        width = maxVal / NBINS;
        cudaEventRecord(ev[1]);

        cudaMemset(d_hist, 0, NBINS * sizeof(unsigned));
        hist_kernel<true><<<grid, 256>>>(d_tensor, n, bins_per_unit, d_hist);
        cudaEventRecord(ev[2]);

        smooth_kernel<<<2, 1024>>>(d_hist, d_weights, radius, d_smooth);
        cudaEventRecord(ev[3]);

        mse_kernel<<<NCANDIDATES, 256>>>(d_smooth, width, static_cast<float>(n), d_mse);
        cudaEventRecord(ev[4]);
        argmin_kernel<<<1, 1024>>>(d_mse, NCANDIDATES, d_best);
        cudaEventRecord(ev[5]);

        kl_kernel<<<NCANDIDATES, 256>>>(d_smooth, d_kl);
        cudaEventRecord(ev[6]);
        argmin_kernel<<<1, 1024>>>(d_kl, NCANDIDATES, d_best + 1);
        cudaEventRecord(ev[7]);

        percentile_kernel<<<1, 1024>>>(d_hist, static_cast<unsigned>(n), d_q, nq, d_pbins);
        cudaEventRecord(ev[8]);

        // Wait for the last event, then read the gaps between events
        cudaEventSynchronize(ev[NSTEPS]);
        cuda_check("GPU pipeline");
        if (run == 0) continue;
        for (int s = 0; s < NSTEPS; ++s) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev[s], ev[s + 1]);
            gpu_ms[s] += ms / GPU_RUNS;
        }
    }

    // Histogram variants for the contention comparison, timed the same way but not part
    // of the pipeline. Each includes clearing its output.
    double variant_ms[3] = {}; // global atomics, shared, shared + zero trick
    for (int run = 0; run <= GPU_RUNS; ++run) {
        // Variant v is timed from ev[2v] to ev[2v + 1]; the flushes between pairs aren't timed
        cudaMemset(d_flush, run, flush_bytes);
        cudaEventRecord(ev[0]);
        cudaMemset(d_hist_global, 0, NBINS * sizeof(unsigned));
        hist_global_kernel<<<grid, 256>>>(d_tensor, n, bins_per_unit, d_hist_global);
        cudaEventRecord(ev[1]);
        cudaMemset(d_flush, run, flush_bytes);
        cudaEventRecord(ev[2]);
        cudaMemset(d_hist_plain, 0, NBINS * sizeof(unsigned));
        hist_kernel<false><<<grid, 256>>>(d_tensor, n, bins_per_unit, d_hist_plain);
        cudaEventRecord(ev[3]);
        cudaMemset(d_flush, run, flush_bytes);
        cudaEventRecord(ev[4]);
        cudaMemset(d_hist, 0, NBINS * sizeof(unsigned)); // same counts as the pipeline wrote
        hist_kernel<true><<<grid, 256>>>(d_tensor, n, bins_per_unit, d_hist);
        cudaEventRecord(ev[5]);

        cudaEventSynchronize(ev[5]);
        cuda_check("histogram variants");
        if (run == 0) continue;
        for (int v = 0; v < 3; ++v) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev[2 * v], ev[2 * v + 1]);
            variant_ms[v] += ms / GPU_RUNS;
        }
    }

    // CPU pipeline, timed per step with the host clock. Each step uses the CPU's own
    // results, so this is the whole algorithm on one thread.
    double cpu_ms[NSTEPS] = {};
    float cpu_max = 0.0f;
    std::vector<unsigned> cpu_hist;
    std::vector<float> cpu_sm, cpu_mse_score, cpu_kl_score;
    int cpu_best_mse = 0, cpu_best_kl = 0;
    std::vector<int> cpu_pbins;
    for (int run = 0; run < CPU_RUNS; ++run) {
        double t = now();
        auto lap = [&](int s) { const double t2 = now(); cpu_ms[s] += (t2 - t) * 1e3 / CPU_RUNS; t = t2; };
        cpu_max = cpu_absmax(act);                                     lap(0);
        cpu_hist = cpu_histogram(act, NBINS / cpu_max);                lap(1);
        cpu_sm = cpu_smooth(cpu_hist, weights);                        lap(2);
        cpu_mse_score = cpu_mse(cpu_sm, cpu_max / NBINS, n);           lap(3);
        cpu_best_mse = cpu_argmin(cpu_mse_score);                      lap(4);
        cpu_kl_score = cpu_kl(cpu_sm);                                 lap(5);
        cpu_best_kl = cpu_argmin(cpu_kl_score);                        lap(6);
        cpu_pbins = cpu_percentile(cpu_hist, n, qs);                   lap(7);
    }

    // Copy the GPU results back
    std::vector<unsigned> hist(NBINS), hist_global(NBINS), hist_plain(NBINS);
    std::vector<float> smooth(NBINS), mse(NCANDIDATES), kl(NCANDIDATES);
    int best[2] = {};
    std::vector<int> pbins(nq);
    cudaMemcpy(hist.data(), d_hist, NBINS * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(hist_global.data(), d_hist_global, NBINS * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(hist_plain.data(), d_hist_plain, NBINS * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(smooth.data(), d_smooth, NBINS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mse.data(), d_mse, NCANDIDATES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(kl.data(), d_kl, NCANDIDATES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best, d_best, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pbins.data(), d_pbins, nq * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check("copy results");

    // Number of entries where GPU and CPU differ by more than abs_tol + rel * |cpu|.
    // Counts must match exactly; float sums are added in a different order, so allow a little.
    auto count_differ = [](const auto& gpu, const auto& cpu, double rel, double abs_tol) {
        int wrong = 0;
        for (size_t k = 0; k < gpu.size(); ++k) {
            const double g = gpu[k], c = cpu[k];
            if (std::fabs(g - c) > abs_tol + rel * std::fabs(c)) wrong++;
        }
        return wrong;
    };
    std::printf("\nchecks against the CPU:\n");
    std::printf("  max: GPU %f, CPU %f\n", maxVal, cpu_max);
    std::printf("  histogram: %d / %d / %d buckets differ (global / shared / zero trick)\n",
                count_differ(hist_global, cpu_hist, 0, 0), count_differ(hist_plain, cpu_hist, 0, 0),
                count_differ(hist, cpu_hist, 0, 0));
    std::printf("  smooth: %d buckets differ\n", count_differ(smooth, cpu_sm, 1e-3, 1e-3));
    std::printf("  mse: %d candidates differ\n", count_differ(mse, cpu_mse_score, 1e-3, 0));
    std::printf("  kl: %d candidates differ\n", count_differ(kl, cpu_kl_score, 1e-3, 1e-5));
    // argmin is checked on the GPU's own scores, since near-ties can rank differently on the CPU's
    std::printf("  argmin: MSE bucket %d (expected %d), KL bucket %d (expected %d)\n",
                128 + best[0], 128 + cpu_argmin(mse), 128 + best[1], 128 + cpu_argmin(kl));
    std::printf("  percentile: %d of %d buckets differ\n", count_differ(pbins, cpu_pbins, 0, 0), nq);

    std::printf("\ncutoffs:\n");
    std::printf("  max: T = %f\n", maxVal);
    std::printf("  MSE: bucket %d, T = %f (CPU scores: bucket %d)\n",
                128 + best[0], static_cast<float>(128 + best[0]) * width, 128 + cpu_best_mse);
    std::printf("  KL: bucket %d, T = %f (CPU scores: bucket %d)\n",
                128 + best[1], (static_cast<float>(128 + best[1]) + 0.5f) * width, 128 + cpu_best_kl);
    for (int k = 0; k < nq; ++k)
        std::printf("  percentile %.4f%%: bucket %d, T = %f\n", qs[k] * 100, pbins[k],
                    static_cast<float>(pbins[k] + 1) * width);

    std::printf("\ntiming: GPU average of %d runs after 1 warm-up, CPU average of %d runs\n",
                GPU_RUNS, CPU_RUNS);
    std::printf("  %-12s %10s %10s %9s\n", "step", "GPU ms", "CPU ms", "speedup");
    double gpu_total = 0, cpu_total = 0;
    for (int s = 0; s < NSTEPS; ++s) {
        std::printf("  %-12s %10.4f %10.3f %8.1fx\n", step_names[s], gpu_ms[s], cpu_ms[s],
                    cpu_ms[s] / gpu_ms[s]);
        gpu_total += gpu_ms[s];
        cpu_total += cpu_ms[s];
    }
    std::printf("  %-12s %10.4f %10.3f %8.1fx\n", "total", gpu_total, cpu_total, cpu_total / gpu_total);
    std::printf("  histogram variants (GPU ms): global atomics %.4f, shared %.4f, shared + zero trick %.4f\n",
                variant_ms[0], variant_ms[1], variant_ms[2]);

    for (cudaEvent_t& e : ev) cudaEventDestroy(e);
    cudaFree(d_tensor);
    cudaFree(d_max);
    cudaFree(d_hist);
    cudaFree(d_hist_global);
    cudaFree(d_hist_plain);
    cudaFree(d_weights);
    cudaFree(d_smooth);
    cudaFree(d_mse);
    cudaFree(d_kl);
    cudaFree(d_best);
    cudaFree(d_q);
    cudaFree(d_pbins);
    cudaFree(d_flush);
    if (pinned) cudaHostUnregister(act.data());
    return 0;
}
