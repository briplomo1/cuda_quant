// Algorithm 1 host code shared by P1 (cuda/calib.cu) and P2 (mpi/calib_mpi.cpp): the file
// format, the timer, the smoothing weights, and the one-thread version of every step.
// P1 uses the one-thread version to check its kernels and as T(1). P2 runs the same
// functions on its share of the data, so both programs score candidates identically.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NBINS 2048
#define LEVELS 128 // Using int8 quantization
#define NCANDIDATES 1921 // i = 128 .. 2048

// Percentiles to report, as fractions
inline const std::vector<double> PERCENTILES = {0.999, 0.9999, 0.99999, 0.999999};

// File format: int32 ndim, int32 dims, then raw float32. dims[0] is the number of images.
// Reads images first .. first + images - 1, or every image from first on when images is 0.
inline std::vector<float> read_tensor(const char* path, size_t first, size_t images, size_t& file_images) {
    // Read activations from file
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    int32_t ndim = 0, dim = 0;
    size_t per_image = 1;
    bool ok = std::fread(&ndim, sizeof(ndim), 1, f) == 1 && ndim >= 1;
    for (int i = 0; ok && i < ndim; ++i) {
        ok = std::fread(&dim, sizeof(dim), 1, f) == 1;
        if (i == 0) file_images = dim; else per_image *= dim;
    }
    if (!ok) { std::fprintf(stderr, "bad header: %s\n", path); std::exit(1); }
    if (images == 0 && first < file_images) images = file_images - first;
    if (images == 0 || first + images > file_images) {
        std::fprintf(stderr, "%s has %zu images\n", path, file_images);
        std::exit(1);
    }
    // Skip the header and the images before first
    std::fseek(f, static_cast<long>(sizeof(int32_t) * (1 + ndim) + first * per_image * sizeof(float)), SEEK_SET);
    const size_t count = images * per_image;
    std::vector<float> data(count);
    if (std::fread(data.data(), sizeof(float), count, f) != count) {
        std::fprintf(stderr, "short read: %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return data;
}

inline double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Smoothing weights: a bell curve with sigma = radius / 2, adding up to 1. radius 0 gives {1}.
inline std::vector<float> gaussian_weights(int radius) {
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

// One-thread version of each step.

// a > m ? a : m instead of std::fmax: fmax's NaN rules stop the compiler from using
// vector instructions, which made this loop 2x slower and the CPU baseline unfair.
inline float cpu_absmax(const float* x, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        m = a > m ? a : m;
    }
    return m;
}

inline std::vector<unsigned> cpu_histogram(const float* x, size_t n, float bins_per_unit) {
    std::vector<unsigned> hist(NBINS, 0);
    for (size_t i = 0; i < n; ++i) {
        int bin = static_cast<int>(std::fabs(x[i] * bins_per_unit));
        if (bin > NBINS - 1) bin = NBINS - 1;
        hist[bin]++;
    }
    return hist;
}

inline std::vector<float> cpu_smooth(const std::vector<unsigned>& hist, const std::vector<float>& weights) {
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

// MSE of cutting at bucket i (128 .. 2048), T = i * width.
inline float mse_score(const std::vector<float>& smooth, float width, size_t n, int i) {
    const float s = static_cast<float>(i) * width / 127.0f;
    double err = 0;
    for (int j = 0; j < NBINS; ++j) {
        const float c = (static_cast<float>(j) + 0.5f) * width;
        const float d = c - s * std::fmin(std::round(c / s), 127.0f);
        err += smooth[j] * static_cast<double>(d) * d;
    }
    return static_cast<float>(err / static_cast<double>(n));
}

// The histogram KL scores: bucket 0 replaced by bucket 1 (the zero fix).
inline std::vector<float> kl_hist(const std::vector<float>& smooth) {
    std::vector<float> h = smooth;
    h[0] = h[1];
    return h;
}

// KL of cutting at bucket i, on h from kl_hist. Groups come from j * 128 / i here,
// which checks the kernel's lo/hi bounds.
inline float kl_score(const std::vector<float>& h, int i) {
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
    return static_cast<float>(sum);
}

inline std::vector<float> cpu_mse(const std::vector<float>& smooth, float width, size_t n) {
    std::vector<float> score(NCANDIDATES);
    for (int cand = 0; cand < NCANDIDATES; ++cand) score[cand] = mse_score(smooth, width, n, 128 + cand);
    return score;
}

inline std::vector<float> cpu_kl(const std::vector<float>& smooth) {
    const std::vector<float> h = kl_hist(smooth);
    std::vector<float> score(NCANDIDATES);
    for (int cand = 0; cand < NCANDIDATES; ++cand) score[cand] = kl_score(h, 128 + cand);
    return score;
}

// Lowest score, first one on a tie.
inline int cpu_argmin(const std::vector<float>& score) {
    int best = 0;
    for (int c = 1; c < static_cast<int>(score.size()); ++c) if (score[c] < score[best]) best = c;
    return best;
}

// Walk the running total until it reaches each target.
inline std::vector<int> cpu_percentile(const std::vector<unsigned>& hist, size_t n, const std::vector<double>& qs) {
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

// The whole algorithm on one thread, every output kept, each step timed (average ms of
// `runs` runs after one warm-up). Defined in calib_cpu.cpp, compiled by the host compiler:
// built through nvcc, the same loops ran up to 2x slower, which made P1's T(1) unfair.
struct CpuResult {
    float max = 0.0f;
    std::vector<unsigned> hist;
    std::vector<float> smooth, mse, kl;
    int best_mse = 0, best_kl = 0;
    std::vector<int> pbins;
    double ms[8] = {}; // absmax, histogram, smooth, mse, argmin_mse, kl, argmin_kl, percentile
};
CpuResult cpu_pipeline(const float* x, size_t n, const std::vector<float>& weights,
                       const std::vector<double>& qs, int runs);

// The thresholds every run prints on one "result:" line, so scripts can grep them.
// Bucket i -> T: MSE i * width, KL (i + 0.5) * width, percentile (bin + 1) * width.
inline void print_result(int radius, size_t first, size_t images, float maxVal, float width,
                         int mse_i, int kl_i, const std::vector<int>& pbins) {
    std::printf("result: radius %d first %zu images %zu max %f mse %f kl %f", radius, first, images,
                maxVal, static_cast<float>(mse_i) * width, (static_cast<float>(kl_i) + 0.5f) * width);
    for (size_t k = 0; k < pbins.size(); ++k)
        std::printf(" p%.6g %f", PERCENTILES[k] * 100, static_cast<float>(pbins[k] + 1) * width);
    std::printf("\n");
}
