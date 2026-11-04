// P1's one-thread baseline, T(1). Compiled by the host compiler, not nvcc, so it gets the
// same code as P2 with one rank.
#include "calib_common.h"

CpuResult cpu_pipeline(const float* x, size_t n, const std::vector<float>& weights,
                       const std::vector<double>& qs, int runs) {
    CpuResult r;
    for (int run = 0; run <= runs; ++run) { // run 0 is a warm-up, as on the GPU
        double t = now();
        auto lap = [&](int s) {
            const double t2 = now();
            if (run > 0) r.ms[s] += (t2 - t) * 1e3 / runs;
            t = t2;
        };
        r.max = cpu_absmax(x, n);                                lap(0);
        r.hist = cpu_histogram(x, n, NBINS / r.max);             lap(1);
        r.smooth = cpu_smooth(r.hist, weights);                  lap(2);
        r.mse = cpu_mse(r.smooth, r.max / NBINS, n);             lap(3);
        r.best_mse = cpu_argmin(r.mse);                          lap(4);
        r.kl = cpu_kl(r.smooth);                                 lap(5);
        r.best_kl = cpu_argmin(r.kl);                            lap(6);
        r.pbins = cpu_percentile(r.hist, n, qs);                 lap(7);
    }
    return r;
}
