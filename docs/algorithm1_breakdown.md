# Algorithm 1: CUDA and MPI breakdown

Both programs pick one INT8 clipping threshold T per layer. They share the same
math and differ in how the work is split up:

| | P1 (CUDA) | P2 (MPI) |
|---|---|---|
| Workers | Thousands of GPU threads in one process | p processes, one per core or node |
| Memory | One shared GPU memory | Each rank has its own private memory |
| How workers combine results | Atomics, shuffles, barriers | Messages (collectives) |
| Report section it feeds | 5, contention and synchronization | 4, communication pattern |

The pipeline for one layer is the same in both:

```
max |x|  ->  histogram  ->  smooth (optional)  ->  KL, MSE search  ->  pick the lowest score
                        ->  percentile (raw histogram)
```

The examples below use layer 01: 512 images x 16 x 32 x 32 = 8,388,608 values.

---

## Part 1: P1 (CUDA)

### Memory and layout

- **`x` is one flat array** of `n` floats. Calibration treats every value the same,
  so the file's images x channels x height x width shape doesn't matter, and every
  kernel indexes `x` with a single `i`. No pitched or 3D allocations are needed.
- **Host memory is pinned by default.** `calib.cu` pins the `std::vector` in place
  with `cudaHostRegister`, so the GPU copies straight from it instead of through a
  driver staging buffer. Pinning costs time once, so the program prints the pin and
  copy times; `./calib FILE --pageable` runs without it for comparison. The upload
  is reported separately from compute time.
- **No streams.** Each kernel needs the previous one's output, and the program
  handles one layer per run, so there is nothing to overlap. Streams would pay off
  only if one run processed all 19 layers, copying layer L + 1 while computing
  layer L.
- **Memory kinds, slowest to fastest:**
  - Global: large and visible to every thread.
  - Shared: small and private to one block.
  - Registers: private to one thread.

### Host steps before the first kernel

1. Read the file and `cudaMalloc` the buffers.
2. `cudaMemcpy` `x` to the GPU, 33.5 MB for layer 01, once per layer.
3. `cudaMemset` `d_max_bits` and `d_hist` to 0, because the kernels add into them.

### `absmax_kernel`: find m = max |x|

```cpp
__global__ void absmax_kernel(const float* x, size_t n, int* d_max_bits);
```

- **Why:** the histogram covers 0 to m, so m is needed first.
- **Launch:** `<<<grid, 256>>>`. The example grid is 320 blocks, so 81,920 threads.
- **Steps:**
  1. Walk `x` by grid-stride: start at your own index and jump 81,920 each time,
     about 102 values per thread. Keep the largest `|x|` in a register.
  2. Reduce the warp's 32 maximums to 1 with `__shfl_down_sync`.
  3. Lane 0 does `atomicMax` on `d_max_bits`. Comparing float bits as ints is safe
     because m >= 0.
- **Memory:** reads `x` from global memory once and writes one int.
- **Combining:** 2,560 atomics, one per warp, instead of 8.4 million.
- **Waiting:** none inside the kernel. Shuffles need no barrier.
- **Host after:** copy the 4 bytes back, which also makes the CPU wait for the GPU,
  and set `inv_width = 2048 / m`.

### `hist_global_kernel`: the contention baseline

```cpp
__global__ void hist_global_kernel(const float* x, size_t n, float inv_width,
                                   unsigned* d_hist);
```

- **Why:** the result is not needed, since `hist_kernel` produces the same
  histogram faster. This version has as much contention as possible, so it is the
  baseline for measuring the amount of contention in the report. It is also the
  simplest correct histogram, so `hist_kernel` is checked against it bin for bin.
- **Launch:** `<<<grid, 256>>>`.
- **Steps:**
  1. Walk `x` by grid-stride.
  2. `bin = min(int(|x| * inv_width), 2047)`.
  3. `atomicAdd(&d_hist[bin], 1)` in global memory.
- **Memory:** reads `x`, then one global atomic per value.
- **Combining:** 8.4 million global atomics, about half of them on bin 0.
- **Waiting:** threads queue behind each other at busy bins. This is the contention
  being measured.

### `hist_kernel<false>` and `hist_kernel<true>`: the real histogram

```cpp
template <bool AGGREGATE_ZEROS>
__global__ void hist_kernel(const float* x, size_t n, float inv_width,
                            unsigned* d_hist);
```

- **Why:** a private copy per block moves almost all collisions into fast shared
  memory.
- **Launch:** `<<<grid, 256>>>`.
- **Steps:**
  1. Threads zero the block's `s_hist[2048]`.
  2. `__syncthreads`.
  3. Walk `x` by grid-stride, compute the bin, and `atomicAdd` on `s_hist[bin]`.
     - `<true>` only: the warp gets its zero lanes from
       `__ballot_sync(__activemask(), a == 0)`, and one zero lane adds the whole
       count with `__popc`. This is one atomic on bin 0 instead of up to 32.
  4. `__syncthreads`.
  5. Threads `atomicAdd` each non-empty `s_hist` bin into `d_hist`.
- **Memory:** reads `x`; counts into `s_hist`, 8 KB of shared memory per block;
  at most 2,048 global atomics per block for the merge.
- **Combining:** collisions only within a block, in shared memory, then one merge.
- **Waiting:** 2 barriers per block.

### `smooth_kernel` (optional)

```cpp
__global__ void smooth_kernel(const unsigned* d_hist, const float* d_weights,
                              int radius, float* d_smooth);
```

- **Why:** turns random spikes and gaps in the sparse tail into a slope, so the
  search is less sensitive to which images were used.
- **Launch:** `<<<2, 1024>>>`, one thread per bin.
- **Steps:**
  1. `b = blockIdx.x * 1024 + threadIdx.x`.
  2. Bin 0: copy it and stop. The ReLU spike is real and must not spread.
  3. Otherwise sum `weight[k] * hist[b + k]` for k = -radius to radius. An index
     below 1 reads bin `1 - j`, and one above 2047 reads bin `4095 - j`, which
     keeps the total count unchanged.
  4. Write `d_smooth[b]`.
- **Memory:** reads `d_hist` and the weights, writes `d_smooth`. All tiny.
- **Combining and waiting:** none.
- **Note:** `radius = 0` with a weight of 1 gives the unsmoothed run, so KL and MSE
  always read `d_smooth`.

### `block_reduce_sum`: helper used inside KL and MSE

```cpp
__device__ float block_reduce_sum(float value, float* s_red);
```

- **Why:** KL and MSE need the total of one number from every thread in a block.
- **Steps:**
  1. Add the warp's 32 values with shuffles. Only lane 0 has the right sum.
  2. Lane 0 writes it to `s_red[warp]`.
  3. `__syncthreads`.
  4. Every thread adds up `s_red` and returns the block total.
- **Memory:** `s_red[32]` in shared memory.
- **Waiting:** 1 barrier per call.

### `kl_kernel`: score each candidate by shape

```cpp
__global__ void kl_kernel(const float* d_smooth, float* d_score);
```

- **Why:** measures how much information is lost by clipping at bin i and merging
  into 128 levels.
- **Launch:** `<<<1921, 256>>>`, one block per candidate `i = 128 + blockIdx.x`.
  Every bin read uses bin 1's value for bin 0, the zero-spike fix.
- **Steps:**
  1. `outliers` = sum of bins i to 2047, using `block_reduce_sum`.
  2. Threads 0 to 127 each own one group. Bin j is in group `j * 128 / i`. Sum the
     group's bins into `s_gsum` and count its non-empty bins into `s_gnz`.
  3. `__syncthreads`.
  4. `q_total` = block sum of `s_gsum`, and `total = outliers + q_total`.
  5. All threads loop over bins j < i, about i / 256 each:
     - `p = bin j / total`, and bin i - 1 also gets the outliers. Skip if p == 0.
     - `q = 0` for an empty bin, else `s_gsum[g] / s_gnz[g] / q_total`.
     - If q == 0, use eps. Add `p * log(p / q)`.
  6. `block_reduce_sum`, then thread 0 writes `d_score[blockIdx.x]`.
- **Memory:** every block reads the same 8 KB `d_smooth`, which stays in cache;
  about 1 KB of shared memory for `s_gsum`, `s_gnz` and `s_red`; one float out.
- **Combining:** within the block only. Blocks never need each other, and the GPU
  starts new blocks as earlier ones finish.
- **Waiting:** 3 to 4 barriers per block.

### `mse_kernel`: score each candidate by error

```cpp
__global__ void mse_kernel(const float* d_smooth, float width, float total,
                           float* d_score);
```

- **Why:** measures the average squared rounding and clipping error at each T.
- **Launch:** `<<<1921, 256>>>`, with `T = i * width` and `s = T / 127`.
- **Steps:**
  1. Each thread handles 8 of the 2,048 bins:
     - `c = (j + 0.5) * width`, the bin center.
     - `qv = s * min(round(c / s), 127)`.
     - `err += bin j * (c - qv)^2`.
  2. `block_reduce_sum`, then divide by `total`.
  3. Thread 0 writes `d_score[blockIdx.x]`.
- **Memory:** reads `d_smooth` from cache, `s_red` in shared memory, one float out.
- **Waiting:** 1 to 2 barriers per block.

### `percentile_kernel`: keep a fixed fraction

```cpp
__global__ void percentile_kernel(const unsigned* d_hist, unsigned total,
                                  const double* q, int nq, int* d_bin);
```

- **Why:** finds the bin where, say, 99.99% of values have been counted. It reads
  the raw histogram.
- **Launch:** `<<<1, 1024>>>`, a single block, 2 bins per thread.
- **Steps:**
  1. Load the bins into shared memory.
  2. Build the running total with a parallel scan, such as `cub::BlockScan`.
  3. `__syncthreads`.
  4. For each q, `target = q * total`.
  5. The thread holding the first bin with `cum[b] >= target` and
     `cum[b - 1] < target` writes `d_bin`. Only one bin can match.
- **Memory:** 8 KB of shared memory for the running total.
- **Waiting:** about 11 barriers in a simple scan, one per doubling step.
- **Host after:** `T = (bin + 1) * width`.

### `argmin_kernel`: pick the winner

```cpp
__global__ void argmin_kernel(const float* d_score, int count, int* d_best);
```

- **Why:** finds the lowest of the 1,921 scores. Run once for KL and once for MSE.
- **Launch:** `<<<1, 1024>>>`, about 2 scores per thread.
- **Steps:**
  1. Each thread keeps its best score and index.
  2. Reduce each warp to one pair with shuffles, keeping the lower index on a tie.
  3. Lane 0 writes the warp's pair to shared memory.
  4. `__syncthreads`.
  5. One thread picks the best pair and writes `d_best`.
- **Waiting:** 1 barrier.

### Host steps after the last kernel

Copy back `d_best` and `d_bin`, then compute each threshold with `w = m / 2048`:

| Method | T |
|---|---|
| Max | m |
| Percentile | (bin + 1) * w |
| MSE | i * w, with i = 128 + d_best |
| KL | (i + 0.5) * w |

Repeat for the next layer.

### CUDA summary

| Kernel | Launch | Work per thread | Main memory | Combining | Barriers |
|---|---|---|---|---|---|
| `absmax_kernel` | grid x 256 | about n / threads values | `x` (global) | Warp shuffle + atomicMax | 0 |
| `hist_global_kernel` | grid x 256 | about n / threads values | `x`, `d_hist` (global) | One global atomic per value | 0 |
| `hist_kernel` | grid x 256 | about n / threads values | `s_hist`, 8 KB (shared) | Shared atomics, then merge | 2 |
| `smooth_kernel` | 2 x 1,024 | 1 bin | `d_hist` (global) | None | 0 |
| `kl_kernel` | 1,921 x 256 | about i / 256 bins | `d_smooth` (cache), about 1 KB shared | `block_reduce_sum` | 3 to 4 |
| `mse_kernel` | 1,921 x 256 | 8 bins | `d_smooth` (cache) | `block_reduce_sum` | 1 to 2 |
| `percentile_kernel` | 1 x 1,024 | 2 bins | 8 KB shared | Scan | about 11 |
| `argmin_kernel` | 1 x 1,024 | about 2 scores | 32 pairs (shared) | Warp shuffle + 1 thread | 1 |

Each kernel also waits for the previous one to finish, about 5 to 10 us per
boundary, and the CPU waits once per layer to read m.

### Speedup on the GPU

- **T(1):** the one-thread CPU loop in `calib.cu`.
- **T(grid):** GPU time with a given number of blocks.
- **Speedup:** T(1) / T(grid). Grid-stride loops stay correct at any grid size, so
  sweep 1, 2, 4, ... blocks up to a full GPU. The search kernels always use 1,921
  blocks, so they don't change with the grid sweep and should be timed on their own.

---

## Part 2: P2 (MPI)

### The idea

Each rank is an ordinary single-threaded program with its own memory. Ranks never
touch each other's data, so there are no atomics and no write contention. Instead,
ranks combine results by sending messages. Every rank runs the same code on its own
share of the data, which is the SPMD (single program, multiple data) model.

### Data distribution

| What | How it is split | Why |
|---|---|---|
| Calibration images | Rank r owns images `r * N / p` to `(r + 1) * N / p` | Images come first in the file, so each share is one contiguous slice |
| Histogram | Each rank builds a local one, then the allreduce gives every rank the full sum | 2,048 bins is small enough to send |
| Candidates | Cyclic: candidate c goes to rank `c mod p` | KL's cost grows with i, so a block split would give the last rank the most expensive candidates |
| Percentile | Every rank computes it | Scanning 2,048 bins is cheaper than a message |

### Steps per layer

Each step mirrors a CUDA kernel. The CUDA name is in brackets.

#### 1. Read my slice (no CUDA equivalent; P1 copies the whole layer)

- **Why:** each rank needs only its own images, and reading them in parallel avoids
  sending data around.
- **Steps:**
  1. `MPI_File_open` the layer file with `MPI_MODE_RDONLY`.
  2. Read the header at offset 0: `ndim`, then the dims. N = `dims[0]`, and
     values per image = the product of the other dims.
  3. `first = r * N / p`, `count = ((r + 1) * N / p - first) * per_image`.
  4. `MPI_File_read_at` at byte offset `4 + 4 * ndim + first * per_image * 4`.
- **Communication:** none. Each rank reads its own bytes.
- **Note:** the integer formula `r * N / p` handles N not divisible by p.

#### 2. Global max [`absmax_kernel`]

- **Why:** all ranks must use the same bin width, so they must agree on m.
- **Steps:**
  1. Loop over my values and keep the largest `|x|`.
  2. `MPI_Allreduce(&local_max, &m, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD)`.
  3. `inv_w = 2048 / m`. Every rank computes the same float, so bins match.
- **Communication:** 1 float, log p rounds.
- **Waiting:** the allreduce is a synchronization point: no rank continues until
  every rank has contributed.

#### 3. Global histogram [`hist_kernel`]

- **Why:** every method chooses T from the full histogram.
- **Steps:**
  1. `uint64_t hist[2048] = {0}`.
  2. Loop over my values: `++hist[min(int(|x| * inv_w), 2047)]`.
  3. `MPI_Allreduce(hist, ghist, 2048, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD)`.
- **Communication:** 2,048 uint64 = 16 KB, the largest message, log p rounds or a
  ring for long messages.
- **Why uint64:** it is the plan's choice for headroom. uint32 would also fit here,
  since the largest layer has 67 million values at 4,096 images, well under the
  4.3 billion limit. Switching to uint32 halves the message to 8 KB.
- **Contention:** none. Each rank writes only its own private array. This is the
  distributed-memory counterpart of the GPU's private block histograms.

#### 4. Smooth (optional) [`smooth_kernel`]

- **Steps:** every rank smooths its own copy of `ghist` with the same weights and
  the same rules: bin 0 copied, reflect at the edges.
- **Communication:** none. All ranks get identical results, so nothing is sent.

#### 5. Search my candidates [`kl_kernel`, `mse_kernel`]

- **Why:** the 1,921 candidates are independent, so they split cleanly.
- **Steps:**
  1. `for (c = rank; c < 1921; c += p)`: score candidate `i = 128 + c` with the
     same KL and MSE math as the kernels, including the zero fix and eps rule.
  2. Keep my best `(score, c)` for KL and for MSE, starting from
     `(INFINITY, -1)`.
- **Communication:** none during the search.
- **Load balance:** with a cyclic split, every rank gets a mix of cheap small-i and
  expensive large-i candidates.

#### 6. Pick the winner [`argmin_kernel`]

- **Steps:**
  1. Pack `{float score; int idx;}` for KL and MSE.
  2. `MPI_Allreduce(best, gbest, 2, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD)`.
     A count of 2 sends both methods in one message.
  3. MINLOC returns the lowest score and, on a tie, the lowest index, matching the
     GPU's first-index rule.
- **Communication:** 2 pairs, log p rounds.

#### 7. Percentile [`percentile_kernel`]

- **Steps:** every rank runs the same running-total scan over `ghist` and finds the
  first bin reaching each q.
- **Communication:** none.

#### 8. Thresholds

Every rank now holds the same m, the same bins and the same winners, so every rank
can compute T. Rank 0 prints or writes them.

### Communication per layer

| Step | Call | Message | Rounds |
|---|---|---|---|
| Global max | `MPI_Allreduce`, `MPI_MAX` | 1 float | log p |
| Global histogram | `MPI_Allreduce`, `MPI_SUM` | 2,048 uint64, 16 KB | log p, or a ring |
| Winners | `MPI_Allreduce`, `MPI_MINLOC` | 2 (score, index) pairs | log p |

The histogram allreduce must wait for the max allreduce, because the bin width
depends on m. One optimization to measure is batching every layer's histogram into
a single allreduce of 19 x 2,048 words, which pays the message latency once
instead of 19 times. That also requires batching the 19 max values first.

### Synchronization and load balance

- **No contention:** ranks share nothing, so there are no atomics or barriers on
  data. The cost moves to communication.
- **Every allreduce is a synchronization point.** A slow rank makes every other
  rank wait inside the next collective. Without a barrier before timing, that
  waiting shows up as communication time even though it is really load imbalance.
- **Sources of imbalance:** an uneven image split (at most one image different) and
  uneven candidate costs, which the cyclic split addresses.

### Timing per phase

Each phase is timed with `MPI_Barrier`, then `MPI_Wtime`, and the maximum over
ranks is reported with `MPI_Reduce` and `MPI_MAX`, because the slowest rank decides
when the job ends.

| Phase | Kind |
|---|---|
| Read slice | I/O, reported separately |
| Local max | Computation |
| Max allreduce | Communication |
| Local histogram | Computation |
| Histogram allreduce | Communication |
| Search | Computation |
| MINLOC allreduce | Communication |

Computation shrinks as 1/p, and communication grows as log p:

```
T_P2(p) ~ (2n + 2048 C) t_c / p  +  log2(p) (3 t_s + 4099 t_w)
```

Here C = 1,921 candidates, t_c = time per operation, t_s = message latency and
t_w = time per 32-bit word. The 4,099 words are 1 for the max, 4,096 for the
histogram and 2 for a MINLOC pair.

### Speedup with MPI

- **T(1):** the one-thread CPU version, plus the 1-rank MPI run to show MPI's own
  overhead.
- **Strong scaling:** 512 images, p = 1, 2, 4, 8, 16, 32, and 64 if available.
- **Weak scaling:** 64 images per rank, up to 4,096 images.
- **First to flatten:** layer 19 has 32,768 values, which is only 1,024 per rank at
  p = 32. The search never shrinks with more images, only with more ranks.

---

## CUDA and MPI side by side

| Step | P1 (CUDA) | P2 (MPI) |
|---|---|---|
| Get the data | `cudaMemcpy` the whole layer to the GPU | Each rank reads its own slice with `MPI_File_read_at` |
| Max | `absmax_kernel`: warp shuffles + `atomicMax` | Local loop + `MPI_Allreduce(MPI_MAX)` |
| Histogram | `hist_kernel`: private shared-memory copy per block, then atomic merge | Private array per rank, then `MPI_Allreduce(MPI_SUM)` |
| Smoothing | `smooth_kernel`, one thread per bin | Every rank smooths its own copy |
| Search | One block per candidate, 1,921 blocks | Cyclic split of candidates across ranks |
| Winner | `argmin_kernel`, one block | Local best + `MPI_Allreduce(MPI_MINLOC)` |
| Percentile | `percentile_kernel`, one block scan | Every rank scans the 2,048 bins |
| Main overhead | Atomic contention, barriers, kernel boundaries | Message latency, waiting at collectives |

Both programs must choose the same bin as the one-thread CPU version on every layer
before any timing is recorded.
