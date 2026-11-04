# Algorithm 2: GPTQ, CUDA and MPI breakdown

GPTQ compresses a layer's **weights** to INT4 (whole numbers from -7 to 7, one scale per
row). Plain rounding (round-to-nearest, RTN) rounds every weight to its closest level.
GPTQ rounds one column at a time and nudges the not-yet-rounded weights to cancel the
error it just made, so the layer's *outputs* change as little as possible.

| | P3 (CUDA) | P4 (MPI) |
|---|---|---|
| Workers | GPU threads; one warp per weight row in the sweep | p processes |
| Hessian | Tiled shared-memory kernel (or cuBLAS SYRK) | Each rank builds a partial H, then one allreduce |
| Factorization | cuSOLVER | LAPACK, on every rank (design A) or rank 0 (design B) |
| Sweep | One warp per row, shuffles, no block barriers | Each rank sweeps its own rows, no messages |
| Report section it feeds | 5, contention and synchronization | 4, communication pattern |

The pipeline for one layer is the same in both:

```
activations -> im2col -> H = 2 X^T X -> dead inputs, damping -> (act-order)
            -> Cholesky, inverse, Cholesky -> U -> row scales
            -> for each block of 128 columns: sweep, then trailing GEMM
            -> (undo act-order) -> INT4 values + scales
```

---

## Part 0: the idea in plain words

### What is being minimized

A layer computes `Y = X W^T`: each row of `X` is one input patch, and each row `w` of `W`
is one output channel's weights (length `d`). Quantizing `w` to `w_hat` changes that
channel's output by `X (w - w_hat)`. GPTQ picks `w_hat` to make that change small:

```
minimize  || X w - X w_hat ||^2
```

- **Rows are independent.** Each output channel has its own row, and the error for one
  row never mentions another. That's where the parallelism comes from.
- **It's not "round each weight."** RTN makes `w - w_hat` small. GPTQ makes the *output*
  error small, which sometimes means rounding a weight the "wrong" way on purpose.

### From X to H

The output error only depends on `X` through one `d x d` matrix:

```
|| X (w - w_hat) ||^2 = 1/2 (w - w_hat)^T H (w - w_hat),   H = 2 X^T X
```

`X` is huge (32,768 rows x 576 for a late ResNet-20 layer at 512 images), but `H` is
only 576 x 576. Once `H` is built, `X` can be thrown away.

- **Diagonal `H[j][j]`:** how much input `j` is used. An error on weight `j` matters in
  proportion to it.
- **Off-diagonal `H[j][k]`:** whether inputs `j` and `k` move together. If they do, an
  error on weight `j` can be cancelled by nudging weight `k`. This is the whole trick.

### Tiny worked example

Two weights, a grid of whole numbers, and `H = [[2, 1], [1, 2]]` (inputs that move
together). Start from `w = [0.6, 0.6]`.

1. Round `w0`: 0.6 becomes 1, an error of -0.4.
2. Push the error onto `w1`: GPTQ's formula moves `w1` down by 0.2, to 0.4.
3. Round `w1`: 0.4 becomes 0.

GPTQ gives `[1, 0]`; RTN gives `[1, 1]`. Output error: RTN 0.96, GPTQ 0.56, 42% lower,
even though `w1` was rounded "the wrong way." Because the inputs move together, rounding
both up made the errors add; GPTQ made them cancel.

### Why the columns are sequential but the rows aren't

GPTQ rounds columns left to right, and every row uses the same order. So:

- **Across rows: fully parallel.** Each row's sweep never reads another row.
- **Across columns: strictly sequential.** Column `j + 1` can't be rounded until column
  `j`'s error has been pushed onto it.

This is the main difference from Algorithm 1, where the 1,921 candidates were all
independent. GPTQ has a critical path of `d` sequential steps, which caps its speedup
(Amdahl's law).

### The three tricks that make it fast

1. **One fixed column order for every row,** so the bookkeeping is computed once.
2. **Cholesky instead of updating `H^-1` after every column.** The numbers the sweep needs
   at column `j` are exactly row `j` of `U`, the upper Cholesky factor of `H^-1`
   (`H^-1 = U^T U`). So one factorization up front replaces `d` update steps:
   ```
   err_j = (w_j - quant(w_j)) / U[j][j]
   w_k  -= err_j * U[j][k]     for every later column k
   ```
3. **Lazy batches of B = 128 columns.** Inside a block, only the block's own columns get
   updated, and each error is saved in `E`. When the block finishes, one matrix multiply
   (`W[:, after] -= E * U[block, after]`) applies all of them to the columns to the right.
   Same math, far fewer passes over memory.

### Scales and act-order

- **Scale per row, fixed before the sweep:** `s = max|w| / 7` from the original row (1 for
  an all-zero row). Levels are `clamp(round(w / s), -7, 7)`.
- **Act-order (optional):** sweep the columns in order of decreasing `H[j][j]`, so the
  most-used inputs are rounded first, while the most weights are left to compensate.
  It needs a sort and two gathers, and the permutation is undone at the end.

### ResNet-20 layer shapes

`d = c_in x 3 x 3` columns, `d_row = c_out` rows. The weights are in `data/layerNN_w.bin`
(`c_out x c_in x 3 x 3`, already folded with batch norm by `prepare.py`), and the inputs
are the same `layerNN_act.bin` files Algorithm 1 uses.

| Layers | Input | Weights (rows x cols) | Stride | X rows at 512 images |
|---|---|---|---|---|
| 01 - 06 | 16 x 32 x 32 | 16 x 144 | 1 | 524,288 |
| 07 | 16 x 32 x 32 | 32 x 144 | 2 | 131,072 |
| 08 - 12 | 32 x 16 x 16 | 32 x 288 | 1 | 131,072 |
| 13 | 32 x 16 x 16 | 64 x 288 | 2 | 32,768 |
| 14 - 18 | 64 x 8 x 8 | 64 x 576 | 1 | 32,768 |
| 19 (FC) | 64 | 10 x 64 | - | 512 |

Padding is 1 for every conv. Layer 00 (the first conv, on the image) can be quantized
too; its input file is `layer00_act.bin`.

**ResNet-20 is small for this algorithm.** At most 64 rows means at most 64 warps in the
sweep or 64 MPI ranks with work. The plan runs the scaling experiments on larger
synthetic shapes too (for example ResNet-18 sizes: up to 512 rows and d = 4,608) so the
plots show real scaling, and uses ResNet-20 for accuracy.

---

## Part 1: P3 (CUDA)

### Data structures

| Name | Type and shape | Where | Purpose |
|---|---|---|---|
| `act` | float, N x C x H x W | Global | Layer input, from the export |
| `X` | float, n_chunk x d | Global, reused | im2col of one chunk of images (for example 64 at a time) |
| `H` | float, d x d | Global | Hessian, summed over chunks; later overwritten by U |
| `perm` | int, d | Global | Act-order permutation |
| `W` | float, d_row x d, row-major | Global | Weights, updated in place by the sweep |
| `scale` | float, d_row | Global | One INT4 scale per row |
| `E` | float, d_row x B | Global | Scaled errors of the current block |
| `Q` | int8, d_row x d | Global | INT4 values, one per byte |
| `s_U` | float, B x B | Shared | The corner of U the sweep reads every column |

### Order for one layer

```
for each image chunk:  im2col_kernel -> hessian_kernel (or cublasSsyrk)
symmetrize_kernel -> dead_inputs_kernel -> diag_mean_kernel -> damp_kernel
(act-order: CUB sort of the diagonal -> gather_columns_kernel on W -> gather_hessian_kernel)
cusolverDnSpotrf -> cusolverDnSpotri -> symmetrize_kernel -> cusolverDnSpotrf   (U in H)
row_scales_kernel
for col0 = 0, 128, 256, ...:  gptq_sweep<128> -> cublasSgemm on the later columns
(act-order: gather_columns_kernel on Q with the inverse permutation)
```

### `im2col_kernel`: turn image patches into rows of X

```cpp
__global__ void im2col_kernel(const float* act, int first_image, int n_images, int c_in,
                              int height, int width, int k, int stride, int pad, float* X);
```

- **Why:** a conv weight row multiplies one 3 x 3 patch across all input channels at a
  time. Laying each patch out as a row of `X` turns the conv into a plain matrix
  multiply, which is what `H = 2 X^T X` needs. The FC layer skips this; its input already
  is `X`.
- **Launch:** one thread per element of `X`, grid-stride over `n_rows x d`.
- **Steps:**
  1. Split the row index into (image, y, x), the output position.
  2. Split the column index into (channel, ky, kx). This order matches how a weight row
     is laid out, `W[out][channel][ky][kx]`.
  3. Read `act[image][channel][y * stride + ky - pad][x * stride + kx - pad]`, or 0 if
     that falls in the padding.
  4. Write `X[row][col]`.
- **Memory:** reads each input value up to 9 times (patches overlap), writes `X` once.
  Chunks keep `X` small: layer 01 for all 512 images would be 300 MB.
- **Combining:** none; every thread writes its own element.

### `hessian_kernel`: H += 2 X^T X, one 32 x 32 tile per block

```cpp
__global__ void hessian_kernel(const float* X, int n_rows, int d, float* H);
```

- **Why:** this is the heavy compute step, about `2 n d^2` operations (22 GFLOP for
  layer 14 at 512 images). It's the classic shared-memory matrix multiply.
- **Launch:** a 2D grid of tiles, 32 x 32 threads each. Each block owns one 32 x 32 tile
  of `H`.
- **Steps:**
  1. Blocks below the diagonal exit, since `H` is symmetric. That halves the work.
  2. Load 32 rows of `X` for the tile's two column ranges into shared memory.
  3. `__syncthreads`.
  4. Each thread multiplies and adds for its one element of the tile.
  5. `__syncthreads`, then repeat from step 2 until the rows run out.
  6. Each thread adds `2 * sum` to its element of `H`.
- **Memory:** each value of `X` loaded from global memory is reused 32 times from shared
  memory. That reuse is why tiling is fast.
- **Combining:** none across blocks. Each block owns its tile, so no atomics.
- **Waiting:** 2 barriers per 32-row step.
- **Comparison for the report:** `cublasSsyrk` (alpha = 2, beta = 1) does the same thing.
  cuBLAS is column-major, so a row-major `X` looks to it like `X^T`; call SYRK with
  `CUBLAS_OP_N` on that to get `X^T X`.
- **Edge tiles:** d = 144 (layers 01 - 07) isn't a multiple of 32, and the FC layer's
  X rows (512) may not fill a step either, so guard every load and store with `< d` and
  `< n_rows`.

### `symmetrize_kernel`: copy the computed triangle to the other one

```cpp
__global__ void symmetrize_kernel(float* H, int d);
```

- **Why:** the tiles only computed the upper triangle, and `potri` also writes only one.
  Run it after the Hessian and again after `potri`.
- **Launch:** one thread per pair `i < j`. **Steps:** `H[j][i] = H[i][j]`.

### `dead_inputs_kernel`: inputs that never fire

```cpp
__global__ void dead_inputs_kernel(float* H, float* W, int d, int d_row);
```

- **Why:** if ReLU makes an input channel always zero, its row and column of `H` are zero
  and `H` can't be inverted. The reference GPTQ sets `H[j][j] = 1` and zeros that weight
  column; its value doesn't matter, since its input is always 0.
- **Launch:** one thread per column `j`. **Steps:** if `H[j][j] == 0`, set it to 1 and zero
  `W[r][j]` for every row.

### `diag_mean_kernel` and `damp_kernel`: damping

```cpp
__global__ void diag_mean_kernel(const float* H, int d, float* d_mean);
__global__ void damp_kernel(float* H, int d, float lambda);
```

- **Why:** a small boost on the diagonal keeps the factorization stable. GPTQ always
  uses `lambda = 0.01 * mean(diag H)`, so it scales with the matrix.
- **`diag_mean`:** one block, grid-stride over the diagonal, block reduction (your
  `block_reduce_sum` from Algorithm 1), divide by `d`.
- **Host:** copy the mean back (4 bytes) and compute `lambda`. **`damp`:** one thread per
  `j`, `H[j][j] += lambda`.

### Act-order (optional): `gather_columns_kernel`, `gather_hessian_kernel`

```cpp
template <typename T>
__global__ void gather_columns_kernel(const T* in, T* out, const int* perm, int rows, int d);
__global__ void gather_hessian_kernel(const float* in, float* out, const int* perm, int d);
```

- **Steps:** sort the diagonal largest-first with
  `cub::DeviceRadixSort::SortPairsDescending` (keys = diagonal, values = 0 .. d - 1),
  which gives `perm`. Then `out[r][j] = in[r][perm[j]]` for `W`, and
  `out[i][j] = in[perm[i]][perm[j]]` for `H`. At the end, gather `Q` with the inverse
  permutation.
- **Launch:** one thread per output element. No combining.

### cuSOLVER: from H to U

```cpp
// cuSOLVER is column-major. A symmetric row-major H looks the same in column-major,
// and LOWER in the column-major view is UPPER in the row-major view.
cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, d, H, d, work, lwork, info); // H = L L^T
cusolverDnSpotri(h, CUBLAS_FILL_MODE_LOWER, d, H, d, work, lwork, info); // H = H^-1, one triangle
symmetrize_kernel<<<...>>>(H, d);                                        // fill the other triangle
cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, d, H, d, work, lwork, info); // H^-1 = L L^T
// The row-major upper triangle of H now holds U, with H^-1 = U^T U.
```

- **Why a library:** the factorization is `O(d^3)` but small here (about 64 million
  operations at d = 576), and getting it right by hand is hard.
- **Checks:** query the workspace with `cusolverDnSpotrf_bufferSize` first, and check
  `info` after every call. Non-zero means `H` wasn't positive definite: raise the damping.
- **Verify the fill mode first.** The row-major/column-major swap is easy to get
  backwards and gives wrong results silently. Check `U` against the CPU version on a
  small matrix before going further.

### `row_scales_kernel`: one INT4 scale per row

```cpp
__global__ void row_scales_kernel(const float* W, int d_row, int d, float* scale);
```

- **Launch:** one warp per row.
- **Steps:** each lane scans its share of the row for the largest `|w|`, the warp reduces
  with shuffles (exactly your `absmax_kernel` pattern), and lane 0 writes
  `max / 7`, or 1 for an all-zero row.
- **Run it on the original weights,** before the sweep changes them.

### `gptq_sweep<B>`: the column sweep, one warp per row

```cpp
template <int B>
__global__ void gptq_sweep(const float* W, int d, int d_row, int col0, int ncols,
                           const float* U, const float* scale, int8_t* Q, float* E);
```

- **Why:** this is the sequential heart of GPTQ. Putting one row on one warp means the
  column-by-column synchronization happens with shuffles inside the warp, with no block
  barriers.
- **Launch:** a few warps per block (for example 4), `ceil(d_row / 4)` blocks, and
  `B * B * 4` bytes of shared memory for `s_U`. Once per block of columns. `ncols` is
  usually `B`, but the last block can be shorter: d = 144 is one block of 128 plus 16.
- **Steps:**
  1. All threads load the `ncols x ncols` corner of `U` for this block into `s_U`.
  2. `__syncthreads`, the only barrier in the kernel.
  3. `row = blockIdx.x * warps_per_block + warp`; exit if `row >= d_row`.
  4. Each lane loads its `B / 32` columns of the row into registers.
  5. For each column `j` in the block:
     - The lane holding column `j` broadcasts it with `__shfl_sync`.
     - Every lane computes the same `q = clamp(round(w_j / s), -7, 7)` and
       `err = (w_j - q * s) / U[j][j]`.
     - The owner writes `Q[row][col0 + j]` and `E[row][j]`.
     - Each lane updates its later columns: `w[k] -= err * U[j][k]` for `k > j`.
- **Memory:** each weight is read and written once per block; `U` comes from shared
  memory every column.
- **Shared memory:** `s_U` is 64 KB at B = 128, above the default 48 KB. Opt in with
  `cudaFuncSetAttribute(gptq_sweep<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)`
  (your GPU allows 99 KB), or use B = 64 (16 KB) and compare.
- **Parallelism:** only `d_row` warps, at most 64 here. That's why the scaling runs need
  bigger shapes.

### `gptq_sweep_barrier<B>`: the same sweep with block barriers

```cpp
template <int B>
__global__ void gptq_sweep_barrier(const float* W, int d, int d_row, int col0, int ncols,
                                   const float* U, const float* scale, int8_t* Q, float* E);
```

- **Why:** the comparison that measures synchronization cost for report section 5, the
  way `hist_global_kernel` measured contention in Algorithm 1.
- **Steps:** threads map to (row, column) pairs with the block's weights in shared
  memory. Each column: compute `q` and `err` per row, `__syncthreads`, update the later
  columns, `__syncthreads`. That's `2 x B` barriers per block instead of 1.

### cuBLAS: the trailing update after each block

```cpp
// W[:, after] -= E * U[block, after]. In column-major terms (W looks like W^T):
//   W^T[after, :] -= U[block, after]^T * E^T
const float minus1 = -1.f, one = 1.f;
const int after = col0 + ncols, m = d - after;   // columns still to quantize
if (m > 0)
    cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, m, d_row, ncols, &minus1,
                U + (size_t)col0 * d + after, d,  // U[block, after], m x ncols in column-major
                E, B,                              // E^T, ncols x d_row
                &one, W + after, d);               // W^T[after.., :], m x d_row
```

- **Why:** applies all of a block's errors to every later column in one big, fast
  multiply instead of `B` small passes.
- **Verify one block against the CPU version first.** Transposes in GEMM calls are the
  classic silent bug.

### Timing for the report

CUDA events around each phase, the same way as P1:

| Phase | Kind |
|---|---|
| im2col + Hessian | Compute-bound (tiles vs cuBLAS) |
| Dead inputs, damping, act-order | Small kernels |
| cuSOLVER (2 potrf + potri) | Library |
| Sweep (all blocks) | Synchronization-bound (shuffle vs barrier version) |
| Trailing GEMMs | Compute-bound |

"Number of processes" for P3: thread blocks for the Hessian kernel, and warps for the
sweep.

---

## Part 2: P4 (MPI)

### The idea

Split the **images** to build `H` (partial Hessians just add, like histograms), then
split the **rows** for the sweep. Rows are independent, so the sequential column loop
needs no messages at all.

### Data distribution

| What | How it is split | Why |
|---|---|---|
| Calibration images | Rank r owns images `r * N / p` to `(r + 1) * N / p`, same as P2 | One contiguous read per rank |
| Hessian | Each rank builds `H_r` from its images, then they're summed | `H = sum of 2 X_r^T X_r` over any split |
| `U` | Every rank has the full matrix | Every row's sweep needs all of `U` |
| Weight rows | Rank r owns rows `r * d_row / p` to `(r + 1) * d_row / p` | Rows are independent |

### Steps per layer

#### 1. Read my images and the weights

`MPI_File_read_at` for my slice of `layerNN_act.bin` (the same `read_slice` as P2), and
every rank reads all of `layerNN_w.bin` (it's small: at most 64 x 576 floats).

#### 2. Local Hessian [`im2col_kernel`, `hessian_kernel`]

- Run im2col on my images in chunks, then `H_r += 2 X^T X`, with the same blocked loops
  as the CPU version, or OpenBLAS `cblas_ssyrk` for both so the comparison is fair.
- **Communication:** none.

#### 3. Combine the Hessians

Two designs to compare:

| Design | Communication | Computation |
|---|---|---|
| **A:** `MPI_Allreduce` of H, every rank factorizes | One allreduce of d^2 floats | The O(d^3) factorization repeated on every rank, in parallel |
| **B:** `MPI_Reduce` to rank 0, rank 0 factorizes, `MPI_Bcast` of U | One reduce + one broadcast of d^2 floats | Factorization once, but the other ranks wait |

Message size: d = 576 is 331,776 floats (1.3 MB), much larger than Algorithm 1's 8 KB
histogram. At ResNet-18 sizes (d = 4,608) it's 85 MB, and the allreduce becomes a real
cost, which makes P4's scaling plots interesting.

#### 4. Prepare and factorize [dead inputs, damping, cuSOLVER]

- Dead inputs and damping on every rank's copy (identical, so no messages).
- LAPACK through LAPACKE: `LAPACKE_spotrf`, `LAPACKE_spotri`, symmetrize,
  `LAPACKE_spotrf` again. These are the CPU counterparts of the cuSOLVER calls.
- Row scales for my rows only.

#### 5. Sweep my rows [`gptq_sweep`, cuBLAS GEMM]

```
for col0 = 0, 128, ...:
    sweep my rows through the block's columns (same math as the kernel), saving E
    cblas_sgemm: my rows' later columns -= E * U[block, after]
```

- **Communication:** none. This is the main design argument for P4.
- **The rejected alternative:** splitting *columns* across ranks would force the owner of
  column `j` to send its errors to everyone after every column: `d` sequential
  broadcasts, each costing about `log p` message latencies. Splitting rows removes that
  dependency completely.

#### 6. Gather

`MPI_Gatherv` collects the INT4 values (int8) and scales on rank 0. Counts differ by
rank when `p` doesn't divide `d_row`, hence the `v`.

### Communication per layer

| Step | Call | Message | Rounds |
|---|---|---|---|
| Hessian (A) | `MPI_Allreduce`, `MPI_SUM` | d^2 floats | log p, or a ring for long messages |
| Hessian (B) | `MPI_Reduce` + `MPI_Bcast` | d^2 floats each | 2 log p |
| Result | `MPI_Gatherv` | d_row x d bytes + d_row floats | p - 1 |

### Load balance

`d_row / p` rows per rank, so with ResNet-20's at most 64 rows, ranks beyond 64 have
nothing to do, and even at p = 22 the split is uneven for 16- or 32-row layers. That's
another reason to also run larger shapes.

### Timing per phase

| Phase | Kind |
|---|---|
| Read | I/O, reported separately |
| im2col + local Hessian | Computation |
| Hessian allreduce (or reduce + bcast) | Communication |
| Factorization | Computation, replicated in design A |
| Sweep + GEMMs | Computation |
| Gather | Communication |

Same method as P2: barrier, `MPI_Wtime`, slowest rank per phase.

### Time model

```
T_P4(p) ~ 2 n d^2 t_c / p                      local Hessian
        + (4/3) d^3 t_c                        factorization, not divided by p (design A)
        + d_row d^2 t_c / p                    sweep
        + log2(p) t_s + 2 d^2 t_w              Hessian allreduce (large-message algorithm)
        + (p - 1) t_s + (d_row d / 4) t_w      gather
```

The factorization doesn't shrink with p, so it's a serial fraction: Amdahl's law caps
the speedup at `1 / (serial share)`. Expect P4 to flatten earlier than P2.

---

## CUDA and MPI side by side

| Step | P3 (CUDA) | P4 (MPI) |
|---|---|---|
| Get the data | Copy the layer to the GPU | Each rank reads its own slice |
| Hessian | Tiled shared-memory kernel, one tile per block, no atomics | Local SYRK per rank, then allreduce (or reduce) |
| Factorization | cuSOLVER, once | LAPACK, on every rank (A) or rank 0 (B) |
| Sweep | One warp per row, shuffles, 1 barrier per block of columns | Each rank's rows, no messages |
| Trailing update | cuBLAS GEMM per block | cblas_sgemm per block, on my rows |
| Main overhead | Sequential column steps, kernel launches per block | Hessian allreduce, replicated factorization |

## Algorithm 1 vs Algorithm 2 (for the comparison section)

| | Algorithm 1 (calibration) | Algorithm 2 (GPTQ) |
|---|---|---|
| Quantizes | Activations, INT8 | Weights, INT4 |
| Parallel structure | Everything independent: values, then 1,921 candidates | Hessian independent; the sweep has `d` sequential steps |
| Heaviest step | Search over candidates | Building H (`2 n d^2`) |
| Largest MPI message | 8 KB histogram | d^2 floats (1.3 MB at d = 576) |
| GPU synchronization | Atomics, a few barriers | Shuffles every column, barriers in the comparison version |
| Expected scaling | Near-linear until the data per rank gets small | Flattens early: serial factorization and sequential columns |

---

## Building it: suggested order

Same approach as Algorithm 1: build the simple version first and check every faster
version against it.

1. **CPU version first,** in `common/gptq_common.h` like `calib_common.h`: im2col,
   Hessian, dead inputs and damping, factorization (a plain Cholesky is fine at these
   sizes), the sweep, and RTN for comparison. P3 checks against it and P4 runs it on its
   share. Check the tiny example above (`[1, 0]`, output error 0.56 vs 0.96) first.
2. **P3 kernels in pipeline order,** checking each against the CPU version:
   im2col -> Hessian (tiles, then cuBLAS) -> prep -> cuSOLVER -> scales -> sweep -> GEMM.
   The sweep and GEMM can match the CPU version exactly in `Q` for most weights, but
   float rounding can flip a few values that sit right between two levels; compare the
   output error rather than requiring identical `Q`.
3. **P4,** reusing P2's `read_slice` and phase timing.
4. **Accuracy:** `prepare.py --eval` only quantizes activations today. It needs a way to
   load quantized weights (`Q * scale` per row) to measure INT4 accuracy for RTN vs GPTQ,
   with and without act-order, and the final W4A8 model with Algorithm 1's best
   thresholds.
5. **Per-layer reconstruction error** for the report: `||X W^T - X W_hat^T||^2` for GPTQ
   vs RTN, which shows what the error compensation buys even before accuracy.

## Experiments for report 2

| Program | Parameter | Range |
|---|---|---|
| P3 | Hessian thread blocks | 1 up to full occupancy |
| P3 | Sweep: shuffle vs barrier version, B = 64 vs 128 | ResNet-20 and larger shapes |
| P3 | Hessian: own kernel vs cuBLAS SYRK | Same shapes |
| P4 | Ranks p | 1 to 22 cores (44 threads) |
| P4 | Design A vs B | Same p sweep |
| Both | Accuracy: FP32, RTN, GPTQ, GPTQ + act-order, W4A8 | ResNet-20 |
