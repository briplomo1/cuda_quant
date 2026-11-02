// Algorithm 2 kernels. Add a main and a CMake target when you start P3.
// Matrices are row-major. cuBLAS and cuSOLVER calls run between these kernels.
//
// Order for one layer:
//   For each image chunk: im2col -> hessian
//   symmetrize -> dead_inputs -> diag_mean -> damp
//   Optional act-order: gather_columns (W) and gather_hessian
//   cuSOLVER potrf -> potri -> symmetrize -> potrf, which leaves U in H
//   row_scales
//   For each block of columns: gptq_sweep -> cuBLAS GEMM on the later columns
//   Optional act-order: gather_columns (Q) with the inverse permutation
#include <cuda_runtime.h>

#include <cstdint>

// One chunk of images to X (n_rows x d), reading 0 in the padding.
// Skip for the FC layer, where X is the activations.
// A conv weight sees one small patch at a time, so each patch becomes a row of X.
//   1. One thread per (row, col) of X.
//   2. Turn row into (image, y, x) and col into (channel, ky, kx).
//   3. Read act[image][channel][y * stride + ky - pad][x * stride + kx - pad],
//      or 0 in the padding.
//   4. Write X[row][col].
__global__ void im2col_kernel(const float* act, int first_image, int n_images,
                              int c_in, int height, int width, int k, int stride,
                              int pad, float* X) {}

// H += 2 X^T X, one 32 x 32 tile per block, on or above the diagonal only.
// H records how the inputs overlap, which is all GPTQ needs from them.
//   1. Blocks below the diagonal exit.
//   2. Load 32 rows of the tile's two column ranges into shared memory.
//   3. __syncthreads.
//   4. Each thread owns one tile element and adds these rows' products to its sum.
//   5. __syncthreads, then repeat from step 2 until the rows run out.
//   6. Each thread adds 2 * sum to its element of H. Only this block writes the
//      tile, so no atomic is needed.
// Tiles at the edge are smaller when d is not a multiple of 32.
__global__ void hessian_kernel(const float* X, int n_rows, int d, float* H) {}

// Copies the computed triangle of H onto the other triangle.
// Run after the Hessian tiles and again after potri.
//   1. One thread per pair i < j.
//   2. H[j][i] = H[i][j].
__global__ void symmetrize_kernel(float* H, int d) {}

// Where H[j][j] == 0: set it to 1 and zero column j of W.
// A zero diagonal means the input never fires, which would break the inversion.
//   1. One thread per column j.
//   2. If H[j][j] == 0, set it to 1.
//   3. Zero column j of W in every row.
__global__ void dead_inputs_kernel(float* H, float* W, int d, int d_row) {}

// mean(diag H), for lambda = 0.01 * mean.
// The damping has to scale with the size of the matrix's values.
//   1. Threads sum the diagonal, sharing it out by grid-stride.
//   2. Reduce the sum across the block.
//   3. Divide by d and write d_mean.
__global__ void diag_mean_kernel(const float* H, int d, float* d_mean) {}

// H[j][j] += lambda.
// A small boost on the diagonal keeps the factorization from failing.
//   1. One thread per j.
//   2. H[j][j] += lambda.
__global__ void damp_kernel(float* H, int d, float lambda) {}

// Optional act-order: out[r][j] = in[r][perm[j]]. Use for W, and for Q with the
// inverse permutation.
// Quantizes the most important columns first, while later columns can still absorb
// their error. perm comes from sorting the diagonal, largest first, with CUB.
//   1. One thread per output position.
//   2. out[r][j] = in[r][perm[j]].
template <typename T>
__global__ void gather_columns_kernel(const T* in, T* out, const int* perm,
                                      int rows, int d) {}

// Optional act-order: out[i][j] = H[perm[i]][perm[j]].
//   1. One thread per output position.
//   2. out[i][j] = in[perm[i]][perm[j]].
__global__ void gather_hessian_kernel(const float* in, float* out, const int* perm,
                                      int d) {}

// scale[r] = max|W[r]| / 7, or 1 for an all-zero row. One warp per row.
// Rows differ in size, so each gets its own INT4 step. Run before the sweep, on the
// original weights.
//   1. One warp takes one row.
//   2. Each lane scans its share of the row's columns for the largest |w|.
//   3. Reduce the warp to one maximum with shuffles.
//   4. Lane 0 writes scale[row].
__global__ void row_scales_kernel(const float* W, int d_row, int d, float* scale) {}

// Quantizes columns col0 .. col0 + ncols - 1, where ncols <= B handles the last
// partial block. One warp per row; writes Q and E.
// Rounds each column in order and passes its error on to the columns after it.
//   1. Load the ncols x ncols corner of U for this block into s_U. Don't read past
//      ncols. Above 48 KB, opt in with cudaFuncSetAttribute.
//   2. __syncthreads, the only barrier in this kernel.
//   3. row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32. Exit if
//      row >= d_row.
//   4. Each lane loads its columns into registers: B / 32 of them.
//   5. For each column j:
//        The lane owning j broadcasts its value wj with __shfl_sync.
//        All lanes compute qi = clamp(round(wj / s), -7, 7).
//        All lanes compute err = (wj - qi * s) / U[j][j].
//        The owner writes Q[row][col0 + j] and E[row][j].
//        Each lane updates its later columns: w[k] -= err * U[j][k].
// Then on the host: cublasSgemm applies E to the columns after this block.
template <int B>
__global__ void gptq_sweep(const float* W, int d, int d_row, int col0, int ncols,
                           const float* U, const float* scale, int8_t* Q, float* E) {}

// Same result using two __syncthreads per column, to measure synchronization cost.
//   1. Load s_U and the block's weights into shared memory.
//   2. __syncthreads.
//   3. For each column j:
//        Compute qi and err for each row and store them in shared memory.
//        __syncthreads.
//        Update the later columns in shared memory.
//        __syncthreads.
//   4. Write Q and E.
template <int B>
__global__ void gptq_sweep_barrier(const float* W, int d, int d_row, int col0,
                                   int ncols, const float* U, const float* scale,
                                   int8_t* Q, float* E) {}
