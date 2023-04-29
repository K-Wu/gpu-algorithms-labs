#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A_LEFT_REG 128
#define TILE_SZ_B_LEFT_REG 16
#define TILE_SZ_RATIO_LEFT_REG (TILE_SZ_A_LEFT_REG / TILE_SZ_B_LEFT_REG)

__global__ void mysgemm_left_reg_right_shmem(int m, int n, int k, const float *A, const float *B, float *C) {

/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is row major
 *
 ********************************************************************/

// Macros for accessing flattened matrices
#define A(row, col) A[(row) + (col) *m]
#define B(row, col) B[(row) *n + (col)]
#define C(row, col) C[(row) + (col) *m]
  __shared__ float shmem[TILE_SZ_RATIO_LEFT_REG][TILE_SZ_B_LEFT_REG];
  // INSERT KERNEL CODE HERE

  int ArowIdx = blockIdx.x * TILE_SZ_A_LEFT_REG + threadIdx.x;

  for (int i = 0; i < (k + TILE_SZ_RATIO_LEFT_REG - 1) / TILE_SZ_RATIO_LEFT_REG; i++) {
    // load A in registers
    float reg[TILE_SZ_RATIO_LEFT_REG];
    if (ArowIdx < m) {
      for (int j = 0; j < TILE_SZ_RATIO_LEFT_REG; j++) {
        reg[j] = (i * TILE_SZ_RATIO_LEFT_REG + j < k) ? A(ArowIdx, i * TILE_SZ_RATIO_LEFT_REG + j) : 0.0f;
      }
    }
    // load B in shared memory
    int shdmemLDBrowIdx = i * TILE_SZ_RATIO_LEFT_REG + threadIdx.x / TILE_SZ_B_LEFT_REG;
    int shdmemLDBcolIdx = blockIdx.y * TILE_SZ_B_LEFT_REG + threadIdx.x % TILE_SZ_B_LEFT_REG;
    shmem[threadIdx.x / TILE_SZ_B_LEFT_REG][threadIdx.x % TILE_SZ_B_LEFT_REG] =
        (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n) ? B(shdmemLDBrowIdx, shdmemLDBcolIdx) : 0.0f;

    __syncthreads();
    // compute C
    if (ArowIdx < m) {
      for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B_LEFT_REG; shdmemColIdx++) {
        int CcolIdx = shdmemColIdx + blockIdx.y * TILE_SZ_B_LEFT_REG;
        if (CcolIdx < n) {
          for (int j = 0; j < TILE_SZ_RATIO_LEFT_REG; j++) {
            C(ArowIdx, CcolIdx) += reg[j] * shmem[j][shdmemColIdx];
          }
        }
      }
    }
    __syncthreads();
  }

  // SSL Hint (9/6/21): try using just one register for the tile of A
  // rather than several--in other words, load one value (per thread)
  // from A and compute using that value rather than loading all values
  // before doing the computation.  This approach seems to be slightly
  // faster than the alternative.
}

#define TILE_SZ_B_RIGHT_REG 128
#define TILE_SZ_A_RIGHT_REG 16
#define TILE_SZ_RATIO_RIGHT_REG (TILE_SZ_B_RIGHT_REG / TILE_SZ_A_RIGHT_REG)
// left shmem right reg
__global__ void mysgemm_left_shmem_right_reg(int m, int n, int k, const float *A, const float *B, float *C) {

/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is row major
 *
 ********************************************************************/

// Macros for accessing flattened matrices
#define A(row, col) A[(row) + (col) *m]
#define B(row, col) B[(row) *n + (col)]
#define C(row, col) C[(row) + (col) *m]
  __shared__ float shmem[TILE_SZ_A_RIGHT_REG][TILE_SZ_RATIO_RIGHT_REG];
  // INSERT KERNEL CODE HERE

  int BcolIdx = blockIdx.x * TILE_SZ_B_RIGHT_REG + threadIdx.x;

  for (int i = 0; i < (k + TILE_SZ_RATIO_RIGHT_REG - 1) / TILE_SZ_RATIO_RIGHT_REG; i++) {
    // load B in registers
    float reg[TILE_SZ_RATIO_RIGHT_REG];
    if (BcolIdx < n) {
      for (int j = 0; j < TILE_SZ_RATIO_RIGHT_REG; j++) {
        reg[j] = (i * TILE_SZ_RATIO_RIGHT_REG + j < k) ? B(i * TILE_SZ_RATIO_RIGHT_REG + j, BcolIdx) : 0.0f;
      }
    }
    // load A in shared memory
    constexpr bool A_column_major = true;
    if (A_column_major) {
      int shdmemLDAcolIdx = i * TILE_SZ_RATIO_RIGHT_REG + threadIdx.x % TILE_SZ_RATIO_RIGHT_REG;
      int shdmemLDArowIdx = blockIdx.y * TILE_SZ_A_RIGHT_REG + threadIdx.x / TILE_SZ_RATIO_RIGHT_REG;
      shmem[threadIdx.x / TILE_SZ_RATIO_RIGHT_REG][threadIdx.x % TILE_SZ_RATIO_RIGHT_REG] =
          (shdmemLDAcolIdx < k && shdmemLDArowIdx < m) ? A(shdmemLDArowIdx, shdmemLDAcolIdx) : 0.0f;
    } else {
      int shdmemLDAcolIdx = i * TILE_SZ_RATIO_RIGHT_REG + threadIdx.x / TILE_SZ_A_RIGHT_REG;
      int shdmemLDArowIdx = blockIdx.y * TILE_SZ_A_RIGHT_REG + threadIdx.x % TILE_SZ_A_RIGHT_REG;
      shmem[threadIdx.x % TILE_SZ_A_RIGHT_REG][threadIdx.x / TILE_SZ_A_RIGHT_REG] =
          (shdmemLDAcolIdx < k && shdmemLDArowIdx < m) ? A(shdmemLDArowIdx, shdmemLDAcolIdx) : 0.0f;
    }

    __syncthreads();
    // compute C
    if (BcolIdx < n) {
      for (int shdmemRowIdx = 0; shdmemRowIdx < TILE_SZ_A_RIGHT_REG; shdmemRowIdx++) {
        int CrowIdx = shdmemRowIdx + blockIdx.y * TILE_SZ_A_RIGHT_REG;
        if (CrowIdx < m) {
          for (int j = 0; j < TILE_SZ_RATIO_RIGHT_REG; j++) {
            // TODO: use register array to store the accumulation results. THe size should be TILE_SZ_A_RIGHT_REG
            C(CrowIdx, BcolIdx) += shmem[shdmemRowIdx][j] * reg[j];
          }
        }
      }
    }

    __syncthreads();
  }

  // SSL Hint (9/6/21): try using just one register for the tile of A
  // rather than several--in other words, load one value (per thread)
  // from A and compute using that value rather than loading all values
  // before doing the computation.  This approach seems to be slightly
  // faster than the alternative.
}

#define TILE_SZ_B_SHMEM_TILED 16
#define TILE_SZ_A_SHMEM_TILED 32 //16 // 32
#define TILE_SZ_K_SHMEM_TILED 8 //16 // 8
#define THREAD_BLOCK_DIM_A_Y_SHMEM_TILED 16
#define THREAD_BLOCK_DIM_B_X_SHMEM_TILED 16
// left shmem right reg
__global__ void mysgemm_shmem_tiled(int m, int n, int k, const float *A, const float *B, float *C) {

/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is row major
 *
 ********************************************************************/

// Macros for accessing flattened matrices
#define A(row, col) A[(row) + (col) *m]
#define B(row, col) B[(row) *n + (col)]
#define C(row, col) C[(row) + (col) *m]
  __shared__ float shmem_A[TILE_SZ_A_SHMEM_TILED][TILE_SZ_K_SHMEM_TILED];
  __shared__ float shmem_B[TILE_SZ_K_SHMEM_TILED][TILE_SZ_B_SHMEM_TILED];
  __shared__ float shmem_C[TILE_SZ_A_SHMEM_TILED][TILE_SZ_B_SHMEM_TILED];
  int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  // INSERT KERNEL CODE HERE

  for (int shdmemRowIdx = threadIdx.y; shdmemRowIdx < TILE_SZ_A_SHMEM_TILED; shdmemRowIdx += THREAD_BLOCK_DIM_A_Y_SHMEM_TILED) {
    for (int shdmemColIdx = threadIdx.x; shdmemColIdx < TILE_SZ_B_SHMEM_TILED; shdmemColIdx += THREAD_BLOCK_DIM_B_X_SHMEM_TILED) {
      shmem_C[shdmemRowIdx][shdmemColIdx] = 0.0f;
    }
  }

  //   int ArowIdx_offset = blockIdx.y * TILE_SZ_A_SHMEM_TILED;
  //   int BcolIdx_offset = blockIdx.x * TILE_SZ_B_SHMEM_TILED;

  for (int i = 0; i < (k + TILE_SZ_K_SHMEM_TILED - 1) / TILE_SZ_K_SHMEM_TILED; i++) {

    // load A in shared memory
    constexpr bool A_column_major = true;
    for (int load_thread_idx = thread_idx; load_thread_idx < TILE_SZ_A_SHMEM_TILED * TILE_SZ_K_SHMEM_TILED;
         load_thread_idx += THREAD_BLOCK_DIM_A_Y_SHMEM_TILED * THREAD_BLOCK_DIM_B_X_SHMEM_TILED) {
      if (A_column_major) {
        int shdmemLDAcolIdx = i * TILE_SZ_K_SHMEM_TILED + load_thread_idx % TILE_SZ_K_SHMEM_TILED;
        int shdmemLDArowIdx = blockIdx.y * TILE_SZ_A_SHMEM_TILED + load_thread_idx / TILE_SZ_K_SHMEM_TILED;
        shmem_A[load_thread_idx / TILE_SZ_K_SHMEM_TILED][load_thread_idx % TILE_SZ_K_SHMEM_TILED] =
            (shdmemLDAcolIdx < k && shdmemLDArowIdx < m) ? A(shdmemLDArowIdx, shdmemLDAcolIdx) : 0.0f;
      } else {
        int shdmemLDAcolIdx = i * TILE_SZ_K_SHMEM_TILED + load_thread_idx / TILE_SZ_A_SHMEM_TILED;
        int shdmemLDArowIdx = blockIdx.y * TILE_SZ_A_SHMEM_TILED + load_thread_idx % TILE_SZ_A_SHMEM_TILED;
        shmem_A[load_thread_idx % TILE_SZ_A_SHMEM_TILED][load_thread_idx / TILE_SZ_A_SHMEM_TILED] =
            (shdmemLDAcolIdx < k && shdmemLDArowIdx < m) ? A(shdmemLDArowIdx, shdmemLDAcolIdx) : 0.0f;
      }
    }

    // load B in shared memory
    for (int load_thread_idx = thread_idx; load_thread_idx < TILE_SZ_B_SHMEM_TILED * TILE_SZ_K_SHMEM_TILED;
         load_thread_idx += THREAD_BLOCK_DIM_A_Y_SHMEM_TILED * THREAD_BLOCK_DIM_B_X_SHMEM_TILED) {
      int shdmemLDBrowIdx = i * TILE_SZ_K_SHMEM_TILED + load_thread_idx / TILE_SZ_B_SHMEM_TILED;
      int shdmemLDBcolIdx = blockIdx.x * TILE_SZ_B_SHMEM_TILED + load_thread_idx % TILE_SZ_B_SHMEM_TILED;
      shmem_B[load_thread_idx / TILE_SZ_B_SHMEM_TILED][load_thread_idx % TILE_SZ_B_SHMEM_TILED] =
          (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n) ? B(shdmemLDBrowIdx, shdmemLDBcolIdx) : 0.0f;
    }
    __syncthreads();
    // compute C
    for (int shdmemRowIdx = threadIdx.y; shdmemRowIdx < TILE_SZ_A_SHMEM_TILED; shdmemRowIdx += THREAD_BLOCK_DIM_A_Y_SHMEM_TILED) {
      for (int shdmemColIdx = threadIdx.x; shdmemColIdx < TILE_SZ_B_SHMEM_TILED; shdmemColIdx += THREAD_BLOCK_DIM_B_X_SHMEM_TILED) {
        for (int j = 0; j < TILE_SZ_K_SHMEM_TILED; j++) {
          shmem_C[shdmemRowIdx][shdmemColIdx] += shmem_A[shdmemRowIdx][j] * shmem_B[j][shdmemColIdx];
        }
      }
    }

    __syncthreads();
  }

  for (int shdmemRowIdx = threadIdx.y; shdmemRowIdx < TILE_SZ_A_SHMEM_TILED; shdmemRowIdx += THREAD_BLOCK_DIM_A_Y_SHMEM_TILED) {
    for (int shdmemColIdx = threadIdx.x; shdmemColIdx < TILE_SZ_B_SHMEM_TILED; shdmemColIdx += THREAD_BLOCK_DIM_B_X_SHMEM_TILED) {
      int CrowIdx = shdmemRowIdx + blockIdx.y * TILE_SZ_A_SHMEM_TILED;
      int CcolIdx = shdmemColIdx + blockIdx.x * TILE_SZ_B_SHMEM_TILED;
      if (CrowIdx < m && CcolIdx < n) {
        C(CrowIdx, CcolIdx) = shmem_C[shdmemRowIdx][shdmemColIdx];
      }
    }
  }

  // SSL Hint (9/6/21): try using just one register for the tile of A
  // rather than several--in other words, load one value (per thread)
  // from A and compute using that value rather than loading all values
  // before doing the computation.  This approach seems to be slightly
  // faster than the alternative.
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                float *C, int ldc) {
  if ((transa != 'N') && (transa != 'n')) {
    printf("unsupported value of 'transa'\n");
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    printf("unsupported value of 'transb'\n");
    return;
  }

  if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
    printf("unsupported value of alpha\n");
    return;
  }

  if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
    printf("unsupported value of beta\n");
    return;
  }

  // Your code need only consider the m, n, k, A, B, and C parameters of
  // the function, which provide the matrix sizes (m, n, k) and data
  // (A, B, C).

  // INSERT CODE HERE
  cudaMemset(C, 0, m * n * sizeof(float));
  // Invoke CUDA kernel -----------------------------------------------------
  // Initialize thread block and kernel grid dimensions ---------------------
  constexpr bool left_reg_flag     = true;
  constexpr bool shmem_tiling_flag = true;
  if (shmem_tiling_flag) {
    dim3 blocks((n + TILE_SZ_B_SHMEM_TILED - 1) / TILE_SZ_B_SHMEM_TILED, (m + TILE_SZ_A_SHMEM_TILED - 1) / TILE_SZ_A_SHMEM_TILED);
    dim3 threadsPerBlock(THREAD_BLOCK_DIM_B_X_SHMEM_TILED, THREAD_BLOCK_DIM_A_Y_SHMEM_TILED);
    mysgemm_shmem_tiled<<<blocks, threadsPerBlock>>>(m, n, k, A, B, C);
  } else {
    if (left_reg_flag) {
      dim3 blocks((m + TILE_SZ_A_LEFT_REG - 1) / TILE_SZ_A_LEFT_REG, (n + TILE_SZ_B_LEFT_REG - 1) / TILE_SZ_B_LEFT_REG);
      dim3 threadsPerBlock(TILE_SZ_A_LEFT_REG);
      mysgemm_left_reg_right_shmem<<<blocks, threadsPerBlock>>>(m, n, k, A, B, C);
    } else {
      // blockDim y = 1, blockDim x = TILE_SZ_B
      // gridDim y = (m+TILE_SZ_A-1)/TILE_SZ_A, gridDim x = (n+TILE_SZ_B-1)/TILE_SZ_B
      dim3 blocks((n + TILE_SZ_B_RIGHT_REG - 1) / TILE_SZ_B_RIGHT_REG, (m + TILE_SZ_A_RIGHT_REG - 1) / TILE_SZ_A_RIGHT_REG);
      dim3 threadsPerBlock(TILE_SZ_B_RIGHT_REG);
      mysgemm_left_shmem_right_reg<<<blocks, threadsPerBlock>>>(m, n, k, A, B, C);
    }
  }
  // INSERT CODE HERE
}
