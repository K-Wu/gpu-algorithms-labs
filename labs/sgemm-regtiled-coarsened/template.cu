#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A_LEFT_REG 128
#define TILE_SZ_B_LEFT_REG 16
#define TILE_SZ_RATIO_LEFT_REG (TILE_SZ_A_LEFT_REG/TILE_SZ_B_LEFT_REG)

__global__ void mysgemm_left_reg_right_shmem(int m, int n, int k, const float *A, const float *B, float* C) {

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
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]
  __shared__ float shmem[TILE_SZ_RATIO_LEFT_REG][TILE_SZ_B_LEFT_REG];
  // INSERT KERNEL CODE HERE

  int ArowIdx = blockIdx.x*TILE_SZ_A_LEFT_REG + threadIdx.x;
  
  for (int i = 0; i < (k+TILE_SZ_RATIO_LEFT_REG-1)/TILE_SZ_RATIO_LEFT_REG; i++) {
    // load A in registers
    float reg[TILE_SZ_RATIO_LEFT_REG];
    if (ArowIdx < m) {
        for (int j = 0; j < TILE_SZ_RATIO_LEFT_REG; j++) {
            reg[j] = (i*TILE_SZ_RATIO_LEFT_REG+j<k)?A(ArowIdx,i*TILE_SZ_RATIO_LEFT_REG+j):0.0f;
        }
    }
    // load B in shared memory
    int shdmemLDBrowIdx = i*TILE_SZ_RATIO_LEFT_REG+threadIdx.x/TILE_SZ_B_LEFT_REG;
    int shdmemLDBcolIdx =blockIdx.y*TILE_SZ_B_LEFT_REG + threadIdx.x%TILE_SZ_B_LEFT_REG;
    shmem[threadIdx.x/TILE_SZ_B_LEFT_REG][threadIdx.x%TILE_SZ_B_LEFT_REG] = (shdmemLDBrowIdx<k && shdmemLDBcolIdx<n)?B(shdmemLDBrowIdx,shdmemLDBcolIdx):0.0f;
    
    __syncthreads();
    // compute C
    if (ArowIdx<m) {
        for (int shdmemColIdx=0;shdmemColIdx<TILE_SZ_B_LEFT_REG;shdmemColIdx++) {
            int CcolIdx=shdmemColIdx+blockIdx.y*TILE_SZ_B_LEFT_REG;
            if (CcolIdx<n) {
                for (int j = 0; j < TILE_SZ_RATIO_LEFT_REG; j++) {
                    C(ArowIdx, CcolIdx)+=reg[j]*shmem[j][shdmemColIdx];
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



#define TILE_SZ_B 128
#define TILE_SZ_A 16
#define TILE_SZ_RATIO (TILE_SZ_B/TILE_SZ_A)
// left shmem right reg 
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

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
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]
  __shared__ float shmem[TILE_SZ_A][TILE_SZ_RATIO];
  // INSERT KERNEL CODE HERE

  int BcolIdx = blockIdx.x*TILE_SZ_B + threadIdx.x;
  
  for (int i = 0; i < (k+TILE_SZ_RATIO-1)/TILE_SZ_RATIO; i++) {
    // load B in registers
    float reg[TILE_SZ_RATIO];
    if (BcolIdx < n) {
        for (int j = 0; j < TILE_SZ_RATIO; j++) {
            reg[j] = (i*TILE_SZ_RATIO+j<k)?B(i*TILE_SZ_RATIO+j,BcolIdx):0.0f;
        }
    }
    // load A in shared memory
    constexpr bool A_column_major = true;
    if (A_column_major){
    int shdmemLDAcolIdx = i*TILE_SZ_RATIO+threadIdx.x%TILE_SZ_RATIO;
    int shdmemLDArowIdx =blockIdx.y*TILE_SZ_A + threadIdx.x/TILE_SZ_RATIO;
    shmem[threadIdx.x/TILE_SZ_RATIO][threadIdx.x%TILE_SZ_RATIO] = (shdmemLDAcolIdx<k && shdmemLDArowIdx<m)?A(shdmemLDArowIdx,shdmemLDAcolIdx):0.0f;
    }
    else{
    int shdmemLDAcolIdx = i*TILE_SZ_RATIO+threadIdx.x/TILE_SZ_A;
    int shdmemLDArowIdx =blockIdx.y*TILE_SZ_A + threadIdx.x%TILE_SZ_A;
    shmem[threadIdx.x%TILE_SZ_A][threadIdx.x/TILE_SZ_A] = (shdmemLDAcolIdx<k && shdmemLDArowIdx<m)?A(shdmemLDArowIdx,shdmemLDAcolIdx):0.0f;
    }
    
    __syncthreads();
    // compute C
    if (BcolIdx<n) {
        for (int shdmemRowIdx=0;shdmemRowIdx<TILE_SZ_A;shdmemRowIdx++) {
            int CrowIdx=shdmemRowIdx+blockIdx.y*TILE_SZ_A;
            if (CrowIdx<m) {
                for (int j = 0; j < TILE_SZ_RATIO; j++) {
                    C(CrowIdx, BcolIdx)+=shmem[shdmemRowIdx][j] * reg[j];
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

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
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

    //INSERT CODE HERE
    cudaMemset(C, 0, m*n*sizeof(float));
    // Invoke CUDA kernel -----------------------------------------------------
    // Initialize thread block and kernel grid dimensions ---------------------
    constexpr bool left_reg_flag = false;
    if (left_reg_flag){
        dim3 blocks((m+TILE_SZ_A_LEFT_REG-1)/TILE_SZ_A_LEFT_REG, (n+TILE_SZ_B_LEFT_REG-1)/TILE_SZ_B_LEFT_REG);
        dim3 threadsPerBlock(TILE_SZ_A_LEFT_REG);
        mysgemm_left_reg_right_shmem<<<blocks,threadsPerBlock>>>(m, n, k, A, B, C);
    }
    else{
        // blockDim y = 1, blockDim x = TILE_SZ_B
        // gridDim y = (m+TILE_SZ_A-1)/TILE_SZ_A, gridDim x = (n+TILE_SZ_B-1)/TILE_SZ_B
        dim3 blocks((n+TILE_SZ_B-1)/TILE_SZ_B, (m+TILE_SZ_A-1)/TILE_SZ_A);
        dim3 threadsPerBlock(TILE_SZ_B);
        mysgemm<<<blocks,threadsPerBlock>>>(m, n, k, A, B, C);
    }
    //INSERT CODE HERE

}

