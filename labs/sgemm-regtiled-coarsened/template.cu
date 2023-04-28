#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

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
  __shared__ float shmem[TILE_SZ_RATIO][TILE_SZ_B];
  // INSERT KERNEL CODE HERE

  int ArowIdx = blockIdx.x*TILE_SZ_A + threadIdx.x;
  
  for (int i = 0; i < (k+TILE_SZ_RATIO-1)/TILE_SZ_RATIO; i++) {
    // load A in registers
    float reg[TILE_SZ_RATIO];
    if (ArowIdx < m) {
        for (int j = 0; j < TILE_SZ_RATIO; j++) {
            reg[j] = (i*TILE_SZ_RATIO+j<k)?A(ArowIdx,i*TILE_SZ_RATIO+j):0.0f;
        }
    }
    // load B in shared memory
    int shdmemLDBrowIdx = i*TILE_SZ_RATIO+threadIdx.x/TILE_SZ_B;
    int shdmemLDBcolIdx =blockIdx.y*TILE_SZ_B + threadIdx.x%TILE_SZ_B;
    shmem[threadIdx.x/TILE_SZ_B][threadIdx.x%TILE_SZ_B] = (shdmemLDBrowIdx<k && shdmemLDBcolIdx<n)?B(shdmemLDBrowIdx,shdmemLDBcolIdx):0.0f;
    
    __syncthreads();
    // compute C
    if (ArowIdx<m) {
        for (int shdmemColIdx=0;shdmemColIdx<TILE_SZ_B;shdmemColIdx++) {
            int CcolIdx=shdmemColIdx+blockIdx.y*TILE_SZ_B;
            if (CcolIdx<n) {
                for (int j = 0; j < TILE_SZ_RATIO; j++) {
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

    // Initialize thread block and kernel grid dimensions ---------------------
    dim3 blocks((m+TILE_SZ_A-1)/TILE_SZ_A, (n+TILE_SZ_B-1)/TILE_SZ_B);
    dim3 threadsPerBlock(TILE_SZ_A);

    // Your code need only consider the m, n, k, A, B, and C parameters of
    // the function, which provide the matrix sizes (m, n, k) and data
    // (A, B, C).

    //INSERT CODE HERE
    cudaMemset(C, 0, m*n*sizeof(float));
    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<blocks,threadsPerBlock>>>(m, n, k, A, B, C);
    //INSERT CODE HERE

}

