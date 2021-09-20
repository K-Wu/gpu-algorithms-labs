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
    float reg0=0.0f;
    float reg1=0.0f;
    float reg2=0.0f;
    float reg3=0.0f;
    float reg4=0.0f;
    float reg5=0.0f;
    float reg6=0.0f;
    float reg7=0.0f;
    if (ArowIdx < m) {
    reg0 = (k>i*TILE_SZ_RATIO)?A(ArowIdx,i*TILE_SZ_RATIO):0.0f;
    reg1 = (k>i*TILE_SZ_RATIO+1)?A(ArowIdx,i*TILE_SZ_RATIO+1):0.0f;
    reg2 = (k>i*TILE_SZ_RATIO+2)?A(ArowIdx,i*TILE_SZ_RATIO+2):0.0f;
    reg3 = (k>i*TILE_SZ_RATIO+3)?A(ArowIdx,i*TILE_SZ_RATIO+3):0.0f;
    reg4 = (k>i*TILE_SZ_RATIO+4)?A(ArowIdx,i*TILE_SZ_RATIO+4):0.0f;
    reg5 = (k>i*TILE_SZ_RATIO+5)?A(ArowIdx,i*TILE_SZ_RATIO+5):0.0f;
    reg6 = (k>i*TILE_SZ_RATIO+6)?A(ArowIdx,i*TILE_SZ_RATIO+6):0.0f;
    reg7 = (k>i*TILE_SZ_RATIO+7)?A(ArowIdx,i*TILE_SZ_RATIO+7):0.0f;
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
                C(ArowIdx, CcolIdx)+=reg0*shmem[0][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg1*shmem[1][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg2*shmem[2][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg3*shmem[3][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg4*shmem[4][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg5*shmem[5][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg6*shmem[6][shdmemColIdx];
                C(ArowIdx, CcolIdx)+=reg7*shmem[7][shdmemColIdx];
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

