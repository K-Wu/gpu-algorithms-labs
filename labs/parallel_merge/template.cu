#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))

__device__ int co_rank(float* A, int A_len, float* B, int B_len, int k){
    int A_lo = max2(0, k - B_len);
    int A_hi = min2(A_len, k);
    int B_lo = max2(0, k - A_len);
    int B_hi = k-A_hi;
    bool flag = true;
    while(flag){
        if (A_hi>0 && B_hi<B_len && A[A_hi-1] > B[B_hi]){
            int delta = ceil_div(A_hi-A_lo,2);
            A_hi = A_hi-delta;
            B_lo = B_hi;
            B_hi = B_hi+delta;
            
        }
        else if (A_hi<A_len && B_hi>0 && B[B_hi-1] >= A[A_hi]){
            int delta = ceil_div(B_hi-B_lo,2);
            A_lo = A_hi;
            A_hi = A_hi+delta;
            B_hi = B_hi-delta;
        }
        else{
            flag = false;
        }
    }
    return A_hi;
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    int k_curr = tid*ceil_div(A_len+B_len, gridDim.x*blockDim.x);
    int k_next = min2(A_len+B_len,(tid+1)*ceil_div(A_len+B_len, gridDim.x*blockDim.x));
    int A_beg = co_rank(A, A_len, B, B_len, k_curr);
    int A_end = co_rank(A, A_len, B, B_len, k_next);
    int B_beg = k_curr - A_beg;
    int B_end = k_next - A_end;
    merge_sequential(&A[A_beg], A_end-A_beg, &B[B_beg], B_end-B_beg, &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    __shared__ float A_tiled[TILE_SIZE];
    __shared__ float B_tiled[TILE_SIZE];
    __shared__ float C_tiled[TILE_SIZE];
    __shared__ int block_pointers[4];//A_beg, A_end, B_beg, B_end
    //int k_curr = blockIdx.x*ceil_div(A_len+B_len, gridDim.x*TILE_SIZE);
    for (int k_curr = TILE_SIZE*blockIdx.x;k_curr<A_len+B_len;k_curr+=gridDim.x*TILE_SIZE){
        int k_next = min2(A_len+B_len,k_curr+TILE_SIZE);
        if (threadIdx.x == 0){
            block_pointers[0] = co_rank(A, A_len, B, B_len, k_curr);
            block_pointers[1] = co_rank(A, A_len, B, B_len, k_next);
            block_pointers[2] = k_curr - block_pointers[0];
            block_pointers[3] = k_next - block_pointers[1];
        }
        __syncthreads();
        for(int load_loopidx = 0;load_loopidx<ceil_div(TILE_SIZE,BLOCK_SIZE);load_loopidx++){
            int load_idx=load_loopidx*BLOCK_SIZE+threadIdx.x;
            if (load_idx<TILE_SIZE){
                if (block_pointers[0]+load_idx<A_len&& load_idx<block_pointers[1]-block_pointers[0]){
                    A_tiled[load_idx] = A[block_pointers[0]+load_idx];
                }
                if (block_pointers[2]+load_idx<B_len&& load_idx<block_pointers[3]-block_pointers[2]){
                    B_tiled[load_idx] = B[block_pointers[2]+load_idx];
                }
            }
        }
        __syncthreads();
        
        int thread_k_curr = min2(k_next, k_curr + ceil_div(TILE_SIZE,BLOCK_SIZE)*threadIdx.x);
        int thread_k_next = min2(k_next, k_curr + ceil_div(TILE_SIZE,BLOCK_SIZE)*(threadIdx.x+1));
        int thread_A_beg = co_rank(A, A_len, B, B_len, thread_k_curr);
        int thread_A_end = co_rank(A, A_len, B, B_len, thread_k_next);
        int thread_B_beg = thread_k_curr - thread_A_beg;
        int thread_B_end = thread_k_next - thread_A_end;
        merge_sequential(&A_tiled[thread_A_beg-block_pointers[0]], thread_A_end-thread_A_beg, &B_tiled[thread_B_beg-block_pointers[2]], thread_B_end-thread_B_beg, &C_tiled[thread_k_curr-k_curr]);
        __syncthreads();
        for (int store_loopidx = 0;store_loopidx<ceil_div(TILE_SIZE,BLOCK_SIZE);store_loopidx++){
            int store_idx=store_loopidx*BLOCK_SIZE+threadIdx.x;
            if (store_idx<TILE_SIZE){
                if (k_curr+store_idx<A_len+B_len){
                    C[k_curr+store_idx] = C_tiled[store_idx];
                }
            }
        }
        __syncthreads();
    }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
