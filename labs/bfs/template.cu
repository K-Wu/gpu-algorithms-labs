#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                           unsigned int *nodeNeighbors,
                                           unsigned int *nodeVisited,
                                           unsigned int *currLevelNodes,
                                           unsigned int *nextLevelNodes,
                                           unsigned int *numCurrLevelNodes,
                                           unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the current level
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (unsigned int i = tid; i < *numCurrLevelNodes; i += blockDim.x * gridDim.x) {
    // Loop over all neighbors of the node
    unsigned int node = currLevelNodes[i];
    for (unsigned int j = nodePtrs[node]; j < nodePtrs[node + 1]; j++) {
      unsigned int neighbor = nodeNeighbors[j];
      // If neighbor hasn't been visited yet
      // Mark it as visited
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        // Add neighbor to global queue
        unsigned int oldIdx    = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[oldIdx] = neighbor;
      }
    }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue (size should be BQ_CAPACITY)
  __shared__ unsigned int sharedQueue[BQ_CAPACITY];
  __shared__ unsigned int sharedQueueHead;
  __shared__ unsigned int globalQueueCopyBeg;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadIdx.x == 0)
    sharedQueueHead = 0;
  __syncthreads();

  // Loop over all nodes in the current level
  for (unsigned int i = tid; i < *numCurrLevelNodes; i += blockDim.x * gridDim.x) {
    // Loop over all neighbors of the node
    unsigned int node = currLevelNodes[i];
    for (unsigned int j = nodePtrs[node]; j < nodePtrs[node + 1]; j++) {
      unsigned int neighbor = nodeNeighbors[j];
      // If neighbor hasn't been visited yet
      // Mark it as visited
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        // Add neighbor to block queue
        unsigned int oldIdx = atomicAdd(&sharedQueueHead, 1);
        // If full, add neighbor to global queue
        if (oldIdx >= BQ_CAPACITY) {
          unsigned int oldGlobalIdx    = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[oldGlobalIdx] = neighbor;
        } else {
          sharedQueue[oldIdx] = neighbor;
        }
      }
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    if (sharedQueueHead > BQ_CAPACITY) {
      sharedQueueHead = BQ_CAPACITY;
    }
    globalQueueCopyBeg = atomicAdd(numNextLevelNodes, sharedQueueHead);
  }
  __syncthreads();
  // Store block queue in global queue
  for (unsigned int i = threadIdx.x; i < sharedQueueHead; i += blockDim.x) {
    nextLevelNodes[globalQueueCopyBeg + i] = sharedQueue[i];
  }
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses NUM_WARP_QUEUES warp queues of capacity
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.
  __shared__ unsigned int warpQueue[WQ_CAPACITY][NUM_WARP_QUEUES];
  __shared__ unsigned int warpQueueHead[NUM_WARP_QUEUES];
  __shared__ unsigned int sharedQueueCopyBeg[NUM_WARP_QUEUES];

  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.
  __shared__ unsigned int sharedQueue[BQ_CAPACITY];
  __shared__ unsigned int sharedQueueHead;
  __shared__ unsigned int globalQueueCopyBeg;

  // Initialize shared memory queues (warp and block)
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadIdx.x == 0)
    sharedQueueHead = 0;
  if (threadIdx.x < NUM_WARP_QUEUES) {
    warpQueueHead[threadIdx.x] = 0;
  }
  __syncthreads();

  // Loop over all nodes in the current level
  for (unsigned int i = tid; i < *numCurrLevelNodes; i += blockDim.x * gridDim.x) {
    // Loop over all neighbors of the node
    unsigned int node = currLevelNodes[i];
    for (unsigned int j = nodePtrs[node]; j < nodePtrs[node + 1]; j++) {
      unsigned int neighbor = nodeNeighbors[j];
      // If neighbor hasn't been visited yet
      // Mark it as visited
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        // Add neighbor to warp queue
        unsigned int oldIdx = atomicAdd(&warpQueueHead[threadIdx.x % 8], 1);
        // If full, add neighbor to block queue
        if (oldIdx >= WQ_CAPACITY) {
          unsigned int oldBlockIdx = atomicAdd(&sharedQueueHead, 1);
          // If full, add neighbor to global queue
          if (oldBlockIdx >= BQ_CAPACITY) {
            unsigned int oldGlobalIdx    = atomicAdd(numNextLevelNodes, 1);
            nextLevelNodes[oldGlobalIdx] = neighbor;
          } else {
            sharedQueue[oldBlockIdx] = neighbor;
          }
        } else {
          warpQueue[oldIdx][threadIdx.x % 8] = neighbor;
        }
      }
    }
  }
  __syncthreads();

  // Allocate space for warp queue to go into block queue
  if (threadIdx.x < NUM_WARP_QUEUES) {
    if (warpQueueHead[threadIdx.x] > WQ_CAPACITY) {
      warpQueueHead[threadIdx.x] = WQ_CAPACITY;
    }
    sharedQueueCopyBeg[threadIdx.x] = atomicAdd(&sharedQueueHead, warpQueueHead[threadIdx.x]);
  }
  __syncthreads();

  // Store warp queues in block queue (use one warp or one thread per queue)
  // Add any nodes that don't fit (remember, space was allocated above)
  //    to the global queue
  for (unsigned int idxWarpQueue = 0; idxWarpQueue < NUM_WARP_QUEUES; idxWarpQueue++) {
    for (unsigned int i = threadIdx.x; i < warpQueueHead[idxWarpQueue]; i += blockDim.x) {
      if (sharedQueueCopyBeg[idxWarpQueue] + i >= BQ_CAPACITY) {
        unsigned int oldGlobalIdx    = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[oldGlobalIdx] = warpQueue[i][idxWarpQueue];
      } else {
        sharedQueue[sharedQueueCopyBeg[idxWarpQueue] + i] = warpQueue[i][idxWarpQueue];
      }
    }
  }
  __syncthreads();

  // Saturate block queue counter (too large if warp queues overflowed)
  // Allocate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    if (sharedQueueHead > BQ_CAPACITY) {
      sharedQueueHead = BQ_CAPACITY;
    }
    globalQueueCopyBeg = atomicAdd(numNextLevelNodes, sharedQueueHead);
  }
  __syncthreads();

  // Store block queue in global queue
  for (unsigned int i = threadIdx.x; i < sharedQueueHead; i += blockDim.x) {
    nextLevelNodes[globalQueueCopyBeg + i] = sharedQueue[i];
  }
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors, unsigned int *nodeVisited, unsigned int *currLevelNodes,
                         unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
                                                        numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors, unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
                                                       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors, unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
                                                      numCurrLevelNodes, numNextLevelNodes);
}
