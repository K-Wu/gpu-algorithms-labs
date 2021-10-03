#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include "template.hu"

__device__ static uint64_t linear_count_triangle_for_single_edge(const uint32_t * const edgeDst, uint32_t firstEdgeBeg, uint32_t firstEdgeEnd, uint32_t secondEdgeBeg, uint32_t secondEdgeEnd, uint32_t src, uint32_t dst){
    uint64_t numTriangles = 0;
    uint32_t firstEdgeIdx  = firstEdgeBeg;
    uint32_t secondEdgeIdx = secondEdgeBeg;
    while (firstEdgeIdx < firstEdgeEnd && secondEdgeIdx < secondEdgeEnd) {
      if (edgeDst[firstEdgeIdx] == edgeDst[secondEdgeIdx]) {
        numTriangles++;
        firstEdgeIdx++;
        secondEdgeIdx++;
      } else if (edgeDst[firstEdgeIdx] < edgeDst[secondEdgeIdx]) {
        firstEdgeIdx++;
      } else {
        secondEdgeIdx++;
      }
    }
    return numTriangles;
}

__device__ static uint64_t binary_count_triangle_for_single_edge(const uint32_t * const edgeDst, uint32_t firstEdgeBeg, uint32_t firstEdgeEnd, uint32_t secondEdgeBeg, uint32_t secondEdgeEnd, uint32_t src, uint32_t dst){
  uint64_t numTriangles = 0;
  
  // binary search on the second edge while linear scan on the first edge
  for (uint32_t firstEdgeIdx = firstEdgeBeg; firstEdgeIdx<firstEdgeEnd; firstEdgeIdx++) {
    uint32_t secondEdgeLo = secondEdgeBeg;
    uint32_t secondEdgeHi = secondEdgeEnd - 1;
    //the item is in [lo, hi]
    //adapted from https://www.geeksforgeeks.org/binary-search/
    while(secondEdgeHi>=secondEdgeLo){
      uint32_t secondEdgeCurr = secondEdgeLo+(secondEdgeHi-secondEdgeLo)/2;
      if (edgeDst[firstEdgeIdx] < edgeDst[secondEdgeCurr]) {
        secondEdgeHi = secondEdgeCurr-1;
      } else if (edgeDst[firstEdgeIdx] > edgeDst[secondEdgeCurr]){
        secondEdgeLo = secondEdgeCurr+1;
      }
      else if (edgeDst[firstEdgeIdx] == edgeDst[secondEdgeCurr]){
        numTriangles++;
        break;
      }
    }
  }
  return numTriangles;
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numEdges) {
    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    uint32_t src = edgeSrc[tid];
    uint32_t dst = edgeDst[tid];

    // Determine how many elements of those two arrays are common
    uint32_t firstEdgeBeg  = rowPtr[src];
    uint32_t firstEdgeEnd  = rowPtr[src + 1];
    uint32_t secondEdgeBeg = rowPtr[dst];
    uint32_t secondEdgeEnd = rowPtr[dst + 1];
    uint64_t numTriangles  = linear_count_triangle_for_single_edge(edgeDst, firstEdgeBeg, firstEdgeEnd, secondEdgeBeg, secondEdgeEnd, src, dst);
    
    // Store the number of triangles in the triangle count array
    triangleCounts[tid] = numTriangles;
  }
}

__global__ static void kernel_tc_hybrid(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
  const uint32_t *const edgeSrc,         //!< node ids for edge srcs
  const uint32_t *const edgeDst,         //!< node ids for edge dsts
  const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
  const size_t numEdges                  //!< how many edges to count triangles for
) {

// Determine the source and destination node for the edge
size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < numEdges) {
// Use the row pointer array to determine the start and end of the neighbor list in the column index array
uint32_t src = edgeSrc[tid];
uint32_t dst = edgeDst[tid];
uint32_t firstEdgeBeg  = rowPtr[src];
uint32_t firstEdgeEnd  = rowPtr[src + 1];
uint32_t secondEdgeBeg = rowPtr[dst];
uint32_t secondEdgeEnd = rowPtr[dst + 1];

// binary search on the second edge, therefore first choose the shorter list as first edge
if (rowPtr[src + 1]-rowPtr[src]<rowPtr[dst + 1]-rowPtr[dst]) {
  firstEdgeBeg  = rowPtr[src];
  firstEdgeEnd  = rowPtr[src + 1];
  secondEdgeBeg = rowPtr[dst];
  secondEdgeEnd = rowPtr[dst + 1];
}
else{
  secondEdgeBeg  = rowPtr[src];
  secondEdgeEnd  = rowPtr[src + 1];
  firstEdgeBeg = rowPtr[dst];
  firstEdgeEnd = rowPtr[dst + 1];

}
uint64_t numTriangles;
// Determine how many elements of those two arrays are common
if (secondEdgeEnd-secondEdgeBeg>=64 && (secondEdgeEnd-secondEdgeBeg)/(firstEdgeEnd-firstEdgeBeg)>=6 ){
  numTriangles  = binary_count_triangle_for_single_edge(edgeDst, firstEdgeBeg, firstEdgeEnd, secondEdgeBeg, secondEdgeEnd, src, dst);
}
else{
  numTriangles  = linear_count_triangle_for_single_edge(edgeDst, firstEdgeBeg, firstEdgeEnd, secondEdgeBeg, secondEdgeEnd, src, dst);
}


// Store the number of triangles in the triangle count array
triangleCounts[tid] = numTriangles;
}
}

uint64_t cpu_reduction(uint64_t *triangleCounts, const size_t numEdges) {
  uint64_t result = 0;
  for (size_t idx = 0; idx < numEdges; idx++) {
    result += triangleCounts[idx];
  }
  return result;
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.

  uint64_t total = 0;

  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  // dim3 dimGrid (ceil(number of non-zeros / dimBlock.x))
  dim3 dimGrid(ceil(view.nnz()), dimBlock.x);
  if (mode == 1) {

    //@@ launch the linear search kernel here
    size_t num_edges_h = view.nnz();
    uint64_t *triangleCounts_h;
    uint64_t *triangleCounts;
    
    //CUDA_RUNTIME(cudaMemcpy(&num_edges_h, &(view.nnz()), sizeof(size_t), cudaMemcpyDeviceToHost));
    cudaMalloc((void**)(&triangleCounts), num_edges_h*sizeof(uint64_t));
    kernel_tc<<<dimGrid,dimBlock>>>(triangleCounts, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    
    triangleCounts_h = (uint64_t*) malloc(sizeof(uint64_t)*num_edges_h);
    cudaMemcpy(triangleCounts_h, triangleCounts, sizeof(uint64_t)*num_edges_h, cudaMemcpyDeviceToHost);
    total=cpu_reduction(triangleCounts_h, num_edges_h);
    cudaFree(triangleCounts);
    free(triangleCounts_h);

  } else if (2 == mode) {

    //@@ launch the linear search kernel here
    size_t num_edges_h = view.nnz();
    uint64_t *triangleCounts_h;
    uint64_t *triangleCounts;
    
    //CUDA_RUNTIME(cudaMemcpy(&num_edges_h, &(view.nnz()), sizeof(size_t), cudaMemcpyDeviceToHost));
    cudaMalloc((void**)(&triangleCounts), num_edges_h*sizeof(uint64_t));
    kernel_tc_hybrid<<<dimGrid,dimBlock>>>(triangleCounts, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    
    triangleCounts_h = (uint64_t*) malloc(sizeof(uint64_t)*num_edges_h);
    cudaMemcpy(triangleCounts_h, triangleCounts, sizeof(uint64_t)*num_edges_h, cudaMemcpyDeviceToHost);
    total=cpu_reduction(triangleCounts_h, num_edges_h);
    cudaFree(triangleCounts);
    free(triangleCounts_h);

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  return total;
}
