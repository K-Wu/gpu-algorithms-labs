#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

//#define TILE_SIZE 30

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {
  __shared__ int shdmem[32*32];
  // INSERT KERNEL CODE HERE
  #define A0(i, j, k) A0[((k)*ny + (j))*nx + (i)]
  #define Anext(i, j, k) Anext[((k)*(ny-2) + (j))*(nx-2) + (i)]
  #define shdmem(i, j) shdmem[((j))*nx + (i)]	
  int xOutIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int yOutIdx = blockIdx.y * blockDim.y + threadIdx.y;
  //int xInIdx = blockIdx.x * blockDim.x + threadIdx.x;
  //int yInIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int currZ;
  int nextZ;
  int lastZ;
  if (xOutIdx<nx-2 && yOutIdx<ny-2){
    lastZ = A0(xOutIdx+1, yOutIdx+1, 0);
    currZ = A0(xOutIdx+1, yOutIdx+1, 1);
    nextZ = A0(xOutIdx+1, yOutIdx+1, 2);
  }
  for (int zIdx = 1; zIdx < nz-1; zIdx++){
    if (xOutIdx<nx-2 && yOutIdx<ny-2){
      shdmem(threadIdx.x, threadIdx.y) = A0(xOutIdx+1, yOutIdx+1, zIdx);
    }
    __syncthreads();
    if (xOutIdx<nx-2 && yOutIdx<ny-2){
    int out_val=0;
    out_val += (lastZ+currZ+nextZ+
		(threadIdx.x==0?A0(xOutIdx, yOutIdx+1, zIdx):shdmem(threadIdx.x-1, threadIdx.y))+(threadIdx.x==(blockDim.x-1)?A0(xOutIdx+2,yOutIdx+1,zIdx):shdmem(threadIdx.x+1, threadIdx.y))+
		(threadIdx.y==0?A0(xOutIdx+1, yOutIdx, zIdx):shdmem(threadIdx.x, threadIdx.y-1))+(threadIdx.y==(blockDim.y-1)?A0(xOutIdx+1,yOutIdx+2,zIdx):shdmem(threadIdx.x, threadIdx.y+1)));


    Anext(xOutIdx, yOutIdx, zIdx-1) = out_val;
    
        lastZ = currZ;
	      currZ = nextZ;
    if (zIdx<nz-2)
	      	      nextZ=A0(xOutIdx, yOutIdx, zIdx+2);
    }

    __syncthreads();
  }
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  // INSERT CODE HERE
  dim3 blocks(/*nx*/ (nx-2+31)/32, /*ny*/ (ny-2+31)/32);
  dim3 threadsPerBlock(32, 32);
  kernel<<<blocks, threadsPerBlock>>>(A0, Anext, nx, ny, nz);
}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
