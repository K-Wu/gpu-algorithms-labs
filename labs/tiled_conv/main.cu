#include "helper.hpp"

// Sequential code for the forward path of the convolution layer
// You should not modify this code
static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  std::fill(Y, Y + ydims.flattened_length(), 0);

  for (auto i : range(0, ydims.num)) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
                                             const shape ydims) {

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }
  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

#define INTEGER_CEILING_DIVIDE(y, x) (((y) + (x) -1) / (x))
const int TILE_WIDTH_SHARED_MATRIX_MULTIPLY_Y = 32; // the height of the output matrix
const int TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X = 32; // the width of the output matrix
const int TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K = 8;  // the length of the intermediate dimension in A*B
const int COARSENING_FACTOR                   = 4;
const int BLOCKDIM_Y                          = TILE_WIDTH_SHARED_MATRIX_MULTIPLY_Y / COARSENING_FACTOR;
const int BLOCKDIM_X                          = TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X;
const int BLOCKDIM_Z                          = 1;

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {
  const int B    = ydims.num;
  const int Kh   = wdims.height;
  const int Kw   = wdims.width;
  const int H_in = ydims.height + Kh - 1;
  const int W_in = ydims.width + Kw - 1;
  assert(ydims.height + Kh - 1 == xdims.height);
  assert(ydims.width + Kw - 1 == xdims.width);
  const int C     = xdims.depth;
  const int M     = ydims.depth;
  const int H_out = H_in - Kh + 1;
  const int W_out = W_in - Kw + 1;

  //@@ YOUR CODE HERE!
  // The code is based on the unrolled uncoarensened conv kernel from the final project implementation by our group of ECE 408 19 Fall,
  // where I was the main contributor to every optimization.

  int idxXUnrolledCol = blockIdx.x * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X + threadIdx.x;
  int idxM            = blockIdx.y * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_Y + threadIdx.y;

  int numXUnrolledRows    = C * Kh * Kw;
  int numXUnrolledColumns = H_out * W_out;

  __shared__ float blockMemA[TILE_WIDTH_SHARED_MATRIX_MULTIPLY_Y][TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K];
  __shared__ float blockMemB[TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K][TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X];

  int idxB    = blockIdx.z * BLOCKDIM_Z + threadIdx.z;
  int idxHOut = idxXUnrolledCol / W_out;
  int idxWOut = idxXUnrolledCol % W_out;

  if (idxB < B) {
    float PValue0 = 0.0;
    float PValue1 = 0.0;
    float PValue2 = 0.0;
    float PValue3 = 0.0;

    for (int idx = 0; idx < INTEGER_CEILING_DIVIDE(numXUnrolledRows, TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K); idx++) {

      // const int B = ydims.num;
      // const int Kh = wdims.height;
      // const int Kw = wdims.width;
      // const int H_in = ydims.height +Kh-1;
      // const int W_in = ydims.width + Kw-1;
      // assert(ydims.height+Kh-1 == xdims.height);
      // assert(ydims.width+Kw-1 == xdims.width);
      // const int C = xdims.depth;
      // const int M = ydims.depth;

#define y4d(i3, i2, i1, i0) Y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) X[(i3) * (C * H_in * W_in) + (i2) * (H_in * W_in) + (i1) * (W_in) + i0]
#define k4d(i3, i2, i1, i0) W[(i3) * (C * Kh * Kw) + (i2) * (Kh * Kw) + (i1) * (Kw) + i0]

      const int threadIdx1D      = threadIdx.y * BLOCKDIM_X + threadIdx.x;
      const int threadNumInBlock = BLOCKDIM_X * BLOCKDIM_Y;
      for (int idxLoad = 0;
           idxLoad < INTEGER_CEILING_DIVIDE(TILE_WIDTH_SHARED_MATRIX_MULTIPLY_Y * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K, threadNumInBlock);
           idxLoad++) {
        int idxYForBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) / TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K;
        int idxXForBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) % TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K;
        int idxWColumnToLoad     = idx * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K + idxXForBlockMemALoad;
        int idxCWColumn          = idxWColumnToLoad / (Kh * Kw);
        int idxMToLoad           = blockIdx.y * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_Y + idxYForBlockMemALoad;
        if ((idxCWColumn < C) && (idxMToLoad < M)) {
          int idxKHeightWColumn                                 = (idxWColumnToLoad % (Kh * Kw)) / Kw;
          int idxKWidthWColumn                                  = (idxWColumnToLoad % (Kh * Kw)) % Kw;
          blockMemA[idxYForBlockMemALoad][idxXForBlockMemALoad] = k4d(idxMToLoad, idxCWColumn, idxKHeightWColumn, idxKWidthWColumn);
        } else {
          blockMemA[idxYForBlockMemALoad][idxXForBlockMemALoad] = 0.0;
        }
      }
      for (int idxLoad = 0;
           idxLoad < INTEGER_CEILING_DIVIDE(TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X, threadNumInBlock);
           idxLoad++) {
        int idxYForBlockMemBLoad  = (threadIdx1D + idxLoad * threadNumInBlock) / TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X;
        int idxXForBlockMemBLoad  = (threadIdx1D + idxLoad * threadNumInBlock) % TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X;
        int idxXUnrolledColToLoad = blockIdx.x * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_X + idxXForBlockMemBLoad;
        int idxHOutToLoad         = idxXUnrolledColToLoad / W_out;
        int idxWOutToLoad         = idxXUnrolledColToLoad % W_out;
        int idxXUnrolledRowToLoad = (idxYForBlockMemBLoad + idx * TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K);
        int idxCXUnrolledRow      = idxXUnrolledRowToLoad / (Kh * Kw);
        if ((idxCXUnrolledRow < C) && (idxHOutToLoad < H_out)) {
          int idxKHeightXUnrolledRow = (idxXUnrolledRowToLoad % (Kh * Kw)) / Kw;
          int idxKWidthXUnrolledRow  = (idxXUnrolledRowToLoad % (Kh * Kw)) % Kw;
          blockMemB[idxYForBlockMemBLoad][idxXForBlockMemBLoad] =
              x4d(idxB, idxCXUnrolledRow, idxHOutToLoad + idxKHeightXUnrolledRow, idxWOutToLoad + idxKWidthXUnrolledRow);
        } else {
          blockMemB[idxYForBlockMemBLoad][idxXForBlockMemBLoad] = 0.0;
        }
      }
      __syncthreads();
      for (int idx2 = 0; idx2 < TILE_WIDTH_SHARED_MATRIX_MULTIPLY_K; idx2++) {
        PValue0 += blockMemA[threadIdx.y][idx2] * blockMemB[idx2][threadIdx.x];
        PValue1 += blockMemA[threadIdx.y + BLOCKDIM_Y][idx2] * blockMemB[idx2][threadIdx.x];
        PValue2 += blockMemA[threadIdx.y + 2 * BLOCKDIM_Y][idx2] * blockMemB[idx2][threadIdx.x];
        PValue3 += blockMemA[threadIdx.y + 3 * BLOCKDIM_Y][idx2] * blockMemB[idx2][threadIdx.x];
      }
      __syncthreads();
    }
    if (idxM < M && idxXUnrolledCol < numXUnrolledColumns) {
      y4d(idxB, idxM, idxHOut, idxWOut)                         = PValue0;
      y4d(idxB, idxM + BLOCKDIM_Y, idxHOut, idxWOut)     = PValue1;
      y4d(idxB, idxM + 2 * BLOCKDIM_Y, idxHOut, idxWOut) = PValue2;
      y4d(idxB, idxM + 3 * BLOCKDIM_Y, idxHOut, idxWOut) = PValue3;
    }
  }

#undef y4d
#undef x4d
#undef k4d
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {

  // Modify this code to configure your optimized kernel.
  //@@ YOUR CODE HERE!!!
  // dim3 dimGrid(1);
  // dim3 dimBlock(32);

  const int Kh            = wdims.height;
  const int Kw            = wdims.width;
  const int M             = ydims.depth;
  const int H_in          = ydims.height + Kh - 1;
  const int W_in          = ydims.width + Kw - 1;
  const int H_out         = H_in - Kh + 1;
  const int W_out         = W_in - Kw + 1;
  const int B             = ydims.num;
  int numWtiledRows       = M;
  int numXUnrolledColumns = H_out * W_out;
  dim3 blockUnrollDimGolden(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
  dim3 gridUnrollDimGolden(INTEGER_CEILING_DIVIDE(numXUnrolledColumns, BLOCKDIM_X),
                           INTEGER_CEILING_DIVIDE(numWtiledRows, BLOCKDIM_Y * COARSENING_FACTOR), INTEGER_CEILING_DIVIDE(B, BLOCKDIM_Z));
  THROW_IF_ERROR(cudaMemset(Y, 0, sizeof(float) * ydims.num * ydims.depth * ydims.height * ydims.width));
  conv_forward_opt_kernel<<<gridUnrollDimGolden, blockUnrollDimGolden>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

static int eval(const shape wDims, const shape xDims, bool doVerify) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," + std::to_string(wDims.depth) + "," +
                         std::to_string(wDims.height) + "," + std::to_string(wDims.width) + " xDims:" + std::to_string(xDims.num) + "," +
                         std::to_string(xDims.depth) + "," + std::to_string(xDims.height) + "," + std::to_string(xDims.width) + "]";
  INFO("Running " << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1), (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY    = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);

  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **) &deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceY, yByteCount));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  if (doVerify) {
    timer_start("Copying output to the CPU");
    THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
    timer_stop();

    convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
    // conv_forward_valid(hostX, xDims, hostW, wDims, expected, ydims);
    verify(expected, hostY, ydims);
  }

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}

TEST_CASE("Convlayer", "[convlayer]") {
#if 1
  // test five times in case code errors depend on data
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
#else
  SECTION("[wDims:32,1,5,5 xDims:50000,1,28,28]") {
    eval({32, 1, 5, 5}, {50000, 1, 28, 28}, false);
  }
#endif
}
