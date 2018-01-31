#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define KERNEL_WIDTH 3 
#define KERNEL_RADIUS 1
#define TILE_WIDTH 4
#define In_Need (KERNEL_WIDTH+TILE_WIDTH-1) //10
__constant__ float kernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  

  int tz = threadIdx.z;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  
  int z_out = blockIdx.z * TILE_WIDTH + tz;
  int y_out = blockIdx.y * TILE_WIDTH + ty;
  int x_out = blockIdx.x * TILE_WIDTH + tx;
  
  int z_r = z_out - KERNEL_RADIUS ;
  int y_r = y_out - KERNEL_RADIUS ;
  int x_r = x_out - KERNEL_RADIUS  ;

  __shared__ float in[In_Need][In_Need][In_Need];
  
  if(z_r >= 0 && z_r < z_size &&
     y_r >= 0 && y_r < y_size &&
     x_r >= 0 && x_r < x_size){
     in[tz][ty][tx]=input[z_r*y_size*x_size+y_r*x_size+x_r];
  }else {
     in[tz][ty][tx]=0.0f;
  }
  __syncthreads();

  float res = 0;
  if(tz<TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH){
    for (unsigned z_kernel = 0; z_kernel< KERNEL_WIDTH;++z_kernel){
      for(unsigned y_kernel = 0; y_kernel< KERNEL_WIDTH;++y_kernel){
        for(unsigned x_kernel = 0; x_kernel< KERNEL_WIDTH;++x_kernel) {
          res += kernel[z_kernel][y_kernel][x_kernel] * in[z_kernel+tz][y_kernel+ty][x_kernel+tx];
        }  
      }
    }
    if(x_out<x_size && y_out< y_size && z_out < z_size)
      output[x_out + (y_out * x_size) + (z_out * y_size * x_size)] = res; 
    __syncthreads();
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");

  cudaMalloc((void**)&deviceInput,z_size * y_size * x_size*sizeof(float));
  cudaMalloc((void**)&deviceOutput,z_size * y_size * x_size*sizeof(float));

  
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  
  cudaMemcpyToSymbol(kernel, hostKernel, KERNEL_WIDTH*KERNEL_WIDTH*KERNEL_WIDTH*sizeof(float));
     
  cudaMemcpy(deviceInput,&hostInput[3],z_size * y_size * x_size*sizeof(float),cudaMemcpyHostToDevice);
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");

  dim3 grid_dim((x_size+TILE_WIDTH-1)/TILE_WIDTH,(y_size+TILE_WIDTH-1)/TILE_WIDTH,(z_size+TILE_WIDTH-1)/TILE_WIDTH);
  dim3 block_dim(In_Need,In_Need,In_Need);
  
  
  conv3d<<<grid_dim,block_dim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(&hostOutput[3],deviceOutput,z_size * y_size * x_size*sizeof(float),cudaMemcpyDeviceToHost);

  /* verify
  std::cout<< (z_size+TILE_WIDTH-1)/TILE_WIDTH<<" "
      << (y_size+TILE_WIDTH-1)/TILE_WIDTH<<" "
      << (x_size+TILE_WIDTH-1)/TILE_WIDTH<<std::endl;
  std::cout<< In_Need<<std::endl;
  for(int i = 0; i < 3 * 3 * 3; ++i){
        if((i % 3) == 0)
            std::cout << std::endl;
        if((i % (3 * 3)) == 0)
            std::cout << std::endl;
        std::cout << hostKernel[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < x_size * y_size * z_size; ++i){
        if((i % x_size) == 0)
            std::cout << std::endl;
        if((i % (x_size * y_size)) == 0)
            std::cout << std::endl;
        std::cout << hostInput[i+3] << " ";
    }
  std::cout << std::endl;
    for(int i = 0; i < x_size * y_size * z_size; ++i){
        if((i % x_size) == 0)
            std::cout << std::endl;
        if((i % (x_size * y_size)) == 0)
            std::cout << std::endl;
        std::cout << hostOutput[i+3] << " ";
    }
  */
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

