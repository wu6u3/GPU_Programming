#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__host__ __device__ int outInvariant(int inValue) {
  return inValue * inValue;
}

__host__ __device__ int outDependent(int value, int inIdx, int outIdx) {
  if (inIdx == outIdx) {
    return 2 * value;
  } else if (inIdx > outIdx) {
    return value / (inIdx - outIdx);
  } else {
    return value / (outIdx - inIdx);
  }
}

__global__ void s2g_gpu_gather_kernel(int *in, int *out, int len) {
  int outIdx = blockIdx.x*blockDim.x+threadIdx.x;
  int results=0;
  if(outIdx < len){
    for (int inIdx = 0; inIdx < len; ++inIdx) {
      int intermediate = outInvariant(in[inIdx]);
      results += outDependent(intermediate, inIdx, outIdx);
    }
    out[outIdx]=results;
  }
}

static void s2g_cpu_scatter(int *in, int *out, int len) {
  for (int outIdx = 0; outIdx < len; ++outIdx) {
    for (int inIdx = 0; inIdx < len; ++inIdx) {
      int intermediate = outInvariant(in[inIdx]);
      out[outIdx] += outDependent(intermediate, inIdx, outIdx);
    }
  }
}

static void s2g_gpu_gather(int *in, int *out, int len) {
 
  printf("start gpu scatter\n");
  int bs = 16;
  
  dim3 grid((len+bs-1)/bs, 1, 1);
  dim3 block(bs, 1, 1);
  
  s2g_gpu_gather_kernel<<<grid, block>>>(in, out, len);
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  int *hostInput;
  int *hostOutput;
  int *deviceInput;
  int *deviceOutput;
  size_t byteCount;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (int *)wbImport(wbArg_getInputFile(args, 0), &inputLength,
                              "Integer");
  hostOutput = (int *)malloc(inputLength * sizeof(int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  byteCount = inputLength * sizeof(int);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, byteCount));
  wbCheck(cudaMalloc((void **)&deviceOutput, byteCount));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, byteCount,
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(deviceOutput, 0, byteCount));
  wbTime_stop(GPU, "Copying input memory to the GPU.");


  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  wbTime_start(Compute, "Performing GPU Gather computation");
  s2g_gpu_gather(deviceInput, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing GPU Gather computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, byteCount,
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbSolution(args, hostOutput, inputLength);

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  free(hostInput);
  free(hostOutput);

  return 0;
}
