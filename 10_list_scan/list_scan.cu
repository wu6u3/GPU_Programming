// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__device__ void sub_scan(float* T, unsigned int t) { 
  
  unsigned int stride=1;
  while(stride < BLOCK_SIZE){
    int index = (t+1)*stride*2 - 1;
    if(index < BLOCK_SIZE){      
      T[index] += T[index-stride];       
    }
    stride = stride*2;
    __syncthreads();
  }
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    int index = (t+1)*stride*2 - 1;
    if(index < BLOCK_SIZE&&index+stride<BLOCK_SIZE) {       
      T[index+stride] += T[index];        
    }        
    stride = stride/2;        
    __syncthreads();    
  }


}

__global__ void post_add(float *inout, float *S,int len){

  unsigned int t = threadIdx.x;    
  unsigned int start = blockDim.x*blockIdx.x;

  
  if(start+t<len&&blockIdx.x>0){
    inout[start+t]+=S[blockIdx.x-1];
  }
  
}
__global__ void scan(float *input, float *output, int len) {
  
  __shared__ float T[BLOCK_SIZE];
  unsigned int t = threadIdx.x;    
  unsigned int start = blockDim.x*blockIdx.x;
  
  if (start+t<len){
    T[t]=input[start+t];
  }else {
    T[t]=0.0f;
  }
  __syncthreads();
  sub_scan(T,t);
  
  if (start+t<len){
    output[start+t]=T[t];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceSin;
  
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  cudaMalloc((void **)&deviceSin, (numElements+BLOCK_SIZE-1)/BLOCK_SIZE* sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 grid_dim((numElements+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);
  dim3 block_dim(BLOCK_SIZE,1,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<grid_dim,block_dim>>>(deviceInput,deviceOutput,numElements);
  
  cudaDeviceSynchronize();
  
  if((numElements+BLOCK_SIZE-1)/BLOCK_SIZE>1){
    int len = (numElements+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(unsigned int i=0;i<(numElements+BLOCK_SIZE-1)/BLOCK_SIZE;++i){
      cudaMemcpy(&deviceSin[i], &deviceOutput[(i+1)*BLOCK_SIZE-1], sizeof(float),
                     cudaMemcpyDeviceToDevice);
    }
    dim3 grid_dim2((len+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);
    scan<<<grid_dim2,block_dim>>>(deviceSin,deviceSin,(numElements+BLOCK_SIZE-1)/BLOCK_SIZE);
    cudaDeviceSynchronize();
  
    post_add<<<grid_dim,block_dim>>>(deviceOutput,deviceSin,numElements);
    cudaDeviceSynchronize();
  }
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost);
  
  
  //wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
  //                   cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

