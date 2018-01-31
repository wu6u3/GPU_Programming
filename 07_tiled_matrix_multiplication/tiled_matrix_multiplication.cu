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

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  const int tile_w = 8;
  
  __shared__ float sub_A[tile_w][tile_w];
  __shared__ float sub_B[tile_w][tile_w];
  
  int tx=threadIdx.x;
  int ty=threadIdx.y; 
  int bx=blockIdx.x;
  int by=blockIdx.y;
  
  int row = by * tile_w + ty;
  int col = bx * tile_w + tx;
    
  float Cvalue=0.0;
  for (int m=0;m<(numAColumns+tile_w-1)/tile_w;++m){
    if (m*tile_w + tx < numAColumns && row < numARows )
      sub_A[ty][tx] = A[row*numAColumns + m*tile_w+tx];
    else 
      sub_A[ty][tx] =0.0;
    if(m*tile_w + ty < numBRows && col < numBColumns ){
      sub_B[ty][tx] = B[(m*tile_w+ty)*numBColumns+col];  
    }
    else 
      sub_B[ty][tx] =0.0;
    __syncthreads();
    for (int k=0;k<tile_w;++k){
      Cvalue+=sub_A[ty][k]*sub_B[k][tx];
    }
    __syncthreads();
  }
  if(row<numCRows && col < numCColumns)
    C[row*numCColumns+col]=Cvalue;

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns  
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  /
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA,numARows*numAColumns*sizeof(float));
  cudaMalloc((void**)&deviceB,numBRows*numBColumns*sizeof(float));
  cudaMalloc((void**)&deviceC,numCRows*numCColumns*sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,numARows*numAColumns*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,numBRows*numBColumns*sizeof(float),cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  float block_size = 8.0;
  dim3 grid_dim(ceil(numCColumns/block_size),ceil(numCRows/block_size),1);
  dim3 block_dim(block_size,block_size,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
	matrixMultiplyShared <<<grid_dim,block_dim>>> (deviceA, deviceB, deviceC, 
                                           numARows,numAColumns,  
                                           numBRows,numBColumns,  
                                           numCRows,numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,numCRows*numCColumns*sizeof(float),cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

