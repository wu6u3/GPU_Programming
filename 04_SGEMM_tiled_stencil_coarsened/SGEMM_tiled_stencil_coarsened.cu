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
#define TILE_WIDTH 16
__global__ void matrixMultiply_kernel(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns) {
  __shared__ float sub_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sub_B[TILE_WIDTH][TILE_WIDTH];
  int tx=threadIdx.x;  int ty=threadIdx.y; 
  int bx=blockIdx.x;   int by=blockIdx.y;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  
  float next_A = A[row+(1*TILE_WIDTH+tx)*numAColumns];
  float current_A = A[row+(0*TILE_WIDTH+tx)*numAColumns];
  float next_B = B[(1*TILE_WIDTH+ty)*numBColumns+col];
  float current_B = B[(0*TILE_WIDTH+ty)*numBColumns+col];

  if (0*TILE_WIDTH + tx < numARows)
    sub_A[ty][tx] = current_A;    
  else 
    sub_A[ty][tx] =0.0f;
  if(0*TILE_WIDTH + ty < numBRows){
    sub_B[ty][tx] = current_B;  
  }
  else 
    sub_B[ty][tx] =0.0f;
  
  __syncthreads();
  
  float Cvalue=0.0f;
  
  for (int m=0;m<(numARows+TILE_WIDTH-1)/TILE_WIDTH+1;++m){  
    if(  (m-1)*TILE_WIDTH + ty < numBRows
      && (m-1)*TILE_WIDTH + tx < numARows ){
      for (int k=0;k<TILE_WIDTH;++k){
          Cvalue+=sub_A[ty][k]*sub_B[k][tx];
      }
     }
    __syncthreads();
    
    sub_A[ty][tx] = next_A;
    //((m+1)*TILE_WIDTH + tx < numARows && row < numAColumns)?A[row+((m+1)*TILE_WIDTH+tx)*numAColumns]:0.0f;
    //current_A = next_A;    
    sub_B[ty][tx] = next_B;
    //((m+1)*TILE_WIDTH + ty < numBRows && col < numBColumns)?B[((m+1)*TILE_WIDTH+ty)*numBColumns+col]:0.0f;
    //current_B = next_B;
      
    next_A = ((m+2)*TILE_WIDTH + tx < numARows && row < numAColumns)?A[row+((m+2)*TILE_WIDTH+tx)*numAColumns]:0.0f;    
    next_B = ((m+2)*TILE_WIDTH + ty < numBRows && col < numBColumns)?B[((m+2)*TILE_WIDTH+ty)*numBColumns+col]:0.0f;
     
    __syncthreads();
  }
  
  if(row<numCRows && col < numCColumns)
    C[row*numCColumns+col]=Cvalue;
  
}

static void matrixMultiply(float *A, float *B, float *C, int numARows,
                           int numAColumns, int numBRows, int numBColumns,
                           int numCRows, int numCColumns) {
  //@@ Insert code to launch matrix multiplication
  dim3 grid_dim((numCColumns+TILE_WIDTH-1)/TILE_WIDTH,(numCRows+TILE_WIDTH-1)/TILE_WIDTH,1);
  dim3 block_dim(TILE_WIDTH,TILE_WIDTH,1);
  
  //@@ Launch the GPU Kernel here
	matrixMultiply_kernel <<<grid_dim,block_dim>>> (A, B, C, 
                                           numARows,numAColumns,  
                                           numBRows,numBColumns,  
                                           numCRows,numCColumns);
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
  numCRows = numAColumns;
  numCColumns = numBColumns;
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void **)&deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void **)&deviceC, sizeof(float) * numCRows * numCColumns);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply(deviceA, deviceB, deviceC, numARows, numAColumns,
                 numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
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
