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

__device__ static float Clamp(float val, float start,float end){ 
  
  float minv = ( val < end )? val: end;
  return (minv> start)? minv : start;
}
static float clamp(float val, float start,float end){ 
  
  float minv = ( val < end )? val: end;
  return (minv> start)? minv : start;
}
void stencil_cpu(float *_out, float *_in, int width, int height, int depth) {

#define out(i, j, k) _out[((i)*width + (j)) * depth + (k)]
#define in(i, j, k) _in[((i)*width + (j)) * depth + (k)]

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        out(i, j, k) = in(i, j, k + 1) + in(i, j, k - 1) +
                       in(i, j + 1, k) + in(i, j - 1, k) +
                       in(i + 1, j, k) + in(i - 1, j, k) - 6 * in(i, j, k);
        out(i,j,k) = clamp(out(i,j,k), 0, 255);
      }
    }
  }
  
#undef out
#undef in
}
#define TILE_SIZE 8

__global__ void stencil(float *output, float *input, int width, int height,
                        int depth) {
  
  #define in(i, j, k)   input[((i)*width + (j))*depth +(k)]
  #define out(i, j, k) output[((i)*width + (j))*depth +(k)]
  int i = blockIdx.x*blockDim.x+threadIdx.x; //x height
  int j = blockIdx.y*blockDim.y+threadIdx.y; //y width
  int tx = threadIdx.x;
  int ty = threadIdx.y; 
  
  __shared__   float  ds_A[TILE_SIZE][TILE_SIZE];
  float bottom = in(i, j, 0);
  float current = in(i, j, 1);
  ds_A[ty][tx] = current;
  float top = in(i, j, 2);
  __syncthreads();
  
  
  for (unsigned k = 1; k < depth - 1; k++) {
    if (i > 0 && i < height-1 && j > 0 && j < width -1 ) {
    out(i, j, k) = Clamp(bottom + top +
	               ((tx > 0) ? ds_A[ty][tx-1]: in(i-1, j, k)) +
	  		        ((tx < TILE_SIZE-1) ? ds_A[ty][tx+1]: in(i+1, j, k)) +
	  		        ((ty > 0)?  ds_A[ty-1][tx]:  in(i, j-1, k)) +
	  		        ((ty < TILE_SIZE-1 )? ds_A[ty+1][tx]: in(i, j+1, k)) -
			          6 * current, 0, 255);
    }
    __syncthreads();
    bottom = current;     
    ds_A[ty][tx] = top;     
    current = top;     
    top = in(i, j, k+2) ; 
    __syncthreads();
    
	/* barrier synch, load top */
  }

  #undef out
  #undef in
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  int bs = TILE_SIZE;
  
  dim3 grid((height+bs-1)/bs, (width+bs-1)/bs, 1);
  dim3 block(bs, bs, 1);
  
  stencil<<<grid, block>>>(deviceOutputData, deviceInputData, width, height, depth);
  
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  width = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData,
             width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData,
             width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData,
             width * height * depth * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  
  
  
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData,
             width * height * depth * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");
  //stencil_cpu(hostOutputData, hostInputData, width, height, depth);
  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}
