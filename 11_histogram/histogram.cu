// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512
#define BLOCK_SIZE_2D 32
//#define p(x,width,height) (x / (width * height))


__global__ void cast_image(float * inputImage, unsigned char *ucharImage,unsigned int len){
  unsigned int ii = blockDim.x*blockIdx.x+threadIdx.x;
  if (ii<len){
    ucharImage[ii] = (unsigned char) (255 * inputImage[ii]);
  } 
}
__global__ void rgb_to_gray( unsigned char *ucharImage, unsigned char *grayImage,
                            unsigned int height,unsigned int width){
  unsigned int ii = blockDim.y*blockIdx.y+threadIdx.y;
  unsigned int jj = blockDim.x*blockIdx.x+threadIdx.x;
  if(ii<height){
    if(jj<width){
        unsigned int idx = ii * width + jj;
        unsigned char r = ucharImage[3*idx];
        unsigned char g = ucharImage[3*idx+1];
        unsigned char b = ucharImage[3*idx+2];
        grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
  }   
}
__global__ void scan(unsigned int *input, float *output, int len, 
                             unsigned int height,unsigned int width) {
  __shared__ float T[BLOCK_SIZE];

  unsigned int t = threadIdx.x;    
  unsigned int start = blockDim.x*blockIdx.x;
  if (start+t<len){
    T[t]=(float)input[start+t]/ (width* height);
  }else {
    T[t]=0.0f;
  }
  __syncthreads(); 
  
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
  
  if (start+t<len){    
    output[start+t]=T[t];
  }
  
}

__global__ void histogram_privatized_kernel(unsigned int * histogram, unsigned char *grayImage,unsigned int len) {
  // Privatized bins
   __shared__ unsigned int histo_private[256];  
  
  if (threadIdx.x < 256) 
    histo_private[threadIdx.x] = 0;  
  __syncthreads();
 unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;// stride is total number of threads    
 int stride = blockDim.x * gridDim.x;    
  while (i < len) {         
    atomicAdd( &(histo_private[grayImage[i]]), 1);         
    i += stride;   
  }
  __syncthreads();  
  if (threadIdx.x < 256)      
    atomicAdd( &(histogram[threadIdx.x]), histo_private[threadIdx.x] );
  
}
__device__ float clamp(float  x, float start, float end){
  float max=(x>start)?x:start;
  return (float)((max<end)?max:end);
}
__device__ float correct_color(unsigned int val, float *cdf,float cdfmin){
   return clamp(255.0f*(cdf[val] - cdfmin)/(1 - cdfmin), 0.0f,255.0f);
}
  
__global__ void equalization(float* outputImage,unsigned char *ucharImage,float * cdf,
                             unsigned int len, float cdfmin){
   unsigned int ii = blockDim.x*blockIdx.x+threadIdx.x;  
   if(ii<len){
     outputImage[ii] =correct_color(ucharImage[ii],cdf,cdf[0])/255.0;
   }
}
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
  float *deviceInput;
  float *deviceOutput;
  unsigned char *ucharImage;
  unsigned char *grayImage;
  unsigned int * histogram;
  float * cdf;
  unsigned int cdfmin=0;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  hostInputImageData=wbImage_getData(inputImage);
  
  int len = imageWidth*imageHeight*imageChannels;
  
  cudaMalloc((void **)&deviceInput, len * sizeof(float));
  cudaMalloc((void **)&deviceOutput, len * sizeof(float));
  cudaMalloc((void **)&ucharImage, len * sizeof(unsigned char));
  cudaMalloc((void **)&grayImage, imageWidth*imageHeight * sizeof(unsigned char));
  
  cudaMalloc((void **)&histogram, 256 * sizeof(unsigned int));
  cudaMalloc((void **)&cdf, 256 * sizeof(float));
  
  
  hostOutputImageData = (float *)malloc(len*sizeof(float));
  cudaMemset(histogram, 0, 256 * sizeof(unsigned int));
    
  cudaMemcpy(deviceInput, hostInputImageData,len*sizeof(float), cudaMemcpyHostToDevice);
  

  
  dim3 grid_dim((len+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);
  dim3 block_dim(BLOCK_SIZE,1,1);
  cast_image<<<grid_dim,block_dim>>>(deviceInput,ucharImage,len);
  cudaDeviceSynchronize();
   
  
  dim3 grid_dim_rgb((imageWidth+BLOCK_SIZE_2D-1)/BLOCK_SIZE_2D,(imageHeight+BLOCK_SIZE_2D-1)/BLOCK_SIZE_2D,1);
  dim3 block_dim_rgb(BLOCK_SIZE_2D,BLOCK_SIZE_2D,1);
  rgb_to_gray<<<grid_dim_rgb,block_dim_rgb>>>(ucharImage, grayImage,imageHeight,imageWidth);
  
  cudaDeviceSynchronize();
  
  dim3 grid_dim_gray((imageWidth*imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);
  dim3 block_dim_gray(BLOCK_SIZE,1,1);
  histogram_privatized_kernel<<<grid_dim_gray,block_dim_gray>>>(histogram,grayImage, imageHeight*imageWidth);
  cudaDeviceSynchronize();
  
  
  dim3 grid_dim_his(1,1,1);
  dim3 block_dim_his(HISTOGRAM_LENGTH,1,1);
  scan<<<grid_dim_his,block_dim_his>>>(histogram, cdf, HISTOGRAM_LENGTH, imageHeight,imageWidth);
  cudaDeviceSynchronize();
  

  equalization<<<grid_dim,block_dim>>>(deviceOutput,ucharImage,cdf,len,cdfmin);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutput,len*sizeof(float), cudaMemcpyDeviceToHost);
  
  wbImage_setData(outputImage,hostOutputImageData);
  wbSolution(args, outputImage);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histogram);
  cudaFree(cdf);
  free(hostOutputImageData);
  free(hostInputImageData);

  return 0;
}

