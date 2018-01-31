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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
/// For simplicity, fix #bins=1024 so scan can use a single block and no
/// padding
#define NUM_BINS 1024

/******************************************************************************
 GPU main computation kernels
*******************************************************************************/

__global__ void gpu_normal_kernel(float *in_val, float *in_pos, float *out,
                                  int grid_size, int num_in) {

  int outIdx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(outIdx < grid_size) {
    float out_value = 0.0f;
    for (int inIdx = 0; inIdx < num_in; ++inIdx) {
      const float in_val2 = in_val[inIdx] * in_val[inIdx];    
      const float dist = in_pos[inIdx] - (float)outIdx;
      out_value += in_val2 / (dist * dist);
    }
    out[outIdx] = out_value ; 
  }
}

__global__ void gpu_cutoff_kernel(float *in_val, float *in_pos, float *out,
                                  int grid_size, int num_in,
                                  float cutoff2) {

  int outIdx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(outIdx < grid_size) {
    float out_value = 0.0f;
    for (int inIdx = 0; inIdx < num_in; ++inIdx) {
      const float in_val2 = in_val[inIdx] * in_val[inIdx];    
      const float dist = in_pos[inIdx] - (float)outIdx;
      if(dist <= cutoff2)
        out_value += in_val2 / (dist * dist);
      
    }
    out[outIdx] = out_value; 
    
  }
}

__global__ void gpu_cutoff_binned_kernel(int *bin_ptrs,
                                         float *in_val_sorted,
                                         float *in_pos_sorted, float *out,
                                         int grid_size, int num_in, float cutoff2 ) {
  

  int outIdx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(outIdx < grid_size) {
    float out_value = 0.0f;
    int binIdx = (int)((outIdx / grid_size) * NUM_BINS);
    const int offset = bin_ptrs[binIdx];
    
    for (int i = 0; i < num_in; ++i) {      
      
      const int inIdx = i+offset;
      const float in_val2 = in_val_sorted[inIdx] * in_val_sorted[inIdx];    
      const float dist = in_pos_sorted[inIdx] - (float)outIdx;     
      
      if(dist <= cutoff2)
        out_value += in_val2 / (dist * dist);
      
    }
    out[outIdx] = out_value; 
  }
  
}

/******************************************************************************
 Main computation functions
*******************************************************************************/

void cpu_normal(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in) {
 
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const float in_val2 = in_val[inIdx] * in_val[inIdx];
      for (int outIdx = 0; outIdx < grid_size; ++outIdx) {
      const float dist = in_pos[inIdx] - (float)outIdx;
      out[outIdx] += in_val2 / (dist * dist);
      
    }
  }
}

void gpu_normal(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_normal_kernel<<<numBlocks, numThreadsPerBlock>>>(in_val, in_pos, out,
                                                       grid_size, num_in);
}

void gpu_cutoff(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in, float cutoff2) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_cutoff_kernel<<<numBlocks, numThreadsPerBlock>>>(
      in_val, in_pos, out, grid_size, num_in, cutoff2);
}

void gpu_cutoff_binned(int *bin_ptrs, float *in_val_sorted,
                       float *in_pos_sorted, float *out, int grid_size, int num_in,
                       float cutoff2) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  
  gpu_cutoff_binned_kernel<<<numBlocks, numThreadsPerBlock>>>(
      bin_ptrs, in_val_sorted, in_pos_sorted, out, grid_size, num_in, cutoff2);
}

/******************************************************************************
 Preprocessing kernels
*******************************************************************************/

__global__ void histogram(float *in_pos, int *bin_counts, int num_in,
                          int grid_size) {

  int inIdx = blockIdx.x*blockDim.x+threadIdx.x;   
  if(inIdx < num_in){
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS); 
    atomicAdd(&bin_counts[binIdx],1);
  }
  __syncthreads();
}

__global__ void scan(int *bin_counts, int *bin_ptrs) {

  __shared__ float T[NUM_BINS];
  unsigned int t = threadIdx.x;    
  unsigned int start = blockDim.x*blockIdx.x;
  
  if (start+t<NUM_BINS){
    T[t]=bin_counts[start+t];
  }else {
    T[t]=0.0f;
  }
  __syncthreads();
  
  unsigned int stride=1;  
  while(stride < NUM_BINS){
    int index = (t+1)*stride*2 - 1;
    if(index < NUM_BINS){      
      T[index] += T[index-stride];       
    }
    stride = stride*2;
    __syncthreads();
  }
  stride = NUM_BINS/2;
  while(stride > 0) {
    int index = (t+1)*stride*2 - 1;
    if(index < NUM_BINS&&index+stride<NUM_BINS) {       
      T[index+stride] += T[index];        
    }        
    stride = stride/2;        
    __syncthreads();    
  }
  
  if (start+t<NUM_BINS&&t>0){
    bin_ptrs[start+t]=T[t-1];
  }else if(start+t<NUM_BINS&&start+t==0){
    bin_ptrs[start+t]=0.0f;
  }
  
}

__global__ void sort(float *in_val, float *in_pos, float *in_val_sorted,
                     float *in_pos_sorted, int grid_size, int num_in,
                     int *bin_counts, int *bin_ptrs) {

  
  int tx = blockIdx.x*blockDim.x+threadIdx.x;  
  if(tx == 0){
    for (int inIdx = 0; inIdx < num_in; ++inIdx) {
      const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
      const int newIdx = bin_ptrs[binIdx + 1] - bin_counts[binIdx];
      --bin_counts[binIdx];
      in_val_sorted[newIdx] = in_val[inIdx];
      in_pos_sorted[newIdx] = in_pos[inIdx];
    
    }
  }
  
}

/******************************************************************************
 Preprocessing functions
*******************************************************************************/

static void cpu_preprocess(float *in_val, float *in_pos,
                           float *in_val_sorted, float *in_pos_sorted,
                           int grid_size, int num_in, int *bin_counts,
                           int *bin_ptrs) {

  // Histogram the input positions
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
    ++bin_counts[binIdx];
    
  }

  // Scan the histogram to get the bin pointers
  bin_ptrs[0] = 0;
  for (int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
    bin_ptrs[binIdx + 1] = bin_ptrs[binIdx] + bin_counts[binIdx];
    
  }
  // Sort the inputs into the bins
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
    const int newIdx = bin_ptrs[binIdx + 1] - bin_counts[binIdx];
    --bin_counts[binIdx];
    in_val_sorted[newIdx] = in_val[inIdx];
    in_pos_sorted[newIdx] = in_pos[inIdx];
    
  }
}

static void gpu_preprocess(float *in_val, float *in_pos,
                           float *in_val_sorted, float *in_pos_sorted,
                           int grid_size, int num_in, int *bin_counts,
                           int *bin_ptrs) {

  const int numThreadsPerBlock = 512;

  // Histogram the input positions
  histogram<<<30, numThreadsPerBlock>>>(in_pos, bin_counts, num_in,
                                        grid_size);
  
  // Scan the histogram to get the bin pointers
  if (NUM_BINS != 1024) {
    wbLog(FATAL, "NUM_BINS must be 1024. Do not change.");
    return;
  }
  scan<<<1, numThreadsPerBlock*2>>>(bin_counts, bin_ptrs);

  // Sort the inputs into the bins
  sort<<<30, numThreadsPerBlock>>>(in_val, in_pos, in_val_sorted,
                                   in_pos_sorted, grid_size, num_in,
                                   bin_counts, bin_ptrs);
}

enum Mode {
  CPU_NORMAL = 1,
  GPU_NORMAL,
  GPU_CUTOFF,
  GPU_BINNED_CPU_PREPROCESSING,
  GPU_BINNED_GPU_PREPROCESSING
};

int main(int argc, char **argv) {
  wbArg_t args;

  // Initialize host variables
  // ----------------------------------------------

  // Variables
  float *in_val_h;
  float *in_pos_h;
  float *out_h;
  float *in_val_d;
  float *in_pos_d;
  float *out_d;
  int grid_size;
  int num_in;
  Mode mode;

  // Constants
  const float cutoff = 3000.0f; // Cutoff distance for optimized computation
  const float cutoff2 = cutoff * cutoff;

  // Extras needed for input binning
  int *bin_counts_h;
  int *bin_ptrs_h;
  float *in_val_sorted_h;
  float *in_pos_sorted_h;
  int *bin_counts_d;
  int *bin_ptrs_d;
  float *in_val_sorted_d;
  float *in_pos_sorted_d;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  mode = (Mode)wbImport_flag(wbArg_getInputFile(args, 0));
  in_val_h = (float *)wbImport(wbArg_getInputFile(args, 1), &num_in);
  in_pos_h = (float *)wbImport(wbArg_getInputFile(args, 2), &num_in);
  grid_size = (int)wbImport_flag(wbArg_getInputFile(args, 3));

  out_h = (float *)calloc(grid_size, sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of inputs is ", num_in);
  wbLog(TRACE, "The grid_size is ", grid_size);
  
  // CPU Preprocessing
  // ------------------------------------------------------
  //mode = GPU_CUTOFF;
  if (mode == GPU_BINNED_CPU_PREPROCESSING) {

    wbTime_start(Generic, "Allocating data for preprocessing");
    // Data structures needed to preprocess the bins on the CPU
    bin_counts_h = (int *)malloc(NUM_BINS * sizeof(int));
    bin_ptrs_h = (int *)malloc((NUM_BINS + 1) * sizeof(int));
    in_val_sorted_h = (float *)malloc(num_in * sizeof(float));
    in_pos_sorted_h = (float *)malloc(num_in * sizeof(float));

    cpu_preprocess(in_val_h, in_pos_h, in_val_sorted_h, in_pos_sorted_h,
                   grid_size, num_in, bin_counts_h, bin_ptrs_h);
    wbTime_stop(Generic, "Allocating data for preprocessing");
    
  }
  // Allocate device variables
  // ----------------------------------------------

  if (mode != CPU_NORMAL) {

    wbTime_start(GPU, "Allocating data");
    // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
    if (mode != GPU_BINNED_CPU_PREPROCESSING) {
      wbCheck(cudaMalloc((void **)&in_val_d, num_in * sizeof(float)));
      wbCheck(cudaMalloc((void **)&in_pos_d, num_in * sizeof(float)));
    }

    // All modes need the output array
    wbCheck(cudaMalloc((void **)&out_d, grid_size * sizeof(float)));

    // Only binning modes need binning information
    if (mode == GPU_BINNED_CPU_PREPROCESSING ||
        mode == GPU_BINNED_GPU_PREPROCESSING) {

      wbCheck(cudaMalloc((void **)&in_val_sorted_d, num_in * sizeof(float)));

      wbCheck(cudaMalloc((void **)&in_pos_sorted_d, num_in * sizeof(float)));

      wbCheck(cudaMalloc((void **)&bin_ptrs_d, (NUM_BINS + 1) * sizeof(int)));

      if (mode == GPU_BINNED_GPU_PREPROCESSING) {
        // Only used in preprocessing but not the actual computation
        wbCheck(cudaMalloc((void **)&bin_counts_d, NUM_BINS * sizeof(int)));
      }
    }

    cudaDeviceSynchronize();
    wbTime_stop(GPU, "Allocating data");
  }

  // Copy host variables to device
  // ------------------------------------------

  if (mode != CPU_NORMAL) {
    wbTime_start(Copy, "Copying data");
    // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
    if (mode != GPU_BINNED_CPU_PREPROCESSING) {
      wbCheck(cudaMemcpy(in_val_d, in_val_h, num_in * sizeof(float), cudaMemcpyHostToDevice));

      wbCheck(cudaMemcpy(in_pos_d, in_pos_h, num_in * sizeof(float), cudaMemcpyHostToDevice));
    }

    // All modes need the output array
    wbCheck(cudaMemset(out_d, 0, grid_size * sizeof(float)));

    if (mode == GPU_BINNED_CPU_PREPROCESSING) {

      wbCheck(cudaMemcpy(in_val_sorted_d, in_val_sorted_h, num_in * sizeof(float), cudaMemcpyHostToDevice));

      wbCheck(cudaMemcpy(in_pos_sorted_d, in_pos_sorted_h, num_in * sizeof(float), cudaMemcpyHostToDevice));

      wbCheck(cudaMemcpy(bin_ptrs_d, bin_ptrs_h, (NUM_BINS + 1) * sizeof(int), cudaMemcpyHostToDevice));

    } else if (mode == GPU_BINNED_GPU_PREPROCESSING) {
      // If preprocessing on the GPU, bin counts need to be initialized
      //  and nothing needs to be copied
      wbCheck(cudaMemset(bin_counts_d, 0, NUM_BINS * sizeof(int)));
    }

    cudaDeviceSynchronize();
    wbTime_stop(Copy, "Copying data");
  }

  // GPU Preprocessing
  // ------------------------------------------------------

  if (mode == GPU_BINNED_GPU_PREPROCESSING) {

    wbTime_start(Compute, "Preprocessing data on the GPU...");

    gpu_preprocess(in_val_d, in_pos_d, in_val_sorted_d, in_pos_sorted_d,
                   grid_size, num_in, bin_counts_d, bin_ptrs_d);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Preprocessing data on the GPU...");
  }

  // Launch kernel
  // ----------------------------------------------------------

  wbLog(TRACE, "Launching kernel ");

  if (mode == CPU_NORMAL) {
    wbTime_start(Compute, "Performing CPU_NORMAL computation");
    cpu_normal(in_val_h, in_pos_h, out_h, grid_size, num_in);
    wbTime_stop(Compute, "Performing CPU_NORMAL Scatter computation");
  } else if (mode == GPU_NORMAL) {
    wbTime_start(Compute, "Performing GPU_NORMAL computation");
    gpu_normal(in_val_d, in_pos_d, out_d, grid_size, num_in);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing GPU_NORMAL computation");
  } else if (mode == GPU_CUTOFF) {
    wbTime_start(Compute, "Performing GPU_CUTOFF computation");
    gpu_cutoff(in_val_d, in_pos_d, out_d, grid_size, num_in, cutoff2);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing GPU_CUTOFF computation");
  } else if (mode == GPU_BINNED_CPU_PREPROCESSING ||
             mode == GPU_BINNED_GPU_PREPROCESSING) {
    wbTime_start(Compute, "Performing GPU_BINNED_CPU_PREPROCESSING || "
                          "GPU_BINNED_GPU_PREPROCESSING  computation");
    gpu_cutoff_binned(bin_ptrs_d, in_val_sorted_d, in_pos_sorted_d, out_d,
                      grid_size, num_in, cutoff2);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing GPU_BINNED_CPU_PREPROCESSING || "
                         "GPU_BINNED_GPU_PREPROCESSING computation");
  } else {
    wbLog(FATAL, "Invalid mode ", mode);
  }

  // Copy device variables from host
  // ----------------------------------------

  if (mode != CPU_NORMAL) {
    wbCheck(cudaMemcpy(out_h, out_d, grid_size * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbCheck(cudaDeviceSynchronize());
  }

  // Verify correctness
  // -----------------------------------------------------

  wbSolution(args, out_h, grid_size);

  // Free memory
  // ------------------------------------------------------------

  free(in_val_h);
  free(in_pos_h);
  free(out_h);
  if (mode == GPU_BINNED_CPU_PREPROCESSING) {
    free(bin_counts_h);
    free(bin_ptrs_h);
    free(in_val_sorted_h);
    free(in_pos_sorted_h);
  }
  if (mode != CPU_NORMAL) {
    if (mode != GPU_BINNED_CPU_PREPROCESSING) {
      cudaFree(in_val_d);
      cudaFree(in_pos_d);
    }
    cudaFree(out_d);
    if (mode == GPU_BINNED_CPU_PREPROCESSING ||
        mode == GPU_BINNED_GPU_PREPROCESSING) {
      cudaFree(in_val_sorted_d);
      cudaFree(in_pos_sorted_d);
      cudaFree(bin_ptrs_d);
      if (mode == GPU_BINNED_GPU_PREPROCESSING) {
        cudaFree(bin_counts_d);
      }
    }
  }

  return 0;
}

