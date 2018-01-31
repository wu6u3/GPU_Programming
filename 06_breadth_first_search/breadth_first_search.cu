#include <stdio.h>
#include <wb.h>

#define wbCheck(stmt)                                                                         \
  do {                                                                                        \
    cudaError_t err = stmt;                                                                   \
    if (err != cudaSuccess) {                                                                 \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                                             \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));                          \
      return -1;                                                                              \
    }                                                                                         \
  } while (0)

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/


__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs, // edges
                                          unsigned int *nodeNeighbors, // dest
                                          unsigned int *nodeVisited, // visited
                                          unsigned int *currLevelNodes, // p_frontier
                                          unsigned int *nextLevelNodes, // 
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < *numCurrLevelNodes) {
    const unsigned int node = currLevelNodes[tid];
    for(unsigned int i = nodePtrs[node]; i < nodePtrs[node + 1]; ++i) {
      const unsigned int was_visited = atomicExch(&(nodeVisited[nodeNeighbors[i]]), 1);
      if(!was_visited) {
          const unsigned int my_global_tail = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[my_global_tail] = nodeNeighbors[i];
      }  
    }
  }
}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs, //edges
                                         unsigned int *nodeNeighbors, //dest
                                         unsigned int *nodeVisited, //visited
                                         unsigned int *currLevelNodes, //p_frontier
                                         unsigned int *nextLevelNodes, //c_frontier
                                         unsigned int *numCurrLevelNodes, //p_frontier_tail
                                         unsigned int *numNextLevelNodes) { //c_frontier_tail
  __shared__ unsigned int nextLevelNodes_s[BQ_CAPACITY];
  __shared__ unsigned int numNextLevelNodes_s, our_numNextLevelNodes_s;
  if(threadIdx.x == 0) numNextLevelNodes_s = 0;
  __syncthreads();
  
  const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < *numCurrLevelNodes) {
    const unsigned int node = currLevelNodes[tid];
    for(unsigned int i = nodePtrs[node]; i < nodePtrs[node + 1]; ++i) {
      const unsigned int was_visited = atomicExch(&(nodeVisited[nodeNeighbors[i]]), 1);
      if(!was_visited) {
        
        const unsigned int my_tail = atomicAdd(&numNextLevelNodes_s, 1);
        if(my_tail < BQ_CAPACITY) {
          nextLevelNodes_s[my_tail] = nodeNeighbors[i];
        } else {
          numNextLevelNodes_s = BQ_CAPACITY;
          const unsigned int my_global_tail = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[my_global_tail] = nodeNeighbors[i];
        }
      }  
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    our_numNextLevelNodes_s = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);  
  }
  __syncthreads();
  for(unsigned int i = threadIdx.x; i < numNextLevelNodes_s; i += blockDim.x) {
    nextLevelNodes[our_numNextLevelNodes_s + i] = nextLevelNodes_s[i];
  }

}

__device__ void gpu_warp_sub_queuing_kernel(unsigned int *nodePtrs, //edges
                                         unsigned int *nodeNeighbors, //dest
                                         unsigned int *nodeVisited, //visited
                                         unsigned int *currLevelNodes, //p_frontier
                                         unsigned int *nextLevelNodes, //c_frontier                                        
                                         unsigned int *numNextLevelNodes){
  /*__shared__*/ unsigned int nextLevelNodes_s[WQ_CAPACITY];
  __shared__ unsigned int numNextLevelNodes_s, our_numNextLevelNodes_s;
  if(threadIdx.x == 0) numNextLevelNodes_s = 0;
  __syncthreads();
  
  const unsigned int tid = threadIdx.x;
  if(tid < WARP_SIZE) {
    const unsigned int node = currLevelNodes[tid];
    for(unsigned int i = nodePtrs[node]; i < nodePtrs[node + 1]; ++i) {
      const unsigned int was_visited = atomicExch(&(nodeVisited[nodeNeighbors[i]]), 1);
      if(!was_visited) {
        
        const unsigned int my_tail = atomicAdd(&numNextLevelNodes_s, 1);
        if(my_tail < WQ_CAPACITY) {
          nextLevelNodes_s[my_tail] = nodeNeighbors[i];
        } else {
          numNextLevelNodes_s = WQ_CAPACITY;
          const unsigned int my_global_tail = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[my_global_tail] = nodeNeighbors[i];
        }
      }  
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    our_numNextLevelNodes_s = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);  
  }
  __syncthreads();
  for(unsigned int i = threadIdx.x; i < numNextLevelNodes_s; i += WARP_SIZE) {
    nextLevelNodes[our_numNextLevelNodes_s + i] = nextLevelNodes_s[i];
  }

  
}
__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  __shared__ unsigned int nextLevelNodes_s[BQ_CAPACITY];
  __shared__ unsigned int numNextLevelNodes_s, our_numNextLevelNodes_s;
  
  if(threadIdx.x == 0) numNextLevelNodes_s = 0;
  __syncthreads();
  
  const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < *numCurrLevelNodes) {
    const unsigned int node = currLevelNodes[tid];
    for(unsigned int i = nodePtrs[node]; i < nodePtrs[node + 1]; ++i) {
      const unsigned int was_visited = atomicExch(&(nodeVisited[nodeNeighbors[i]]), 1);
      if(!was_visited) {
        
        const unsigned int my_tail = atomicAdd(&numNextLevelNodes_s, 1);
        if(my_tail < BQ_CAPACITY) {
          nextLevelNodes_s[my_tail] = nodeNeighbors[i];
        } else {
          numNextLevelNodes_s = BQ_CAPACITY;
          const unsigned int my_global_tail = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[my_global_tail] = nodeNeighbors[i];
        }
      }  
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    our_numNextLevelNodes_s = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);  
  }
  __syncthreads();
  for(unsigned int i = threadIdx.x; i < numNextLevelNodes_s; i += blockDim.x) {
    nextLevelNodes[our_numNextLevelNodes_s + i] = nextLevelNodes_s[i];
  }
}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                 unsigned int *nodeVisited, unsigned int *currLevelNodes,
                 unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                 unsigned int *numNextLevelNodes) {
  
  // Loop over all nodes in the curent level
  for (unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if (!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
                ++(*numNextLevelNodes);
              }
            }
          }
        }

        void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                                unsigned int *nodeVisited, unsigned int *currLevelNodes,
                                unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                                unsigned int *numNextLevelNodes) {

          const unsigned int numBlocks = 45;
          gpu_global_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited,
                                                               currLevelNodes, nextLevelNodes,
                                                               numCurrLevelNodes, numNextLevelNodes);
        }

        void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                               unsigned int *nodeVisited, unsigned int *currLevelNodes,
                               unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                               unsigned int *numNextLevelNodes) {

          const unsigned int numBlocks = 45;
          gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited,
                                                              currLevelNodes, nextLevelNodes,
                                                              numCurrLevelNodes, numNextLevelNodes);
        }

        void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                              unsigned int *nodeVisited, unsigned int *currLevelNodes,
                              unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                              unsigned int *numNextLevelNodes) {

          const unsigned int numBlocks = 45;
          gpu_warp_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited,
                                                             currLevelNodes, nextLevelNodes,
                                                             numCurrLevelNodes, numNextLevelNodes);
        }

        void setupProblem(const unsigned int numNodes, unsigned int **nodePtrs_h,
                          unsigned int **nodeNeighbors_h, unsigned int **nodeVisited_h,
                          unsigned int **nodeVisited_ref, unsigned int **currLevelNodes_h,
                          unsigned int **nextLevelNodes_h, unsigned int **numCurrLevelNodes_h,
                          unsigned int **numNextLevelNodes_h, const int *nodePtrs_file,
                          const int *nodeNeighbors_file, const unsigned int data_size,
                          const int *currLevelNodes_file, const unsigned int currentLevelNode_size) {

          // Initialize node pointers
          *nodePtrs_h = (unsigned int *)malloc((numNodes + 1) * sizeof(unsigned int));
          *nodeVisited_h = (unsigned int *)malloc(numNodes * sizeof(unsigned int));
          *nodeVisited_ref = (unsigned int *)malloc(numNodes * sizeof(unsigned int));

          (*nodePtrs_h)[0] = 0;
          for (unsigned int node = 0; node < numNodes; node++) {
            (*nodePtrs_h)[node + 1] = nodePtrs_file[node + 1];
          }

          *nodeNeighbors_h = (unsigned int *)malloc(data_size * sizeof(unsigned int));

          for (unsigned int neighborIdx = 0; neighborIdx < data_size; neighborIdx++) {
            (*nodeNeighbors_h)[neighborIdx] = nodeNeighbors_file[neighborIdx];
          }

          memset((*nodeVisited_h), 0, numNodes * sizeof(int));
          memset((*nodeVisited_ref), 0, numNodes * sizeof(int));

          *numCurrLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
          **numCurrLevelNodes_h = currentLevelNode_size;

          *currLevelNodes_h = (unsigned int *)malloc((**numCurrLevelNodes_h) * sizeof(unsigned int));

          for (unsigned int idx = 0; idx < currentLevelNode_size; idx++) {
            unsigned int node = currLevelNodes_file[idx];
            (*currLevelNodes_h)[idx] = node;
            (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 1;
          }

          // Prepare next level containers (i.e. output variables)
          *numNextLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
          **numNextLevelNodes_h = 0;
          *nextLevelNodes_h = (unsigned int *)malloc((numNodes) * sizeof(unsigned int));
        }

        int main(int argc, char *argv[]) {

          wbArg_t args;

          args = wbArg_read(argc, argv);
          std::cerr << "args read\n";
          // Initialize host variables
          // ----------------------------------------------

          // Variables
          unsigned int numNodes;
          unsigned int *nodePtrs_h;
          unsigned int *nodeNeighbors_h;
          unsigned int *nodeVisited_h;
          unsigned int *nodeVisited_ref;
          unsigned int *currLevelNodes_h;
          unsigned int *nextLevelNodes_h;
          unsigned int *numCurrLevelNodes_h;
          unsigned int *numNextLevelNodes_h;
          unsigned int *nodePtrs_d;
          unsigned int *nodeNeighbors_d;
          unsigned int *nodeVisited_d;
          unsigned int *currLevelNodes_d;
          unsigned int *nextLevelNodes_d;
          unsigned int *numCurrLevelNodes_d;
          unsigned int *numNextLevelNodes_d;

          ///////////////////////////////////////////////////////

          enum Mode { CPU = 1, GPU_GLOBAL_QUEUE, GPU_BLOCK_QUEUE, GPU_WARP_QUEUE };
          Mode mode;

          int inputLength1; // Total number of Nodes + 1
          int inputLength2; // Total number of Neighbor nodes
          int inputLength3; // Total number of nodes in the current level
          int *hostInput1;  // Node Pointers
          int *hostInput2;  // Node Neighbors
          int *hostInput3;  // Current level Nodes

          wbTime_start(Generic, "Importing data and creating memory on host");

          mode = (Mode)wbImport_flag(wbArg_getInputFile(args, 0));

          hostInput1 = (int *)wbImport(wbArg_getInputFile(args, 1), &inputLength1, "Integer");

          // Total  number of nodes in the graph
          numNodes = inputLength1 - 1;

          hostInput2 = (int *)wbImport(wbArg_getInputFile(args, 2), &inputLength2, "Integer");

          hostInput3 = (int *)wbImport(wbArg_getInputFile(args, 3), &inputLength3, "Integer");

          wbTime_stop(Generic, "Importing data and creating memory on host");

          wbTime_start(Generic, "Setting up the problem...");

          // Initialize graph from imput files and setting up pointers
          setupProblem(numNodes, &nodePtrs_h, &nodeNeighbors_h, &nodeVisited_h, &nodeVisited_ref,
                       &currLevelNodes_h, &nextLevelNodes_h, &numCurrLevelNodes_h,
                       &numNextLevelNodes_h, hostInput1, hostInput2, inputLength2, hostInput3,
                       inputLength3);

          wbTime_stop(Generic, "Setting up the problem...");

          // Allocate device variables
          // ----------------------------------------------
          
          
          //mode = CPU;
          
          if (mode != CPU) {

            wbTime_start(GPU, "Allocating GPU memory.");

            wbCheck(cudaMalloc((void **)&nodePtrs_d, (numNodes + 1) * sizeof(unsigned int)));

            wbCheck(cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(unsigned int)));

            wbCheck(cudaMalloc((void **)&nodeNeighbors_d, inputLength2 * sizeof(unsigned int)));

            wbCheck(cudaMalloc((void **)&numCurrLevelNodes_d, sizeof(unsigned int)));

            wbCheck(
                cudaMalloc((void **)&currLevelNodes_d, (*numCurrLevelNodes_h) * sizeof(unsigned int)));

            wbCheck(cudaMalloc((void **)&numNextLevelNodes_d, sizeof(unsigned int)));

            wbCheck(cudaMalloc((void **)&nextLevelNodes_d, (numNodes) * sizeof(unsigned int)));

            wbTime_stop(GPU, "Allocating GPU memory.");
          }

          // Copy host variables to device
          // ------------------------------------------

          if (mode != CPU) {
            wbTime_start(GPU, "Copying input memory to the GPU.");

            wbCheck(cudaMemcpy(nodePtrs_d, nodePtrs_h, (numNodes + 1) * sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

            wbCheck(cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

            wbCheck(cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h, inputLength2 * sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

            wbCheck(cudaMemcpy(numCurrLevelNodes_d, numCurrLevelNodes_h, sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

            wbCheck(cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
                               (*numCurrLevelNodes_h) * sizeof(unsigned int), cudaMemcpyHostToDevice));

            wbCheck(cudaMemset(nextLevelNodes_d, 0, (numNodes) * sizeof(unsigned int)));

            wbCheck(cudaMemset(numNextLevelNodes_d, 0, sizeof(unsigned int)));

            wbTime_stop(GPU, "Copying input memory to the GPU.");
          }

          // Launch kernel
          // ----------------------------------------------------------

          printf("Launching kernel ");

  if (mode == CPU) {
    wbTime_start(Compute, "Performing CPU queuing computation");
    cpu_queuing(nodePtrs_h, nodeNeighbors_h, nodeVisited_h, currLevelNodes_h, nextLevelNodes_h,
                numCurrLevelNodes_h, numNextLevelNodes_h);
    wbTime_stop(Compute, "Performing CPU queuing computation");
  } else if (mode == GPU_GLOBAL_QUEUE) {
    wbTime_start(Compute, "Performing GPU global queuing computation");
    gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d, currLevelNodes_d,
                       nextLevelNodes_d, numCurrLevelNodes_d, numNextLevelNodes_d);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing GPU global queuing computation");
  } else if (mode == GPU_BLOCK_QUEUE) {
    wbTime_start(Compute, "Performing GPU block global queuing computation");
    gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d, currLevelNodes_d,
                      nextLevelNodes_d, numCurrLevelNodes_d, numNextLevelNodes_d);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing GPU block global queuing computation");
  } else if (mode == GPU_WARP_QUEUE) {
    wbTime_start(Compute, "Performing GPU warp global queuing computation");
    gpu_warp_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d, currLevelNodes_d,
                     nextLevelNodes_d, numCurrLevelNodes_d, numNextLevelNodes_d);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing GPU warp global queuing computation");
  } else {
    // printf("Invalid mode!\n");
    // exit(0);
  }

  // Copy device variables from host
  // ----------------------------------------

  if (mode != CPU) {

    wbTime_start(Copy, "Copying output memory to the CPU");

    wbCheck(cudaMemcpy(numNextLevelNodes_h, numNextLevelNodes_d, sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));

    wbCheck(cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d, numNodes * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");
  }

  // Verify correctness
  // -----------------------------------------------------

  // sorting output to compare the frontier
  if (mode == CPU) {
    printf("\nNum of node = %d\nnon-sort\n", *numNextLevelNodes_h);
    for(unsigned int i = 0; i < *numNextLevelNodes_h; ++i){
      printf("%d\n",nextLevelNodes_h[i]);
    }
  }
  wbSort(nextLevelNodes_h, numNodes);

  wbTime_start(Generic, "Verifying results...");

  wbSolution(args, (int *)nextLevelNodes_h, *numNextLevelNodes_h);
  if (mode == CPU) {
    printf("\n num of node = %d\n", *numNextLevelNodes_h);
  
    for(unsigned int i = 0; i < *numNextLevelNodes_h; ++i){
      printf("%d\n",nextLevelNodes_h[i]);
    }
  }
  
  wbTime_stop(Generic, "Verifying results...");

  // Free memory
  // ------------------------------------------------------------

  free(nodePtrs_h);
  free(nodeVisited_h);
  free(nodeVisited_ref);
  free(nodeNeighbors_h);
  free(numCurrLevelNodes_h);
  free(currLevelNodes_h);
  free(numNextLevelNodes_h);
  free(nextLevelNodes_h);
  if (mode != CPU) {
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(nodePtrs_d);
    cudaFree(nodeVisited_d);
    cudaFree(nodeNeighbors_d);
    cudaFree(numCurrLevelNodes_d);
    cudaFree(currLevelNodes_d);
    cudaFree(numNextLevelNodes_d);
    cudaFree(nextLevelNodes_d);
    wbTime_stop(GPU, "Freeing GPU Memory");
  }

  return 0;
}

