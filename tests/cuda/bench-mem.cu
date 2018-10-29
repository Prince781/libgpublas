// see https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/
#include <stdio.h>
#include <assert.h>

__global__ void do_stuff(float *dev_A, size_t size) {
    /* blockDim.{x,y} is a constant = the dimension of the grid */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        dev_A[i] = 13;
    }
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    if (result != cudaSuccess)
	abort();
  }
#endif
  return result;
}

void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int  n,
                   const char   *desc)
{
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // peform kernel operation on h_a to make sure it's on the GPU
  do_stuff<<<(n + 255)/256 /*blocks*/, 256/*threads*/>>>(d, n);

  // events for timing
  cudaEvent_t startEvent, stopEvent; 

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (unsigned i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

int main()
{
  unsigned int nElements = 4*1024*1024;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;   
  float *h_aPinned, *h_bPinned;
  float *h_aManaged, *h_bManaged;

  // device array
  float *d_a;

  // allocate and initialize
  h_aPageable = (float*)malloc(bytes);                    // host pageable
  h_bPageable = (float*)malloc(bytes);                    // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
  checkCuda( cudaMallocManaged((void **) &h_aManaged, bytes) );
  checkCuda( cudaMallocManaged((void **) &h_bManaged, bytes) );
  checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

  for (unsigned i = 0; i < nElements; ++i) h_aPageable[i] = i;      
  memcpy(h_aPinned, h_aPageable, bytes);
  memcpy(h_aPinned, h_aManaged, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);
  memset(h_bManaged, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
  profileCopies(h_aManaged, h_bManaged, d_a, nElements, "Managed");

  printf("n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}
