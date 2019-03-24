#include "runtime.h"
#include <stdbool.h>

#if USE_OPENCL
cl_context opencl_ctx;
cl_command_queue opencl_cmd_queue;
bool opencl_finegrained;



#endif

runtime_error_t runtime_init(void) {
#if USE_CUDA
    return cudaSuccess;
#else
#error "TODO"
#endif
}

runtime_error_t runtime_fini(void) {
#if USE_CUDA
    return cudaSuccess;
#else
#error "TODO"
#endif
}

runtime_error_t runtime_memcpy_htod(void *gpubuf, const void *hostbuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMemcpy(gpubuf, hostbuf, size, cudaMemcpyHostToDevice);
#else
#error "TODO"
#endif
    return err;
}

runtime_error_t runtime_memcpy_dtoh(void *hostbuf, const void *gpubuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMemcpy(hostbuf, gpubuf, size, cudaMemcpyDeviceToHost);
#else
#error "TODO"
#endif
    return err;
}

runtime_error_t runtime_malloc(void **gpubuf_in, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMalloc(gpubuf_in, size);
#else
    err = runtime_malloc_shared(gpubuf_in, size);
#endif
    return err;
}

runtime_error_t runtime_malloc_shared(void **sharedbuf_in, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMallocManaged(sharedbuf_in, size, cudaMemAttachGlobal);
#else
    *sharedbuf_in = clSVMAlloc(opencl_ctx, CL_MEM_READ_WRITE, size, 0);
    err = RUNTIME_ERROR_SUCCESS;
#endif
    return err;
}

runtime_error_t runtime_free(void *gpubuf) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaFree(gpubuf);
#else
    clSVMFree(opencl_ctx, gpubuf);
    err = RUNTIME_ERROR_SUCCESS;
#endif
    return err;
}

runtime_error_t runtime_svm_map(void *sharedbuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    // sharedbuf was allocated with cudaMallocManaged, so we do nothing
    err = RUNTIME_ERROR_SUCCESS;
#else
    err = opencl_finegrained ? RUNTIME_ERROR_SUCCESS :
        clEnqueueSVMMap(opencl_cmd_queue, 
            CL_TRUE /* block until this completes */, 
            CL_MEM_READ_WRITE, sharedbuf, size, 0, NULL, NULL);
#endif
    return err;
}

runtime_error_t runtime_svm_unmap(void *sharedbuf) {
    runtime_error_t err;
#if USE_CUDA
    // sharedbuf was allocated with cudaMallocManaged, so we do nothing
    err = RUNTIME_ERROR_SUCCESS;
#else
    err = opencl_finegrained ? RUNTIME_ERROR_SUCCESS :
        clEnqueueSVMUnmap(opencl_cmd_queue,
            sharedbuf,
            0, NULL, NULL);
#endif
    return err;
}
