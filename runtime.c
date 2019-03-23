#include "runtime.h"

runtime_error_t runtime_memcpy_htod(void *gpubuf, const void *hostbuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMemcpy(gpubuf, hostbuf, size, cudaMemcpyHostToDevice);
#else
#endif
    return err;
}

runtime_error_t runtime_memcpy_dtoh(void *hostbuf, const void *gpubuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMemcpy(hostbuf, gpubuf, size, cudaMemcpyDeviceToHost);
#else
#endif
    return err;
}

runtime_error_t runtime_malloc(void **gpubuf_in, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMalloc(gpubuf_in, size);
#else
#error "TODO"
#endif
    return err;
}

runtime_error_t runtime_malloc_shared(void **sharedbuf_in, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMallocManaged(sharedbuf_in, size, cudaMemAttachGlobal);
#else
#error "TODO"
#endif
    return err;
}

runtime_error_t runtime_free(void *gpubuf) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaFree(gpubuf);
#else
#endif
    return err;
}
