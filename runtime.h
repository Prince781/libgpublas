#ifndef RUNTIME_H
#define RUNTIME_H

#if USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

typedef cudaError_t runtime_error_t;

struct runtime_device {
    int id;
    char name[128];
};

#elif USE_OPENCL
#include <CL/cl.h>
#include "clext.h"

typedef cl_int runtime_error_t;
#else
#error "Only CUDA and OpenCL are supported"
#endif

/**
 * Initialize the runtime (CUDA/OpenCL).
 */
runtime_error_t runtime_init(void);

/**
 * Copy memory from hostbuf -> gpubuf
 */
runtime_error_t runtime_memcpy_htod(void *gpubuf, const void *hostbuf, size_t size);

/**
 * Copy memory from gpubuf -> hostbuf
 */
runtime_error_t runtime_memcpy_dtoh(void *hostbuf, const void *gpubuf, size_t size);

/**
 * Allocate new memory onto the device and return a pointer to it in gpubuf_in.
 */
runtime_error_t runtime_malloc(void **gpubuf_in, size_t size);

/**
 * Allocate shared memory between the device and host, and return a pointer to it in sharedbuf_in.
 */
runtime_error_t runtime_malloc_shared(void **sharedbuf_in, size_t size);

/**
 * Free memory allocated on the device.
 */
runtime_error_t runtime_free(void *gpubuf);

#endif
