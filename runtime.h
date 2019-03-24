#ifndef RUNTIME_H
#define RUNTIME_H

#if USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

typedef cudaError_t runtime_error_t;

#define RUNTIME_ERROR_SUCCESS cudaSuccess

#define runtime_is_error(x) (x != cudaSuccess)

#define runtime_error_name cudaGetErrorName
#define runtime_error_string cudaGetErrorString

/**
 * Any expressions that ultimately make a CUDA kernel call should be wrapped with this.
 * 
 * This disables the Object Tracker for the current thread and calls cudaDeviceSynchronize(),
 * to avoid bus errors either caused by trying to access unified memory during a kernel call 
 * (alloc_managed() writes the buffer size to the buffer) or after the kernel has been called
 * but the results have not been synchronized.
 */
#define call_kernel(expr) {\
    obj_tracker_internal_enter();\
    expr;\
    obj_tracker_internal_leave();\
    if (b2c_must_synchronize)\
        cudaDeviceSynchronize();\
} while (0)

typedef void *runtime_gpubuf_t;

#elif USE_OPENCL
#include <CL/cl.h>
#include "clext.h"

typedef cl_int runtime_error_t;

#define RUNTIME_ERROR_SUCCESS CL_SUCCESS

#define runtime_is_error(x) (x != CL_SUCCESS)
#define runtime_error_name clGetErrorString
#define runtime_error_string clGetErrorString

#define call_kernel(expr) {\
    obj_tracker_internal_enter();\
    expr;\
    obj_tracker_internal_leave();\
    if (b2c_must_synchronize)\
        /* TODO */;\
} while (0)

/**
 * Memory for exclusive use on the device.
 */
typedef cl_mem runtime_gpubuf_t;

#else
#error "Only CUDA and OpenCL are supported"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the runtime (only used for OpenCL so far).
 */
runtime_error_t runtime_init(void);

/**
 * Deinitialize the runtime
 */
runtime_error_t runtime_fini(void);

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

/**
 * Map a shared buffer into the host address space. If fine-grained access is
 * supported, this does nothing.
 */
runtime_error_t runtime_svm_map(void *sharedbuf, size_t size);

/**
 * Unmap a shared buffer into the host address space. If fine-grained access 
 * is supported, this does nothing.
 */
runtime_error_t runtime_svm_unmap(void *sharedbuf);

#ifdef __cplusplus
};
#endif

#endif
