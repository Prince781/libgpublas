#ifndef gpublas_H
#define gpublas_H

#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>

#if USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#elif USE_OPENCL
#include <CL/cl.h>
#else
#error "Only OpenCL and CUDA are supported."
#endif

#include "lib/obj_tracker.h"

#include "common.h"

/**
 * Any expressions that ultimately make a CUDA kernel call should be wrapped with this.
 * 
 * This disables the Object Tracker for the current thread and calls cudaDeviceSynchronize(),
 * to avoid bus errors either caused by either trying to access unified memory during a kernel 
 * call (alloc_managed() writes the buffer size to the buffer) or after the kernel has been called
 * but the results have not been synchronized.
 */
#if USE_CUDA
#define call_cuda_kernel(expr) {\
    obj_tracker_internal_enter();\
    expr;\
    obj_tracker_internal_leave();\
    if (b2c_must_synchronize)\
        cudaDeviceSynchronize();\
} while (0)
#elif USE_OPENCL
#define call_opencl_kernel(expr) {\
    obj_tracker_internal_enter();\
    expr;\
    obj_tracker_internal_leave();\
    if (b2c_must_synchronize)\
        /* TODO */;\
} while (0)
#endif

struct b2c_options {
    bool debug_execfail;
    bool debug_exec;
    bool trace_copy;
};

extern struct b2c_options b2c_options;
extern bool b2c_must_synchronize;

#if USE_CUDA
extern cublasHandle_t b2c_handle;
typedef cudaError_t b2c_runtime_error_t;
#elif USE_OPENCL
extern cl_context b2c_context;
extern cl_command_queue cl_queue;
typedef cl_int b2c_runtime_error_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * If status is not "cudaSuccess", prints the error message and exits.
 */
#if USE_CUDA
static inline void b2c_fatal_error(cudaError_t status, const char *domain)
{
    if (status != cudaSuccess) {
        writef(STDERR_FILENO, "%s: %s : %s\n", domain, cudaGetErrorName(status), cudaGetErrorString(status));
        abort();
    }
}

#define B2C_ERRORCHECK(name, status) \
do {\
    if (b2c_options.debug_execfail) {\
        switch (status) {\
            case CUBLAS_STATUS_EXECUTION_FAILED:\
                writef(STDERR_FILENO, "gpublas: failed to execute " #name "\n");\
                break;\
            case CUBLAS_STATUS_NOT_INITIALIZED:\
                writef(STDERR_FILENO, "gpublas: not initialized\n");\
                break;\
            case CUBLAS_STATUS_ARCH_MISMATCH:\
                writef(STDERR_FILENO, "gpublas:" #name " not supported\n");\
                break;\
            default:\
                break;\
        }\
    }\
    if (status == CUBLAS_STATUS_SUCCESS && b2c_options.debug_exec)\
        writef(STDERR_FILENO, "gpublas: calling %s()\n", #name);\
} while (0)

#elif USE_OPENCL
const char *clGetErrorString(cl_int status);

static inline void b2c_fatal_error(cl_int status, const char *domain)
{
    if (status != CL_SUCCESS) {
        writef(STDERR_FILENO, "%s: %s\n", domain, clGetErrorString(status));
        abort();
    }
}

#define B2C_ERRORCHECK(name, status) \
do {\
    if (b2c_options.debug_execfail) {\
        if (status != CL_SUCCESS) {\
            writef(STDERR_FILENO, "gpublas: failed to execute " #name ": %s\n", clGetErrorString(status));\
        }\
    }\
    if (status == CL_SUCCESS && b2c_options.debug_exec)\
        writef(STDERR_FILENO, "gpublas: calling %s()\n", #name);\
} while (0)

#endif


/**
 * Creates a new GPU buffer and copies the CPU buffer to it.
 * Returns the GPU buffer, which may be NULL on failure.
 * Free with cudaFree().
 */
void *b2c_copy_to_gpu(const void *hostbuf, size_t size, b2c_runtime_error_t *err);

/**
 * Creates a new CPU buffer and copies the GPU buffer to it.
 * Returns the CPU buffer, which may be NULL on failure.
 * Free with free().
 */
void *b2c_copy_to_cpu(const void *gpubuf, size_t size, b2c_runtime_error_t *err);

/**
 * Copies an existing GPU buffer to an existing CPU buffer.
 */
void b2c_copy_from_gpu(void *hostbuf, const void *gpubuf, size_t size, b2c_runtime_error_t *err);

/**
 * If the buffer is on the CPU, copies it to the GPU and returns a GPU pointer.
 * {*info_in} is set to NULL. If the buffer is NULL, returns a new GPU buffer
 * and {*info_in} is set to NULL.
 * If the buffer is shared, returns the same pointer. {*info_in} points to the
 * object tracking information for this buffer.
 * If there is a fatal error, this function will abort.
 * The remaining arguments are each a pair (gpubuf, gpubuf_info) to perform cleanup on
 * in the event of an error. The list is terminated by a NULL pointer for a
 * gpubuf.
 */
void *b2c_place_on_gpu(void *hostbuf, 
        size_t size,
        const struct objinfo **info_in,
        b2c_runtime_error_t *err,
        void *gpubuf2,
        ...);

/**
 * If {info} is NULL, the GPU buffer will be freed with cudaFree(). Otherwise
 * nothing will happen.
 * If there is a fatal error, this function will abort.
 */
void b2c_cleanup_gpu_ptr(void *gpubuf, const struct objinfo *info, b2c_runtime_error_t *err);

#ifdef __cplusplus
};
#endif

#endif
