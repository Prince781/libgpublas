#ifndef BLAS2CUDA_H
#define BLAS2CUDA_H

#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "lib/obj_tracker.h"

#include "common.h"

#define B2C_ERRORCHECK(name, status) \
do {\
    if (b2c_options.debug_execfail) {\
        switch (status) {\
            case CUBLAS_STATUS_EXECUTION_FAILED:\
                writef(STDERR_FILENO, "blas2cuda: failed to execute " #name "\n");\
                break;\
            case CUBLAS_STATUS_NOT_INITIALIZED:\
                writef(STDERR_FILENO, "blas2cuda: not initialized\n");\
                break;\
            case CUBLAS_STATUS_ARCH_MISMATCH:\
                writef(STDERR_FILENO, "blas2cuda:" #name " not supported\n");\
                break;\
            default:\
                break;\
        }\
    }\
    if (status == CUBLAS_STATUS_SUCCESS && b2c_options.debug_exec)\
        writef(STDERR_FILENO, "blas2cuda: calling %s()\n", #name);\
} while (0)


/**
 * Any expressions that ultimately make a CUDA kernel call should be wrapped with this.
 * 
 * This disables the Object Tracker for the current thread and calls cudaDeviceSynchronize(),
 * to avoid bus errors either caused by either trying to access unified memory during a kernel 
 * call (alloc_managed() writes the buffer size to the buffer) or after the kernel has been called
 * but the results have not been synchronized.
 */
#define call_cuda_kernel(expr) {\
    obj_tracker_internal_enter();\
    expr;\
    obj_tracker_internal_leave();\
    if (b2c_must_synchronize)\
        cudaDeviceSynchronize();\
} while (0)

struct options {
    bool debug_execfail;
    bool debug_exec;
    bool trace_copy;
};

extern struct options b2c_options;
extern cublasHandle_t b2c_handle;
extern bool b2c_must_synchronize;

#ifdef __cplusplus
extern "C" {
#endif

void init_cublas(void);

/**
 * If status is not "cudaSuccess", prints the error message and exits.
 */
static inline void b2c_fatal_error(cudaError_t status, const char *domain)
{
    if (status != cudaSuccess) {
        writef(STDERR_FILENO, "%s: %s : %s\n", domain, cudaGetErrorName(status), cudaGetErrorString(status));
        abort();
    }
}

/**
 * Creates a new GPU buffer and copies the CPU buffer to it.
 * Returns the GPU buffer, which may be NULL on failure.
 * Free with cudaFree().
 */
void *b2c_copy_to_gpu(const void *hostbuf, size_t size);

/**
 * Creates a new CPU buffer and copies the GPU buffer to it.
 * Returns the CPU buffer, which may be NULL on failure.
 * Free with free().
 */
void *b2c_copy_to_cpu(const void *gpubuf, size_t size);

/**
 * Copies an existing GPU buffer to an existing CPU buffer.
 */
void b2c_copy_from_gpu(void *hostbuf, const void *gpubuf, size_t size);

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
        void *gpubuf2,
        ...);

/**
 * If {info} is NULL, the GPU buffer will be freed with cudaFree(). Otherwise
 * nothing will happen.
 * If there is a fatal error, this function will abort.
 */
void b2c_cleanup_gpu_ptr(void *gpubuf, const struct objinfo *info);

#ifdef __cplusplus
};
#endif

#endif
