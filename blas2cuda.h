#ifndef BLAS2CUDA_H
#define BLAS2CUDA_H

#include <stdbool.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define B2C_ERRORCHECK(name, status) \
do {\
    if (status == CUBLAS_STATUS_EXECUTION_FAILED) {\
        if (b2c_options.debug_execfail) {\
            fprintf(stderr, "blas2cuda: failed to execute " #name "\n");\
        }\
    }\
} while (0)

struct options {
    bool debug_execfail;
    bool trace_copy;
};

extern struct options b2c_options;
extern cublasHandle_t b2c_handle;

/**
 * Copy host (CPU) buffer to GPU. Returns a pointer to GPU buffer.
 * If copy fails, returns NULL.
 * Free with cudaFree().
 */
void *b2c_copy_to_gpu(const void *devbuf, size_t size);

/**
 * Copy GPU buffer to CPU.
 * If copy fails, returns NULL.
 * Free with free().
 */
void *b2c_copy_to_cpu(const void *gpubuf, size_t size);

#endif
