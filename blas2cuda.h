#ifndef BLAS2CUDA_H
#define BLAS2CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdbool.h>

struct options {
    bool debug;
};

extern struct options b2c_options;
extern cublasHandle_t b2c_handle;

#endif
