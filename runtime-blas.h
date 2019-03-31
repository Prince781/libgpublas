// abstraction layer for BLAS runtimes
#ifndef BLAS_RUNTIME_H
#define BLAS_RUNTIME_H

#if USE_CUDA
#include <cublas_v2.h>
#include <stdio.h>

typedef cublasStatus_t runtime_blas_error_t;

#define RUNTIME_BLAS_ERROR_SUCCESS CUBLAS_STATUS_SUCCESS 

static inline const char *runtime_blas_error_msg(runtime_blas_error_t error) {
    static char buf[1024];
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "The operation completed successfully";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "The cuBLAS library was not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "Resource allocation failed inside the cuBLAS library";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "An unsupported value or parameter was passed to the function";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "The function requires a feature absent from the device architecture";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "An access to GPU memory space failed";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "The GPU program failed to execute";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "An internal cuBLAS operation failed";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "The functionnality requested is not supported";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "The functionnality requested requires some license and an error was detected when trying to check the current license";
        default:
            {
                snprintf(buf, sizeof buf, "Unknown cuBLAS error (%d)", (int) error);
                return buf;
            }
            break;
    }
}

#elif USE_OPENCL
#include <CL/cl.h>
#include "clext.h"
#include <clblast_c.h>
#include "clblast_ext.h"
#include <string.h>

typedef CLBlastStatusCode runtime_blas_error_t;

#define RUNTIME_BLAS_ERROR_SUCCESS CL_SUCCESS

static inline const char *runtime_blas_error_msg(runtime_blas_error_t error) {
    const char *cl_errmsg = clGetErrorString(error);

    if (strcmp(cl_errmsg, "CL_UNKNOWN_ERROR") != 0)
        return CLBlastGetErrorString(error);
    else
        return cl_errmsg;
}

#else
#error "Only CUDA and OpenCL are supported"
#endif

#ifdef __cplusplus
extern "C" {
#endif

runtime_blas_error_t runtime_blas_init(void);

runtime_blas_error_t runtime_blas_fini(void);

#ifdef __cplusplus
};
#endif

#endif
