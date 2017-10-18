#include "blas2cuda.h"
#include <stdio.h>

cublasHandle_t b2c_handle;

__attribute__((constructor))
void blas2cuda_init(void)
{
    switch (cublasCreate(&b2c_handle)) {
        case CUBLAS_STATUS_SUCCESS:
            /* do nothing */
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            fprintf(stderr, "cuBLAS: failed to allocate resources\n");
        case CUBLAS_STATUS_NOT_INITIALIZED:
        default:
            fprintf(stderr, "cuBLAS: failed to initialize cuBLAS\n");
            exit(EXIT_FAILURE);
            break;
    }
}

__attribute__((destructor))
void blas2cuda_fini(void)
{
    if (cublasDestroy(b2c_handle) == CUBLAS_STATUS_NOT_INITIALIZED)
        fprintf(stderr, "cuBLAS: failed to destroy. Not initialized\n");
}
