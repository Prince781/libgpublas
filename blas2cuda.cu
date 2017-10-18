#include "blas2cuda.h"
#include <stdio.h>
#include <stdlib.h>

cublasHandle_t b2c_handle;

struct options b2c_options = { false };

static void set_options(void) {
    /* TODO: use secure_getenv() ? */
    char *options = getenv("BLAS2CUDA_OPTIONS");
    char *saveptr = NULL;
    char *option = NULL;

    option = strtok_r(options, ",", &saveptr);
    while (option != NULL) {
        if (strcmp(option, "debug") == 0)
            b2c_options.debug = true;
        option = strtok_r(NULL, ",", &saveptr);
    }
}

void *b2c_copy_to_gpu(const void *devbuf, size_t size)
{
    void *gpubuf = NULL;

    cudaMalloc(&gpubuf, size);

    if (gpubuf == NULL)
        return gpubuf;

    cudaMemcpy(gpubuf, devbuf, size, cudaMemcpyHostToDevice);

    return gpubuf;
}

void *b2c_copy_to_cpu(const void *gpubuf, size_t size)
{
    void *devbuf = NULL;

    devbuf = malloc(size);

    if (devbuf == NULL)
        return devbuf;

    cudaMemcpy(devbuf, gpubuf, size, cudaMemcpyDeviceToHost);

    return devbuf;
}

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

    set_options();
}

__attribute__((destructor))
void blas2cuda_fini(void)
{
    if (cublasDestroy(b2c_handle) == CUBLAS_STATUS_NOT_INITIALIZED)
        fprintf(stderr, "cuBLAS: failed to destroy. Not initialized\n");
}
