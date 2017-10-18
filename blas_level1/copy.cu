#include "level1.h"

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void cblas_scopy (const int n,
        const float *x,
        const int incx, 
        float *y, 
        const int incy)
{
    float *gpu_x;
    float *gpu_y;
    cublasStatus_t status;

    gpu_x = (float *) b2c_copy_to_gpu(x, n * sizeof(*x));

    if (gpu_x == NULL || cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    gpu_y = (float *) b2c_copy_to_gpu(y, n * sizeof(*y));

    if (gpu_y == NULL || cudaPeekAtLastError() != cudaSuccess) {
        cudaFree(gpu_x);
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    status = cublasScopy(b2c_handle, n, gpu_x, incx, gpu_y, incy);

    B2C_ERRORCHECK(cblas_scopy, status);

    cudaFree(gpu_x);
    cudaFree(gpu_y);
}
