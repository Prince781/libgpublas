#include "level1.h"

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define size(n,stride,sz) (n * stride * sz)

void cblas_scopy (const int n,
        const float *x,
        const int incx, 
        float *y, 
        const int incy)
{
    float *gpu_x;
    float *gpu_y;
    cublasStatus_t status;
    const int size_x = size(n,incx,sizeof(*x));
    const int size_y = size(n,incy,sizeof(*y));

    gpu_x = (float *) b2c_copy_to_gpu(x, size_x);

    if (gpu_x == NULL || cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    cudaMalloc(&gpu_y, size_y);

    if (gpu_y == NULL || cudaPeekAtLastError() != cudaSuccess) {
        cudaFree(gpu_x);
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    status = cublasScopy(b2c_handle, n, gpu_x, incx, gpu_y, incy);

    B2C_ERRORCHECK(cblas_scopy, status);

    cudaMemcpy(y, gpu_y, size_y, cudaMemcpyDeviceToHost);

    cudaFree(gpu_x);
    cudaFree(gpu_y);
}
