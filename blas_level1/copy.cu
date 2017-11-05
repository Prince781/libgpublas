#include "level1.h"
#include "../lib/callinfo.h"
#include "../lib/obj_tracker.h"

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define size(n,stride,sz) (n * stride * sz)

void cblas_scopy (const int n,
        const float *x,
        const int incx, 
        float *y, 
        const int incy)
{
    const float *gpu_x;
    float *gpu_y;
    cublasStatus_t status;
    const int size_x = size(n,incx,sizeof(*x));
    const int size_y = size(n,incy,sizeof(*y));
    const struct objinfo *x_info, *y_info;

    if ((x_info = obj_tracker_objinfo((void *) x))) {
        gpu_x = x;
    } else
        gpu_x = (const float *) b2c_copy_to_gpu(x, size_x);

    if (gpu_x == NULL || cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    if ((y_info = obj_tracker_objinfo((void *) y)))
        gpu_y = y;
    else
        cudaMalloc(&gpu_y, size_y);

    if (gpu_y == NULL || cudaPeekAtLastError() != cudaSuccess) {
        if (!x_info)
            cudaFree((void *) gpu_x);
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    status = cublasScopy(b2c_handle, n, gpu_x, incx, gpu_y, incy);

    B2C_ERRORCHECK(cblas_scopy, status);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    if (!x_info)
        cudaFree((void *) gpu_x);
    if (!y_info)
        cudaFree(gpu_y);
}
