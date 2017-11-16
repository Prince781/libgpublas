#include "level1.h"
#include "../lib/callinfo.h"
#include "../lib/obj_tracker.h"

typedef cublasStatus_t (*copy_t)(cublasHandle_t, int, const void *, int, void *,
        int);

static void _cblas_copy (const int n,
        const void *x,
        const int incx, 
        void *y, 
        const int incy,
        size_t elem_size,
        copy_t copy_func)
{
    const void *gpu_x;
    void *gpu_y;
    cublasStatus_t status;
    const int size_x = size(n,incx,elem_size);
    const int size_y = size(n,incy,elem_size);
    const struct objinfo *x_info, *y_info;

    if ((x_info = obj_tracker_objinfo((void *)x))) {
        gpu_x = x;
    } else
        gpu_x = (const void *) b2c_copy_to_gpu(x, size_x);

    if (gpu_x == NULL || cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    if ((y_info = obj_tracker_objinfo(y)))
        gpu_y = y;
    else
        cudaMalloc(&gpu_y, size_y);

    if (gpu_y == NULL || cudaPeekAtLastError() != cudaSuccess) {
        if (!x_info)
            cudaFree((void *) gpu_x);
        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        return;
    }

    status = copy_func(b2c_handle, n, gpu_x, incx, gpu_y, incy);

    B2C_ERRORCHECK(cblas_copy, status);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    if (!x_info)
        cudaFree((void *) gpu_x);
    if (!y_info)
        cudaFree(gpu_y);
}

DECLARE_CBLAS__COPY(s, float) {
    _cblas_copy(n, (const void *)x, incx, (void *)y, incy, sizeof(*x), (copy_t) &cublasScopy);
}

DECLARE_CBLAS__COPY(d, double) {
    _cblas_copy(n, (const void *)x, incx, (void *)y, incy, sizeof(*x), (copy_t) &cublasDcopy);
}

DECLARE_CBLAS__COPY(c, float _Complex) {
    _cblas_copy(n, (const void *)x, incx, (void *)y, incy, sizeof(*x), (copy_t) &cublasCcopy);
}

DECLARE_CBLAS__COPY(z, double _Complex) {
    _cblas_copy(n, (const void *)x, incx, (void *)y, incy, sizeof(*x), (copy_t) &cublasZcopy);
}
