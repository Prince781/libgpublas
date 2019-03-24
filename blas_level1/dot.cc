#include "level1.h"

template <typename T>
static void _cblas_dot (const int n,
        const T *x,
        const int incx,
        const T *y,
        const int incy,
        T *result,
        cublasStatus_t dot_func(cublasHandle_t, int, const T *, int, const T *, int, T *))
{
    const T *gpu_x, *gpu_y;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);
    gpu_y = (const T *) b2c_place_on_gpu((void *) y, size_y, &y_info, 
            (void *) gpu_x, x_info,
            NULL);

    call_kernel(dot_func(b2c_handle, n, gpu_x, incx, gpu_y, incy, result));

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__DOT(s, float) {
    float result;
    _cblas_dot(n, x, incx, y, incy, &result, &cublasSdot);
    return result;
}

DECLARE_CBLAS__DOT(d, double) {
    double result;
    _cblas_dot(n, x, incx, y, incy, &result, &cublasDdot);
    return result;
}
