#include "level1.h"

template <typename T>
static void _cblas_rotm (const int n,
        T *x, const int incx,
        T *y, const int incy,
        const T *param,
        cublasStatus_t rotm_func(cublasHandle_t, int, T *, int, T *, int, const T *))
{
    T *gpu_x, *gpu_y;
    const struct objinfo *x_info, *y_info;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));


    gpu_x = (T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);
    gpu_y = (T *) b2c_place_on_gpu((void *) y, size_y, &y_info, 
            (void *) gpu_x, x_info,
            NULL);

    rotm_func(b2c_handle, n, gpu_x, incx, gpu_y, incy, param);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!x_info)
        b2c_copy_from_gpu(x, gpu_x, size_x);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

void cblas_srotm (const int n,
        float *x, const int incx, 
        float *y, const int incy, 
        const float *param)
{
    _cblas_rotm(n, x, incx, y, incy, param, &cublasSrotm);
}

void cblas_drotm (const int n, 
        double *x, const int incx, 
        double *y, const int incy, 
        const double *param)
{
    _cblas_rotm(n, x, incx, y, incy, param, &cublasDrotm);
}
