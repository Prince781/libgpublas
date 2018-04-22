#include "level1.h"
#include <cuComplex.h>

template <typename T>
static void _cblas_axpy (const int n,
        const T a,
        const T *x, const int incx,
        T *y, const int incy,
        cublasStatus_t axpy_func(cublasHandle_t, int,
            const T *,
            const T *, int,
            T *, int))
{
    const T *gpu_x;
    T *gpu_y;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);
    gpu_y = (T *) b2c_place_on_gpu(NULL, size_y, &y_info, 
            (void *) gpu_x, x_info,
            NULL);

    axpy_func(b2c_handle, n, &a, gpu_x, incx, gpu_y, incy);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__AXPY(s, float) {
    _cblas_axpy(n, a, x, incx, y, incy, &cublasSaxpy);
}

DECLARE_CBLAS__AXPY(d, double) {
    _cblas_axpy(n, a, x, incx, y, incy, &cublasDaxpy);
}

DECLARE_CBLAS__AXPY(c, float _Complex) {
    _cblas_axpy(n, cu(a), (const cuComplex *) x, incx, (cuComplex *) y, incy, &cublasCaxpy);
}

DECLARE_CBLAS__AXPY(z, double _Complex) {
    _cblas_axpy(n, cu(a), (const cuDoubleComplex *) x, incx,
            (cuDoubleComplex *) y, incy, &cublasZaxpy);
}
