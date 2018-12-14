#include "level1.h"

template <typename T>
static void _cblas_axpy (const int n,
        const T a,
        const T *x, const int incx,
        T *y, const int incy,
#if USE_CUDA
        cublasStatus_t axpy_func(cublasHandle_t, int,
            const T *,
            const T *, int,
            T *, int)
#elif USE_OPENCL
        decltype(clblast::Axpy<T>) *axpy_func
#endif
        )
{
    const T *gpu_x;
    T *gpu_y;
    const int size_x = b2c_size(1, n-1, incx, sizeof(*x));
    const int size_y = b2c_size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info;
    b2c_runtime_error_t err;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, &err, NULL);
    gpu_y = (T *) b2c_place_on_gpu(NULL, size_y, &y_info, &err, 
            (void *) gpu_x, x_info,
            NULL);

#if USE_CUDA
    call_cuda_kernel(axpy_func(b2c_handle, n, &a, gpu_x, incx, gpu_y, incy));
    err = cudaGetLastError();
#elif USE_OPENCL
    /* TODO */
#endif

    b2c_fatal_error(err, __func__);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y, &err);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info, &err);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info, &err);
}

DECLARE_CBLAS__AXPY(s, float) {
#if USE_CUDA
    _cblas_axpy(n, a, x, incx, y, incy, &cublasSaxpy);
#elif USE_OPENCL
    _cblas_axpy(n, a, x, incx, y, incy, &clblast::Axpy<float>);
#endif
}

DECLARE_CBLAS__AXPY(d, double) {
#if USE_CUDA
    _cblas_axpy(n, a, x, incx, y, incy, &cublasDaxpy);
#elif USE_OPENCL
    _cblas_axpy(n, a, x, incx, y, incy, &clblast::Axpy<double>);
#endif
}

DECLARE_CBLAS__AXPY(c, float _Complex) {
#if USE_CUDA
    _cblas_axpy(n, cu(a), (const cuComplex *) x, incx, (cuComplex *) y, incy, &cublasCaxpy);
#elif USE_OPENCL
    _cblas_axpy(n, a, x, incx, y, incy, &clblast::Axpy<float _Complex>);
#endif
}

DECLARE_CBLAS__AXPY(z, double _Complex) {
#if USE_CUDA
    _cblas_axpy(n, cu(a), (const cuDoubleComplex *) x, incx,
            (cuDoubleComplex *) y, incy, &cublasZaxpy);
#elif USE_OPENCL
    _cblas_axpy(n, a, x, incx, y, incy, &clblast::Axpy<double _Complex>);
#endif
}
