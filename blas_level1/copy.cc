#include "level1.h"

template <typename T>
static void _cblas_copy (const int n,
        const T *x,
        const int incx, 
        T *y, 
        const int incy,
#if USE_CUDA
        cublasStatus_t copy_func(cublasHandle_t, int, const T *, int, T *, int)
#elif USE_OPENCL
        decltype(clblast::Copy<T>) *copy_func
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
    call_cuda_kernel(copy_func(b2c_handle, n, gpu_x, incx, gpu_y, incy));
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

DECLARE_CBLAS__COPY(s, float) {
#if USE_CUDA
    _cblas_copy(n, x, incx, y, incy, &cublasScopy);
#elif USE_OPENCL
    _cblas_copy(n, x, incx, y, incy, &clblast::Copy<float>);
#endif
}

DECLARE_CBLAS__COPY(d, double) {
#if USE_CUDA
    _cblas_copy(n, x, incx, y, incy, &cublasDcopy);
#elif USE_OPENCL
    _cblas_copy(n, x, incx, y, incy, &clblast::Copy<double>);
#endif
}

DECLARE_CBLAS__COPY(c, float _Complex) {
#if USE_CUDA
    _cblas_copy(n, (const cuComplex *) x, incx, (cuComplex *) y, incy, &cublasCcopy);
#elif USE_OPENCL
    _cblas_copy(n, x, incx, y, incy, &clblast::Copy<float _Complex>);
#endif
}

DECLARE_CBLAS__COPY(z, double _Complex) {
#if USE_CUDA
    _cblas_copy(n, (const cuDoubleComplex *) x, incx, (cuDoubleComplex *) y, incy, &cublasZcopy);
#elif USE_OPENCL
    _cblas_copy(n, x, incx, y, incy, &clblast::Copy<double _Complex>);
#endif
}
