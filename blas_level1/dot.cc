#include "level1.h"

template <typename T>
static void _cblas_dot (const int n,
        const T *x,
        const int incx,
        const T *y,
        const int incy,
        T *result,
#if USE_CUDA
        cublasStatus_t dot_func(cublasHandle_t, int, const T *, int, const T *, int, T *)
#elif USE_OPENCL
        decltype(clblast::Dot<T>) *dot_func
#endif
        )
{
    const T *gpu_x, *gpu_y;
    const int size_x = b2c_size(1, n-1, incx, sizeof(*x));
    const int size_y = b2c_size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info;
    b2c_runtime_error_t err;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, &err, NULL);
    gpu_y = (const T *) b2c_place_on_gpu((void *) y, size_y, &y_info, &err, 
            (void *) gpu_x, x_info,
            NULL);

#if USE_CUDA
    call_cuda_kernel(dot_func(b2c_handle, n, gpu_x, incx, gpu_y, incy, result));
    err = cudaGetLastError();
#elif USE_OPENCL
#endif

    b2c_fatal_error(err, __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info, &err);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info, &err);
}

DECLARE_CBLAS__DOT(s, float) {
    float result;
#if USE_CUDA
    _cblas_dot(n, x, incx, y, incy, &result, &cublasSdot);
#elif USE_OPENCL
    _cblas_dot(n, x, incx, y, incy, &result, &clblast::Dot<float>);
#endif
    return result;
}

DECLARE_CBLAS__DOT(d, double) {
    double result;
#if USE_CUDA
    _cblas_dot(n, x, incx, y, incy, &result, &cublasDdot);
#elif USE_OPENCl
    _cblas_dot(n, x, incx, y, incy, &result, &clblast::Dot<double>);
#endif
    return result;
}
