#include "level1.h"

template <typename T, typename R>
static void _cblas_asum (const int n,
        const T *x, 
        const int incx, 
        R *result,
#if USE_CUDA
        cublasStatus_t asum_func(cublasHandle_t, const int, const T *, const int, R *)
#elif USE_OPENCL
        decltype(clblast::Asum<T>) *asum_func
#endif
        )
{
    const T *gpu_x;
    const int size_x = b2c_size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;
    b2c_runtime_error_t err;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, &err, NULL);

#if USE_CUDA
    call_cuda_kernel(asum_func(b2c_handle, n, gpu_x, incx, result));
    err = cudaGetLastError();
#elif USE_OPENCL
    /* TODO */
#endif

    b2c_fatal_error(err, __func__);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info, &err);
}

DECLARE_CBLAS__ASUM(s, float) {
    float result;

#if USE_CUDA
    _cblas_asum(n, x, incx, &result, &cublasSasum);
#elif USE_OPENCL
    _cblas_asum(n, x, incx, &result, &clblast::Asum<float>);
#endif
    return result;
}

DECLARE_CBLAS__ASUM(sc, float _Complex) {
    float result;

#if USE_CUDA
    _cblas_asum(n, (cuComplex *) x, incx, &result, &cublasScasum);
#elif USE_OPENCL
    _cblas_asum(n, x, incx, &result, &clblast::Asum<float _Complex>);
#endif
    return result;
}

DECLARE_CBLAS__ASUM(d, double) {
    double result;

#if USE_CUDA
    _cblas_asum(n, x, incx, &result, &cublasDasum);
#elif USE_OPENCL
    _cblas_asum(n, x, incx, &result, &clblast::Asum<double>);
#endif
    return result;
}

DECLARE_CBLAS__ASUM(dz, double _Complex) {
    double result;

#if USE_CUDA
    _cblas_asum(n, (cuDoubleComplex *) x, incx, &result, &cublasDzasum);
#elif USE_OPENCL
    _cblas_asum(n, x, incx, &result, &clblast::Asum<double _Complex>);
#endif
    return result;
}
