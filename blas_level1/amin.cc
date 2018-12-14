#include "level1.h"

template <typename T, typename R>
static void _cblas_i_amin(const int n,
        const T *x,
        const int incx,
        R *result
#if USE_CUDA
        ,
        cublasStatus_t amin_func(cublasHandle_t, int, 
            const T *, int,
            R *)
#endif
        )
{
    const T *gpu_x;
    const int size_x = b2c_size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;
    b2c_runtime_error_t err;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, &err, NULL);

#if USE_CUDA
    call_cuda_kernel(amin_func(b2c_handle, n, gpu_x, incx, result));
    err = cudaGetLastError();
#elif USE_OPENCL
    cl_host_ptr<R> result_clm(result);
    cl_host_const_ptr<const T> x_clm(gpu_x, size_x);
    clblast::Amin<T>(n, result_clm, 0, x_clm, 0, incx, &cl_queue);
#endif

    b2c_fatal_error(err, __func__);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info, &err);
}

DECLARE_CBLAS_I_AMIN(s, float) {
    CBLAS_INDEX idx;
#if USE_CUDA
    _cblas_i_amin(n, x, incx, &idx, &cublasIsamin);
#elif USE_OPENCL
    _cblas_i_amin(n, x, incx, &idx);
#endif
    return idx;
}

DECLARE_CBLAS_I_AMIN(d, double) {
    CBLAS_INDEX idx;
#if USE_CUDA
    _cblas_i_amin(n, x, incx, &idx, &cublasIdamin);
#elif USE_OPENCL
    _cblas_i_amin(n, x, incx, &idx);
#endif
    return idx;

}

DECLARE_CBLAS_I_AMIN(c, float _Complex) {
    CBLAS_INDEX idx;
#if USE_CUDA
    _cblas_i_amin(n, (cuComplex *) x, incx, &idx, &cublasIcamin);
#elif USE_OPENCL
    _cblas_i_amin(n, x, incx, &idx);
#endif
    return idx;
}

DECLARE_CBLAS_I_AMIN(z, double _Complex) {
    CBLAS_INDEX idx;
#if USE_CUDA
    _cblas_i_amin(n, (cuDoubleComplex *) x, incx, &idx, &cublasIzamin);
#elif USE_OPENCL
    _cblas_i_amin(n, x, incx, &idx);
#endif
    return idx;
}

