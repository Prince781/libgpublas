#include "level1.h"

template <typename T, typename R>
static void _cblas_nrm2(const int n,
        const T *x,
        const int incx,
        R *result,
        cublasStatus_t nrm2_func(cublasHandle_t, int, const T *, int, R *))
{
    const T *gpu_x;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);

    nrm2_func(b2c_handle, n, gpu_x, incx, result);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__NRM2(s, float) {
    float result;
    _cblas_nrm2(n, x, incx, &result, &cublasSnrm2);
    return result;
}

DECLARE_CBLAS__NRM2(d, double) {
    double result;
    _cblas_nrm2(n, x, incx, &result, &cublasDnrm2);
    return result;
}

DECLARE_CBLAS__NRM2(sc, float _Complex) {
    float result;
    _cblas_nrm2(n, (const cuComplex *) x, incx, &result, &cublasScnrm2);
    return result;
}

DECLARE_CBLAS__NRM2(dz, double _Complex) {
    double result;
    _cblas_nrm2(n, (const cuDoubleComplex *) x, incx, &result, &cublasDznrm2);
    return result;
}

