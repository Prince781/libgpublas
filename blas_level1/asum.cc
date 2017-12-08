#include "level1.h"
#include "../lib/callinfo.h"
#include "../lib/obj_tracker.h"

template <typename T, typename R>
static void _cblas_asum (const int n,
        const T *x, 
        const int incx, 
        R *result,
        cublasStatus_t asum_func(cublasHandle_t, const int, const T *, const int, R *))
{
    const T *gpu_x;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);

    asum_func(b2c_handle, n, gpu_x, incx, result);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__ASUM(s, float) {
    float result;

    _cblas_asum(n, x, incx, &result, &cublasSasum);
    return result;
}

DECLARE_CBLAS__ASUM(sc, float _Complex) {
    float result;

    _cblas_asum(n, (cuComplex *) x, incx, &result, &cublasScasum);
    return result;
}

DECLARE_CBLAS__ASUM(d, double) {
    double result;

    _cblas_asum(n, x, incx, &result, &cublasDasum);
    return result;
}

DECLARE_CBLAS__ASUM(dz, double _Complex) {
    double result;

    _cblas_asum(n, (cuDoubleComplex *) x, incx, &result, &cublasDzasum);
    return result;
}
