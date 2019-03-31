#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level1.h"
#include "../blas2cuda.h"

template <typename T, typename R>
static void _cblas_i_amax(const int n,
        const T *x,
        const int incx,
        R *result,
        cublasStatus_t amax_func(cublasHandle_t, int, 
            const T *, int,
            R *))
{
    const T *gpu_x;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;
    extern cublasHandle_t b2c_handle;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);

    call_kernel(amax_func(b2c_handle, n, gpu_x, incx, result));

    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS_I_AMAX(s, float) {
    CBLAS_INDEX idx;
    _cblas_i_amax(n, x, incx, &idx, &cublasIsamax);
    return idx;
}

DECLARE_CBLAS_I_AMAX(d, double) {
    CBLAS_INDEX idx;
    _cblas_i_amax(n, x, incx, &idx, &cublasIdamax);
    return idx;

}

DECLARE_CBLAS_I_AMAX(c, float _Complex) {
    CBLAS_INDEX idx;
    _cblas_i_amax(n, (cuComplex *) x, incx, &idx, &cublasIcamax);
    return idx;
}

DECLARE_CBLAS_I_AMAX(z, double _Complex) {
    CBLAS_INDEX idx;
    _cblas_i_amax(n, (cuDoubleComplex *) x, incx, &idx, &cublasIzamax);
    return idx;
}

