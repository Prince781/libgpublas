#include "level1.h"

template <typename T, typename R>
static void _cblas_i_amin(const int n,
        const T *x,
        const int incx,
        R *result,
        cublasStatus_t amin_func(cublasHandle_t, int, 
            const T *, int,
            R *))
{
    const T *gpu_x;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);

    amin_func(b2c_handle, n, gpu_x, incx, result);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS_I_AMIN(s, float) {
    CBLAS_INDEX idx;
    _cblas_i_amin(n, x, incx, &idx, &cublasIsamin);
    return idx;
}

DECLARE_CBLAS_I_AMIN(d, double) {
    CBLAS_INDEX idx;
    _cblas_i_amin(n, x, incx, &idx, &cublasIdamin);
    return idx;

}

DECLARE_CBLAS_I_AMIN(c, float _Complex) {
    CBLAS_INDEX idx;
    _cblas_i_amin(n, (cuComplex *) x, incx, &idx, &cublasIcamin);
    return idx;
}

DECLARE_CBLAS_I_AMIN(z, double _Complex) {
    CBLAS_INDEX idx;
    _cblas_i_amin(n, (cuDoubleComplex *) x, incx, &idx, &cublasIzamin);
    return idx;
}

