#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level2.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

template <typename S, typename T>
static void _cblas_hpr(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n,
        const S alpha,
        const T *x, const int incx,
        T *a,
        cublasStatus_t hpr_func(cublasHandle_t,
            cublasFillMode_t,
            int, const S *,
            const T *, int,
            T *),
        geam_t<T> geam_func,
        T geam_alpha, T geam_beta)
{
    T *gpu_a;
    const T *gpu_x;
    const int size_a = size(0, n, n+1, sizeof(*a))/2;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *a_info, *x_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = n;
        cols_a = (n+1)/2;

        gpu_a = transpose(a, size_a, &rows_a, &cols_a, rows_a, geam_func);
    } else {
        rows_a = n;
        cols_a = (n+1)/2;
        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);

    call_kernel(
        hpr_func(b2c_handle, fillmode, 
                n, &alpha,
                gpu_x, incx,
                gpu_a)
    );

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!a_info) {
        if (Layout == CblasRowMajor)
            transpose(gpu_a, size_a, &rows_a, &cols_a, n, geam_func);
        b2c_copy_from_gpu(a, gpu_a, size_a);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__HPR(c, float, float _Complex) {
    float _Complex _alpha = 1.0;
    float _Complex _beta = 1.0;
    _cblas_hpr(Layout, uplo,
            n, alpha, 
            (cuComplex *) x, incx,
            (cuComplex *) ap,
            &cublasChpr,
            &cublasCgeam,
            cu(_alpha),
            cu(_beta));
}

DECLARE_CBLAS__HPR(z, double, double _Complex) {
    double _Complex _alpha = 1.0;
    double _Complex _beta = 1.0;
    _cblas_hpr(Layout, uplo,
            n, alpha, 
            (cuDoubleComplex *) x, incx,
            (cuDoubleComplex *) ap,
            &cublasZhpr,
            &cublasZgeam,
            cu(_alpha),
            cu(_beta));
}
