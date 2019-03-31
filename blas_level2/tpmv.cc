#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level2.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

template <typename T>
void _cblas_tpmv(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const CBLAS_DIAG diag,
        const int n,
        const T *a,
        T *x, const int incx,
        cublasStatus_t tpmv_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            cublasDiagType_t,
            int,
            const T *, 
            T *, int),
        geam_t<T> geam_func,
        T geam_alpha = 1.0, T geam_beta = 0.0)
{
    const T *gpu_a;
    T *gpu_x;
    const int size_a = size(0, n, n+1, sizeof(*a))/2;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *a_info, *x_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);
    const cublasOperation_t op = cu(trans);
    const cublasDiagType_t cdiag = cu(diag);

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = n;
        cols_a = (n+1)/2;

        gpu_a = transpose(a, size_a, &rows_a, &cols_a, rows_a, geam_func);
    } else {
        rows_a = n;
        cols_a = (n+1)/2;
        gpu_a = (const T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);

    call_kernel(
        tpmv_func(b2c_handle, fillmode,
                op, cdiag,
                n,
                gpu_a,
                gpu_x, incx)
    );

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!x_info)
        b2c_copy_from_gpu(x, gpu_x, size_x);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__TPMV(s, float) {
    _cblas_tpmv(Layout, uplo, trans, diag,
            n,
            ap,
            x, incx,
            &cublasStpmv,
            &cublasSgeam);
}


DECLARE_CBLAS__TPMV(d, double) {
    _cblas_tpmv(Layout, uplo, trans, diag,
            n,
            ap,
            x, incx,
            &cublasDtpmv,
            &cublasDgeam);
}


DECLARE_CBLAS__TPMV(c, float _Complex) {
    float _Complex _alpha = 1.0;
    float _Complex _beta = 0.0;
    _cblas_tpmv(Layout, uplo, trans, diag,
            n,
            (cuComplex *) ap,
            (cuComplex *) x, incx,
            &cublasCtpmv,
            &cublasCgeam,
            cu(_alpha),
            cu(_beta));
}


DECLARE_CBLAS__TPMV(z, double _Complex) {
    double _Complex _alpha = 1.0;
    double _Complex _beta = 0.0;
    _cblas_tpmv(Layout, uplo, trans, diag,
            n,
            (cuDoubleComplex *) ap,
            (cuDoubleComplex *) x, incx,
            &cublasZtpmv,
            &cublasZgeam,
            cu(_alpha),
            cu(_beta));

}
