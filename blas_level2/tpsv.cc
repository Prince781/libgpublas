#include "level2.h"

template <typename T>
void _cblas_tpsv(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const CBLAS_DIAG diag,
        const int n,
        const T *ap,
        T *x, const int incx,
        cublasStatus_t tpsv_func(cublasHandle_t,
            cublasFillMode_t,
            cublasOperation_t, cublasDiagType_t,
            int, const T *,
            T *, int),
        geam_t<T> geam_func,
        T geam_alpha = 1.0, T geam_beta = 0.0)
{
}


DECLARE_CBLAS__TPSV(s, float) {
    _cblas_tpsv(Layout, uplo, trans, diag,
            n,
            ap,
            x, incx,
            &cublasStpsv,
            &cublasSgeam);
}

DECLARE_CBLAS__TPSV(d, double) {
    _cblas_tpsv(Layout, uplo, trans, diag,
            n,
            ap,
            x, incx,
            &cublasDtpsv,
            &cublasDgeam);
}

DECLARE_CBLAS__TPSV(c, float _Complex) {
    float _Complex _alpha = 1.0;
    float _Complex _beta = 0.0;
    _cblas_tpsv(Layout, uplo, trans, diag,
            n,
            (cuComplex *) ap,
            (cuComplex *) x, incx,
            &cublasCtpsv,
            &cublasCgeam,
            *(cuComplex *) &_alpha,
            *(cuComplex *) &_beta);
}

DECLARE_CBLAS__TPSV(z, double _Complex) {
    double _Complex _alpha = 1.0;
    double _Complex _beta = 0.0;
    _cblas_tpsv(Layout, uplo, trans, diag,
            n,
            (cuDoubleComplex *) ap,
            (cuDoubleComplex *) x, incx,
            &cublasZtpsv,
            &cublasZgeam,
            *(cuDoubleComplex *) &_alpha,
            *(cuDoubleComplex *) &_beta);
}
