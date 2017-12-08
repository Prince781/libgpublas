#include "level2.h"

template <typename T>
void _cblas_tbmv(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const CBLAS_DIAG diag,
        const int n, const int k,
        const T *a, const int lda,
        T *x, const int incx,
        cublasStatus_t tbmv_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            cublasDiagType_t,
            int, int,
            const T*, int,
            T *, int),
        geam_t<T> geam_func,
        T geam_alpha = 1.0, T geam_beta = 0.0)
{
    const T *gpu_a;
    T *gpu_x;
    const int size_a = size(0, n, lda, sizeof(*a));
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *a_info, *x_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);
    const cublasOperation_t op = cu(trans);
    const cublasDiagType_t cdiag = cu(diag);

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = lda;
        cols_a = n;

        gpu_a = transpose(a, size_a, &rows_a, &cols_a, rows_a, geam_func);
    } else {
        rows_a = lda;
        cols_a = n;
        gpu_a = (const T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);

    tbmv_func(b2c_handle, fillmode,
            op, cdiag,
            cols_a, k,
            gpu_a, rows_a,
            gpu_x, incx);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!x_info)
        b2c_copy_from_gpu(x, gpu_x, size_x);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__TBMV(s, float) {
    _cblas_tbmv(Layout, uplo, trans, diag,
            n, k,
            a, lda,
            x, incx,
            &cublasStbmv,
            &cublasSgeam);
}


DECLARE_CBLAS__TBMV(d, double) {
    _cblas_tbmv(Layout, uplo, trans, diag,
            n, k,
            a, lda,
            x, incx,
            &cublasDtbmv,
            &cublasDgeam);
}


DECLARE_CBLAS__TBMV(c, float _Complex) {
    float _Complex _alpha = 1.0;
    float _Complex _beta = 0.0;
    _cblas_tbmv(Layout, uplo, trans, diag,
            n, k,
            (cuComplex *) a, lda,
            (cuComplex *) x, incx,
            &cublasCtbmv,
            &cublasCgeam,
            cu(_alpha),
            cu(_beta));
}


DECLARE_CBLAS__TBMV(z, double _Complex) {
    double _Complex _alpha = 1.0;
    double _Complex _beta = 0.0;
    _cblas_tbmv(Layout, uplo, trans, diag,
            n, k,
            (cuDoubleComplex *) a, lda,
            (cuDoubleComplex *) x, incx,
            &cublasZtbmv,
            &cublasZgeam,
            cu(_alpha),
            cu(_beta));

}
