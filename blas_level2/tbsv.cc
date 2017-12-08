#include "level2.h"

template <typename T>
void _cblas_tbsv(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const CBLAS_DIAG diag,
        const int n, const int k,
        const T *a, const int lda,
        T *x, const int incx,
        cublasStatus_t tbsv_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            cublasDiagType_t,
            int, int,
            const T *, int,
            T *, int),
        geam_t<T> geam_func,
        T geam_alpha = 1.0, T geam_beta = 0.0)
{
    const T *gpu_a;
    T *gpu_x;
    const int size_a = size(n, lda, sizeof(*a));
    const int size_x = size(n, incx, sizeof(*x));
    const struct objinfo *a_info, *x_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = (cublasFillMode_t) (uplo - CblasUpper);
    const cublasOperation_t op = (cublasOperation_t) (trans - CblasNoTrans);
    const cublasDiagType_t cdiag = (cublasDiagType_t) (diag - CblasNonUnit);

    if (Layout == CblasRowMajor) {
        T *gpu_a_trans;

        a_info = NULL;
        rows_a = n;
        cols_a = lda;

        gpu_a_trans = (T *) b2c_copy_to_gpu((void *) a, size_a);
        
        /* transpose A */
        geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                rows_a, cols_a,
                &geam_alpha,
                gpu_a_trans, lda,
                &geam_beta,
                NULL, 0,
                gpu_a_trans, lda);
        
        if (cudaPeekAtLastError() != cudaSuccess)
            b2c_fatal_error(cudaGetLastError(), __func__);

        gpu_a = gpu_a_trans;
    } else {
        gpu_a = (const T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);

    tbsv_func(b2c_handle, fillmode,
            op, cdiag,
            n, k,
            gpu_a, lda,
            gpu_x, incx);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!x_info)
        b2c_copy_from_gpu(x, gpu_x, size_x);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);

}

DECLARE_CBLAS__TBSV(s, float) {
    _cblas_tbsv(Layout, uplo, trans, diag,
            n, k,
            a, lda,
            x, incx,
            &cublasStbsv,
            &cublasSgeam);
}

DECLARE_CBLAS__TBSV(d, double) {
    _cblas_tbsv(Layout, uplo, trans, diag,
            n, k,
            a, lda,
            x, incx,
            &cublasDtbsv,
            &cublasDgeam);
}

DECLARE_CBLAS__TBSV(c, float _Complex) {
    float _Complex _alpha = 1.0;
    float _Complex _beta = 0.0;
    _cblas_tbsv(Layout, uplo, trans, diag,
            n, k,
            (cuComplex *) a, lda,
            (cuComplex *) x, incx,
            &cublasCtbsv,
            &cublasCgeam,
            *(cuComplex *) &_alpha,
            *(cuComplex *) &_beta);
}

DECLARE_CBLAS__TBSV(z, double _Complex) {
    double _Complex _alpha = 1.0;
    double _Complex _beta = 0.0;
    _cblas_tbsv(Layout, uplo, trans, diag,
            n, k,
            (cuDoubleComplex *) a, lda,
            (cuDoubleComplex *) x, incx,
            &cublasZtbsv,
            &cublasZgeam,
            *(cuDoubleComplex *) &_alpha,
            *(cuDoubleComplex *) &_beta);
}