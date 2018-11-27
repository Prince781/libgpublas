#include "level3.h"

template <typename T>
void _cblas_trmm(const CBLAS_LAYOUT Layout,
        const CBLAS_SIDE side,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_DIAG diag,
        const int m, const int n,
        const T alpha,
        const T *a, const int lda,
        T *b, const int ldb,
        cublasStatus_t trmm_func(cublasHandle_t,
            cublasSideMode_t,
            cublasFillMode_t,
            cublasOperation_t,
            cublasDiagType_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_a;
    T *gpu_b;
    int rows_a, cols_a,
        rows_b, cols_b;
    int size_a, size_b;
    cublasSideMode_t cside = cu(side);
    cublasFillMode_t cuplo = cu(uplo);
    cublasOperation_t ctransa = cu(transa);
    cublasDiagType_t cdiag = cu(diag);
    const struct objinfo *a_info, *b_info;

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        b_info = NULL;

        rows_a = lda;
        cols_a = (side == CblasLeft) ? m : n;
        size_a = size(0, rows_a, cols_a, sizeof(*a));
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);

        rows_b = lda;
        cols_b = m;
        size_b = size(0, rows_b, cols_b, sizeof(*b));
        gpu_b = transpose(b, size_b, &rows_b, &cols_b, ldb, geam_func);
    } else {
        cols_a = lda;
        rows_a = (side == CblasLeft) ? m : n;
        size_a = size(0, rows_a, cols_a, sizeof(*a));

        cols_b = ldb;
        rows_b = (side == CblasLeft) ? m : n;
        size_b = size(0, rows_b, cols_b, sizeof(*b));

        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
        gpu_b = (T *) b2c_place_on_gpu((void *) b, size_b, &b_info, 
                (void *) gpu_a, &a_info,
                NULL);
    }

    call_cuda_kernel(
        trmm_func(b2c_handle,
                cside, cuplo,
                ctransa, cdiag,
                m, n,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                gpu_b, ldb)
    );

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!b_info) {
        if (Layout == CblasRowMajor)
            transpose(gpu_b, size_b, &rows_b, &cols_b, ldb, geam_func);
        b2c_copy_from_gpu(b, gpu_b, size_b);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_b, b_info);
}

DECLARE_CBLAS__TRMM(s, float) {
    _cblas_trmm(Layout, 
            side, uplo, 
            transa, diag,
            m, n,
            alpha,
            a, lda,
            b, ldb,
            &cublasStrmm,
            &cublasSgeam);
}

DECLARE_CBLAS__TRMM(d, double) {
    _cblas_trmm(Layout, 
            side, uplo, 
            transa, diag,
            m, n,
            alpha,
            a, lda,
            b, ldb,
            &cublasDtrmm,
            &cublasDgeam);
}

DECLARE_CBLAS__TRMM(c, float _Complex) {
    _cblas_trmm(Layout, 
            side, uplo, 
            transa, diag,
            m, n,
            cu(alpha),
            (cuComplex *) a, lda,
            (cuComplex *) b, ldb,
            &cublasCtrmm,
            &cublasCgeam);
}

DECLARE_CBLAS__TRMM(z, double _Complex) {
    _cblas_trmm(Layout, 
            side, uplo, 
            transa, diag,
            m, n,
            cu(alpha),
            (cuDoubleComplex *) a, lda,
            (cuDoubleComplex *) b, ldb,
            &cublasZtrmm,
            &cublasZgeam);
}
