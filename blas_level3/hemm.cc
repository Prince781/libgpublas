#include "level3.h"

template <typename T>
void _cblas_hemm(const CBLAS_LAYOUT Layout,
        const CBLAS_SIDE side,
        const CBLAS_UPLO uplo,
        const int m, const int n,
        const T alpha,
        const T *a, const int lda,
        const T *b, const int ldb,
        const T beta,
        T *c, const int ldc,
        cublasStatus_t hemm_func(cublasHandle_t,
            cublasSideMode_t, cublasFillMode_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_a, *gpu_b;
    T *gpu_c;
    int rows_a, cols_a,
        rows_b, cols_b,
        rows_c, cols_c;
    int size_a, size_b, size_c;
    cublasSideMode_t cside = cu(side);
    cublasFillMode_t cuplo = cu(uplo);
    const struct objinfo *a_info, *b_info, *c_info;

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        b_info = NULL;
        c_info = NULL;

        rows_a = lda;
        cols_a = (side == CblasLeft) ? m : n;
        size_a = size(0, rows_a, cols_a, sizeof(*a));
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);

        rows_b = lda;
        cols_b = m;
        size_b = size(0, rows_b, cols_b, sizeof(*b));
        gpu_b = transpose(b, size_b, &rows_b, &cols_b, ldb, geam_func);

        rows_c = m;
        cols_c = n;
        size_c = size(0, rows_c, cols_c, sizeof(*c));
        gpu_c = transpose(c, size_c, &rows_c, &cols_c, ldc, geam_func);
    } else {
        cols_a = lda;
        rows_a = (side == CblasLeft) ? m : n;
        size_a = size(0, rows_a, cols_a, sizeof(*a));

        cols_b = ldb;
        rows_b = (side == CblasLeft) ? m : n;
        size_b = size(0, rows_b, cols_b, sizeof(*b));

        cols_c = m;
        rows_c = n;
        size_c = size(0, rows_c, cols_c, sizeof(*c));

        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
        gpu_b = (T *) b2c_place_on_gpu((void *) b, size_b, &b_info, 
                (void *) gpu_a, &a_info,
                NULL);
        gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
                (void *) gpu_a, &a_info,
                (void *) gpu_b, &b_info,
                NULL);
    }

    hemm_func(b2c_handle,
            cside, cuplo,
            m, n,
            &alpha,
            gpu_a, lda,
            gpu_b, ldb,
            &beta,
            gpu_c, ldc);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!c_info) {
        if (Layout == CblasRowMajor)
            transpose(gpu_c, size_c, &rows_c, &cols_c, ldc, geam_func);
        b2c_copy_from_gpu(c, gpu_c, size_c);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_b, b_info);
    b2c_cleanup_gpu_ptr((void *) gpu_c, c_info);
}


DECLARE_CBLAS__HEMM(c, float _Complex) {
    _cblas_hemm(Layout,
            side, uplo,
            m, n, 
            cu(*(typeof(*a) *)alpha),
            (cuComplex *)a, lda,
            (cuComplex *)b, ldb,
            cu(*(typeof(*a) *)beta),
            (cuComplex *)c, ldc,
            &cublasChemm,
            &cublasCgeam);
}

DECLARE_CBLAS__HEMM(z, double _Complex) {
    _cblas_hemm(Layout,
            side, uplo,
            m, n, 
            cu(*(typeof(*a) *)alpha),
            (cuDoubleComplex *)a, lda,
            (cuDoubleComplex *)b, ldb,
            cu(*(typeof(*a) *)beta),
            (cuDoubleComplex *)c, ldc,
            &cublasZhemm,
            &cublasZgeam);
}
