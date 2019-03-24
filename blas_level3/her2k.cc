#include "level3.h"

template <typename S, typename T>
void _cblas_her2k(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const T alpha,
        const T *a, const int lda,
        const T *b, const int ldb,
        const S beta,
        T *c, const int ldc,
        cublasStatus_t her2k_func(cublasHandle_t,
            cublasFillMode_t,
            cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            const S *,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_a, *gpu_b;
    T *gpu_c;
    int rows_a, cols_a,
        rows_b, cols_b,
        rows_c, cols_c;
    int size_a, size_b, size_c;
    cublasFillMode_t cuplo = cu(uplo);
    cublasOperation_t ctrans = cu(trans);
    const struct objinfo *a_info, *b_info, *c_info;

    rows_c = ldc;
    cols_c = n;
    if (trans == CblasNoTrans) {
        if (Layout == CblasColMajor) {
            rows_a = lda;
            cols_a = k;
            rows_b = ldb;
            cols_b = k;
        } else {
            rows_a = lda;
            cols_a = n;
            rows_b = lda;
            cols_b = n;
        }
    } else {
        if (Layout == CblasColMajor) {
            rows_a = lda;
            cols_a = n;
            rows_b = ldb;
            cols_b = n;
        } else {
            rows_a = lda;
            cols_a = k;
            rows_b = ldb;
            cols_b = k;
        }
    }


    size_a = size(0, rows_a, cols_a, sizeof(*a));
    size_b = size(0, rows_b, cols_b, sizeof(*b));
    size_c = size(0, rows_c, cols_c, sizeof(*c));

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        b_info = NULL;
        c_info = NULL;

        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);
        gpu_b = transpose(b, size_b, &rows_b, &cols_b, ldb, geam_func);
        gpu_c = transpose(c, size_c, &rows_c, &cols_c, ldc, geam_func);
    } else {
        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
        gpu_b = (T *) b2c_place_on_gpu((void *) b, size_b, &b_info, 
                (void *) gpu_a, &a_info,
                NULL);
        gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
                (void *) gpu_a, &a_info,
                (void *) gpu_b, &b_info,
                NULL);
    }

    call_kernel(
        her2k_func(b2c_handle,
                cuplo, ctrans,
                n, k,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                &beta,
                gpu_c, ldc)
    );

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

DECLARE_CBLAS__HER2K(c, float, float _Complex) {
    _cblas_her2k(Layout, uplo, trans,
            n, k,
            cu(*alpha),
            (cuComplex *) a, lda,
            (cuComplex *) b, ldb,
            beta,
            (cuComplex *) c, ldc,
            &cublasCher2k,
            &cublasCgeam);
}

DECLARE_CBLAS__HER2K(z, double, double _Complex) {
    _cblas_her2k(Layout, uplo, trans,
            n, k,
            cu(*alpha),
            (cuDoubleComplex *) a, lda,
            (cuDoubleComplex *) b, ldb,
            beta,
            (cuDoubleComplex *) c, ldc,
            &cublasZher2k,
            &cublasZgeam);
}
