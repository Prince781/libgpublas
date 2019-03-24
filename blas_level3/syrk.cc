#include "level3.h"

template <typename T>
void _cblas_syrk(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const T alpha,
        const T *a, const int lda,
        const T beta,
        T *c, const int ldc,
        cublasStatus_t syrk_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_a;
    T *gpu_c;
    int rows_a, cols_a,
        rows_c, cols_c;
    int size_a, size_c;
    cublasFillMode_t cuplo = cu(uplo);
    cublasOperation_t ctrans = cu(trans);
    const struct objinfo *a_info, *c_info;

    rows_c = ldc;
    cols_c = n;
    if (trans == CblasNoTrans) {
        if (Layout == CblasColMajor) {
            rows_a = lda;
            cols_a = k;
        } else {
            rows_a = lda;
            cols_a = n;
        }
    } else {
        if (Layout == CblasColMajor) {
            rows_a = lda;
            cols_a = n;
        } else {
            rows_a = lda;
            cols_a = k;
        }
    }


    size_a = size(0, rows_a, cols_a, sizeof(*a));
    size_c = size(0, rows_c, cols_c, sizeof(*c));

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        c_info = NULL;

        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);
        gpu_c = transpose(c, size_c, &rows_c, &cols_c, ldc, geam_func);
    } else {
        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
        gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
                (void *) gpu_a, &a_info,
                NULL);
    }

    call_kernel(
        syrk_func(b2c_handle,
                cuplo, ctrans,
                n, k,
                &alpha,
                gpu_a, lda,
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
    b2c_cleanup_gpu_ptr((void *) gpu_c, c_info);

}

DECLARE_CBLAS__SYRK(s, float) {
    _cblas_syrk(Layout, uplo, trans,
            n, k,
            alpha,
            a, lda,
            beta,
            c, ldc,
            &cublasSsyrk,
            &cublasSgeam);
}

DECLARE_CBLAS__SYRK(d, double) {
    _cblas_syrk(Layout, uplo, trans,
            n, k,
            alpha,
            a, lda,
            beta,
            c, ldc,
            &cublasDsyrk,
            &cublasDgeam);
}

DECLARE_CBLAS__SYRK(c, float _Complex) {
    _cblas_syrk(Layout, uplo, trans,
            n, k,
            cu(alpha),
            (cuComplex *) a, lda,
            cu(beta),
            (cuComplex *) c, ldc,
            &cublasCsyrk,
            &cublasCgeam);
}

DECLARE_CBLAS__SYRK(z, double _Complex) {
    _cblas_syrk(Layout, uplo, trans,
            n, k,
            cu(alpha),
            (cuDoubleComplex *) a, lda,
            cu(beta),
            (cuDoubleComplex *) c, ldc,
            &cublasZsyrk,
            &cublasZgeam);
}
