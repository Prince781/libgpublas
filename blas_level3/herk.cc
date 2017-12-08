#include "level3.h"

template <typename S, typename T>
void _cblas_herk(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const S alpha,
        const T *a, const int lda,
        const S beta,
        T *c, const int ldc,
        cublasStatus_t herk_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            int, int,
            const S *,
            const T *, int,
            const S *,
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

    assign_dims(Layout, trans, rows_a, cols_a, lda, k, n);
    if (Layout == CblasRowMajor) {
        a_info = NULL;
        c_info = NULL;

        size_a = size(0, rows_a, cols_a, sizeof(*a));
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);

        rows_c = ldc;
        cols_c = n;
        size_c = size(0, rows_c, cols_c, sizeof(*c));
        gpu_c = transpose(c, size_c, &rows_c, &cols_c, ldc, geam_func);
    } else {
        size_a = size(0, rows_a, cols_a, sizeof(*a));
        
        rows_c = ldc;
        cols_c = n;
        size_c = size(0, rows_c, cols_c, sizeof(*c));

        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
        gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
                (void *) gpu_a, &a_info,
                NULL);
    }

    herk_func(b2c_handle,
            cuplo, ctrans,
            n, k,
            &alpha,
            gpu_a, lda,
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
    b2c_cleanup_gpu_ptr((void *) gpu_c, c_info);
}


DECLARE_CBLAS__HERK(c, float, float _Complex) {
    _cblas_herk(Layout, uplo, trans,
            n, k,
            alpha,
            (cuComplex *) a, lda,
            beta,
            (cuComplex *) c, ldc,
            &cublasCherk,
            &cublasCgeam);
}
