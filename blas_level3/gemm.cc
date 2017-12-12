#include "level3.h"

template <typename T>
static inline int compute_size(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
        const T *a, const int lda, const int dim1, const int dim2)
{
    if ((Layout == CblasColMajor && transa == CblasNoTrans)
            || (Layout == CblasRowMajor && (transa == CblasTrans || transa == CblasConjTrans))) {
        return size(0, lda, dim1, sizeof(*a));
    }

    return size(0, lda, dim2, sizeof(*a));
}

template <typename T>
void _cblas_gemm(const CBLAS_LAYOUT Layout,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb,
        const int m, const int n, const int k,
        const T alpha,
        const T *a, const int lda,
        const T *b, const int ldb,
        const T beta,
        T *c, const int ldc,
        cublasStatus_t gemm_func(cublasHandle_t,
            cublasOperation_t transa, cublasOperation_t transb,
            int, int, int,
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_a, *gpu_b;
    T *gpu_c;
    int rows_a = 0, cols_a = 0,
        rows_b = 0, cols_b = 0,
        rows_c = 0, cols_c = 0;
    int size_a, size_b, size_c;
    cublasOperation_t opa = cu(transa);
    cublasOperation_t opb = cu(transb);
    const struct objinfo *a_info, *b_info, *c_info;

    size_a = compute_size(Layout, transa, a, lda, k, m);
    size_b = compute_size(Layout, transb, b, lda, k, n);
    size_c = (Layout == CblasColMajor) ?
        size(0, ldc, n, sizeof(*c)) :
        size(0, ldc, m, sizeof(*c));

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        b_info = NULL;
        c_info = NULL;

        assign_dims(Layout, transa, rows_a, cols_a, lda, m, k);
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);

        assign_dims(Layout, transa, rows_b, cols_b, ldb, k, n);
        gpu_b = transpose(b, size_b, &rows_b, &cols_b, ldb, geam_func);

        rows_c = ldc;
        cols_c = m;
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

    gemm_func(b2c_handle,
            opa, opb,
            m, n, k,
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


DECLARE_CBLAS__GEMM(s, float) {
    _cblas_gemm(Layout,
            transa,
            transb,
            m, n, k,
            alpha,
            a, lda,
            b, ldb,
            beta,
            c, ldc,
            &cublasSgemm,
            &cublasSgeam);
}

DECLARE_CBLAS__GEMM(d, double) {
    _cblas_gemm(Layout,
            transa,
            transb,
            m, n, k,
            alpha,
            a, lda,
            b, ldb,
            beta,
            c, ldc,
            &cublasDgemm,
            &cublasDgeam);
}

DECLARE_CBLAS__GEMM(c, float _Complex) {
    _cblas_gemm(Layout,
            transa,
            transb,
            m, n, k,
            cu(alpha),
            (cuComplex *)a, lda,
            (cuComplex *)b, ldb,
            cu(beta),
            (cuComplex *)c, ldc,
            &cublasCgemm,
            &cublasCgeam);
}

DECLARE_CBLAS__GEMM(z, double _Complex) {
    _cblas_gemm(Layout,
            transa,
            transb,
            m, n, k,
            cu(alpha),
            (cuDoubleComplex *)a, lda,
            (cuDoubleComplex *)b, ldb,
            cu(beta),
            (cuDoubleComplex *)c, ldc,
            &cublasZgemm,
            &cublasZgeam);
}
