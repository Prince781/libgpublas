#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

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
void _b2c_gemm(const CBLAS_TRANSPOSE transa,
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
            T *, int))
{
    const T *gpu_a, *gpu_b;
    T *gpu_c;
    int size_a, size_b, size_c;
    const struct objinfo *a_info, *b_info, *c_info;

    size_a = compute_size(CblasColMajor, transa, a, lda, k, m);
    size_b = compute_size(CblasColMajor, transb, b, ldb, n, k);
    size_c = size(0, ldc, n, sizeof(*c));

    gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    gpu_b = (T *) b2c_place_on_gpu((void *) b, size_b, &b_info, 
            (void *) gpu_a, &a_info,
            NULL);
    gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
            (void *) gpu_a, &a_info,
            (void *) gpu_b, &b_info,
            NULL);

    call_kernel(
        gemm_func(b2c_handle,
                cu(transa), cu(transb),
                m, n, k,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                &beta,
                gpu_c, ldc)
    );

    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!c_info) {
        b2c_copy_from_gpu(c, gpu_c, size_c);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_b, b_info);
    b2c_cleanup_gpu_ptr((void *) gpu_c, c_info);
}

// Fortran wrappers

F77_gemm(s, float) {
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
            &cublasSgemm);
}

F77_gemm(d, double) {
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
            &cublasDgemm);
}

F77_gemm(c, float _Complex) {
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            cu(*alpha),
            (cuComplex *)a, *lda,
            (cuComplex *)b, *ldb,
            cu(*beta),
            (cuComplex *)c, *ldc,
            &cublasCgemm);
}

F77_gemm(z, double _Complex) {
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            cu(*alpha),
            (cuDoubleComplex *)a, *lda,
            (cuDoubleComplex *)b, *ldb,
            cu(*beta),
            (cuDoubleComplex *)c, *ldc,
            &cublasZgemm);
}
