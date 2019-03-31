#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

template <typename S, typename T>
void _b2c_her2k(const CBLAS_UPLO uplo,
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
            T *, int))
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
        rows_a = lda;
        cols_a = k;
        rows_b = ldb;
        cols_b = k;
    } else {
        rows_a = lda;
        cols_a = n;
        rows_b = ldb;
        cols_b = n;
    }


    size_a = size(0, rows_a, cols_a, sizeof(*a));
    size_b = size(0, rows_b, cols_b, sizeof(*b));
    size_c = size(0, rows_c, cols_c, sizeof(*c));


    gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    gpu_b = (T *) b2c_place_on_gpu((void *) b, size_b, &b_info, 
            (void *) gpu_a, &a_info,
            NULL);
    gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
            (void *) gpu_a, &a_info,
            (void *) gpu_b, &b_info,
            NULL);

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

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!c_info) {
        b2c_copy_from_gpu(c, gpu_c, size_c);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_b, b_info);
    b2c_cleanup_gpu_ptr((void *) gpu_c, c_info);
}

F77_her2k(c, float, float _Complex) {
    _b2c_her2k(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu(*alpha),
            (cuComplex *) a, *lda,
            (cuComplex *) b, *ldb,
            *beta,
            (cuComplex *) c, *ldc,
            &cublasCher2k);
}

F77_her2k(z, double, double _Complex) {
    _b2c_her2k(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu(*alpha),
            (cuDoubleComplex *) a, *lda,
            (cuDoubleComplex *) b, *ldb,
            *beta,
            (cuDoubleComplex *) c, *ldc,
            &cublasZher2k);
}
