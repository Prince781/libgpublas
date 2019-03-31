#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

template <typename T>
void _b2c_hemm(const CBLAS_SIDE side,
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
            T *, int))
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

    call_kernel(
        hemm_func(b2c_handle,
                cside, cuplo,
                m, n,
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


F77_hemm(c, float _Complex) {
    _b2c_hemm(c_side(*side), c_uplo(*uplo),
            *m, *n, 
            cu(*(typeof(*a) *)alpha),
            (cuComplex *)a, *lda,
            (cuComplex *)b, *ldb,
            cu(*(typeof(*a) *)beta),
            (cuComplex *)c, *ldc,
            &cublasChemm);
}

F77_hemm(z, double _Complex) {
    _b2c_hemm(c_side(*side), c_uplo(*uplo),
            *m, *n, 
            cu(*(typeof(*a) *)alpha),
            (cuDoubleComplex *)a, *lda,
            (cuDoubleComplex *)b, *ldb,
            cu(*(typeof(*a) *)beta),
            (cuDoubleComplex *)c, *ldc,
            &cublasZhemm);
}
