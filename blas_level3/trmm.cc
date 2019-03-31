#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

template <typename T>
void _b2c_trmm(const CBLAS_SIDE side,
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
            T *, int))
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

    call_kernel(
        trmm_func(b2c_handle,
                cside, cuplo,
                ctransa, cdiag,
                m, n,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                gpu_b, ldb)
    );

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!b_info) {
        b2c_copy_from_gpu(b, gpu_b, size_b);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_b, b_info);
}

F77_trmm(s, float) {
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            *alpha,
            a, *lda,
            b, *ldb,
            &cublasStrmm);
}

F77_trmm(d, double) {
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            *alpha,
            a, *lda,
            b, *ldb,
            &cublasDtrmm);
}

F77_trmm(c, float _Complex) {
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            cu(*alpha),
            (cuComplex *) a, *lda,
            (cuComplex *) b, *ldb,
            &cublasCtrmm);
}

F77_trmm(z, double _Complex) {
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            cu(*alpha),
            (cuDoubleComplex *) a, *lda,
            (cuDoubleComplex *) b, *ldb,
            &cublasZtrmm);
}
