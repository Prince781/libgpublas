#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../runtime-blas.h"
#include "../runtime-mem.hpp"

#if USE_CUDA
extern cublasHandle_t b2c_cublas_handle;

template <typename T>
using trmm_t = cublasStatus_t (*)(cublasHandle_t,
            cublasSideMode_t,
            cublasFillMode_t,
            cublasOperation_t,
            cublasDiagType_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            T *, int);
#else
extern cl_command_queue opencl_cmd_queue;

template <typename T>
using trmm_t = clblasStatus (*)(clblasOrder order, 
                                clblasSide side,
                                clblasUplo uplo, 
                                clblasTranspose transA, 
                                clblasDiag diag, 
                                size_t M, size_t N,
                                T alpha, 
                                const cl_mem A, size_t offA, size_t lda, 
                                cl_mem B, size_t offB, size_t ldb, 
                                cl_uint numCommandQueues, cl_command_queue *commandQueues, 
                                cl_uint numEventsInWaitList, const cl_event *eventWaitList, 
                                cl_event *events);
#endif


template <typename S, typename T>
void _b2c_trmm(const CBLAS_SIDE side,
        const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_DIAG diag,
        const int m, const int n,
        const S alpha,
        const T *a, const int lda,
        T *b, const int ldb,
        trmm_t<S> trmm_func)
{
    gpuptr<const T> gpu_a(a, size(0, lda, m, sizeof *a));
    gpuptr<T> gpu_b(b, size(0, ldb, m, sizeof *b));

    call_kernel(
#if USE_CUDA
        trmm_func(b2c_handle,
                cu(side), cu(uplo),
                cu(transa), cu(diag),
                m, n,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                gpu_b, ldb)
#else
        trmm_func(clblasColumnMajor,
                  clb(side), clb(uplo),
                  clb(transa), clb(diag),
                  m, n,
                  alpha,
                  gpu_a, 0, lda,
                  gpu_b, 0, ldb,
                  1, &opencl_cmd_queue,
                  0, NULL,
                  NULL)
#endif
    );
}

template <typename T>
bool trmm_check(const char *func_name,
                char *side, char *uplo, char *transa, char *diag,
                int *m, int *n,
                T *alpha,
                T *a, int *lda,
                T *b, int *ldb) {
    int lside = runtime_blas_lsame(side, "L");
    int nrowa;
    int upper = runtime_blas_lsame(uplo, "U");
    int info;

    if (lside)
        nrowa = *m;
    else
        nrowa = *n;
    
    info = 0;

    if (!lside && !runtime_blas_lsame(side, "R"))
        info = 1;
    else if (!upper && !runtime_blas_lsame(uplo, "L"))
        info = 2;
    else if (!runtime_blas_lsame(transa, "N") &&
             !runtime_blas_lsame(transa, "T") &&
             !runtime_blas_lsame(transa, "C"))
        info = 3;
    else if (!runtime_blas_lsame(diag, "U") && !runtime_blas_lsame(diag, "N"))
        info = 4;
    else if (*m < 0)
        info = 5;
    else if (*n < 0)
        info = 6;
    else if (*lda < std::max(1, nrowa))
        info = 9;
    else if (*ldb < std::max(1, *m))
        info = 11;
    
    if (info != 0) {
        runtime_blas_xerbla(func_name, info);
        return false;
    }

    // quick return if possible
    if (*m == 0 || *n == 0)
        return false;
    
    // and when alpha == 0
    if (*alpha == 0) {
        for (int j=1; j<=*n; j++)
            for (int i=1; i<=*m; i++)
                b[IDX2F(i, j, *ldb)] = 0;
        return false;
    }

    return true;
}

F77_trmm(s, float) {
    if (!trmm_check(__func__, 
                    side, uplo, transa, diag, 
                    m, n, alpha, a, lda, b, ldb))
        return;
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            *alpha,
            a, *lda,
            b, *ldb,
#if USE_CUDA
            &cublasStrmm
#else
            &clblasStrmm
#endif
    );
}

F77_trmm(d, double) {
    if (!trmm_check(__func__, 
                    side, uplo, transa, diag, 
                    m, n, alpha, a, lda, b, ldb))
        return;
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            *alpha,
            a, *lda,
            b, *ldb,
#if USE_CUDA
            &cublasDtrmm
#else
            &clblasDtrmm
#endif
    );
}

F77_trmm(c, float _Complex) {
    if (!trmm_check(__func__, 
                    side, uplo, transa, diag, 
                    m, n, alpha, a, lda, b, ldb))
        return;
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            cu2(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
#if USE_CUDA
            &cublasCtrmm
#else
            &clblasCtrmm
#endif
    );
}

F77_trmm(z, double _Complex) {
    if (!trmm_check(__func__, 
                    side, uplo, transa, diag, 
                    m, n, alpha, a, lda, b, ldb))
        return;
    _b2c_trmm(c_side(*side), c_uplo(*uplo), 
            c_trans(*transa), c_diag(*diag),
            *m, *n,
            cu2(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
#if USE_CUDA
            &cublasZtrmm
#else
            &clblasZtrmm
#endif
    );
}
