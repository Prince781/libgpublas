#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../runtime-blas.h"
#include "../runtime-mem.hpp"

#if USE_CUDA
extern cublasHandle_t b2c_cublas_handle;
#else
extern cl_command_queue opencl_cmd_queue;
#endif

template <typename T, typename S>
#if USE_CUDA
using syr2k_t = cublasStatus_t (*)(cublasHandle_t,
            cublasFillMode_t,
            cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int);
#else
using syr2k_t = clblasStatus (*)(clblasOrder order, 
        clblasUplo uplo, clblasTranspose transAB, 
        size_t N, size_t K, 
        S alpha, 
        const cl_mem A, size_t offA, size_t lda, 
        const cl_mem B, size_t offB, size_t ldb, 
        S beta, 
        cl_mem C, size_t offC, size_t ldc, 
        cl_uint numCommandQueues, cl_command_queue *commandQueues, 
        cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events);
#endif


template <typename T, typename S>
void _b2c_syr2k(const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const S alpha,
        const T *a, const int lda,
        const T *b, const int ldb,
        const S beta,
        T *c, const int ldc,
        syr2k_t<T,S> syr2k_func)
{
    gpuptr<const T> gpu_a(a, size(0, lda, trans == CblasNoTrans ? k : n, sizeof *a));
    gpuptr<const T> gpu_b(b, size(0, ldb, trans == CblasNoTrans ? k : n, sizeof *b));
    gpuptr<T> gpu_c(c, size(0, ldc, n, sizeof *c));

    call_kernel(
#if USE_CUDA
        syr2k_func(b2c_cublas_handle,
                cu(uplo), cu(trans),
                n, k,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                &beta,
                gpu_c, ldc)
#else
        syr2k_func(clblasColumnMajor, 
            clb(uplo), clb(trans),
            n, k,
            alpha,
            gpu_a, 0, lda,
            gpu_b, 0, ldb,
            beta,
            gpu_c, 0, ldc,
            1, &opencl_cmd_queue, 0, NULL, NULL)
#endif
    );
}

template <typename T, bool is_complex = std::is_same<T, float _Complex>::value || std::is_same<T, double _Complex>::value>
bool syr2k_check(const char *func_name,
                 char *uplo, 
                 char *trans,
                 int *n, int *k,
                 T *alpha,
                 T *a, int *lda,
                 T *b, int *ldb,
                 T *beta,
                 T *c, int *ldc) {
    int nrowa = runtime_blas_lsame(trans, "N") ? *n : *k;
    int upper = runtime_blas_lsame(uplo, "U");
    int info = 0;

    if (!upper && !runtime_blas_lsame(uplo, "L"))
        info = 1;
    else if (!runtime_blas_lsame(trans, "N") && 
             !runtime_blas_lsame(trans, "T") && 
             (is_complex || !runtime_blas_lsame(trans, "C")))
        info = 2;
    else if (*n < 0)
        info = 3;
    else if (*k < 0)
        info = 4;
    else if (*lda < std::max(1, nrowa))
        info = 7;
    else if (*ldb < std::max(1, nrowa))
        info = 9;
    else if (*ldc < std::max(1, *n))
        info = 12;

    if (info != 0) {
        runtime_blas_xerbla(func_name, info);
        return false;
    }

    if (*n == 0 || ((*alpha == 0 || *k == 0) && *beta == 1))
        return false;

    if (*alpha == 0 || *n == 0 || *k == 0) {
        if (upper) {
            if (*beta == 0) {
                for (int j=1; j<=*n; ++j)
                    for (int i=1; i<=j; ++i)
                        c[IDX2F(i,j,*ldc)] = 0;
            } else {
                for (int j=1; j<=*n; ++j)
                    for (int i=1; i<=j; ++i)
                        c[IDX2F(i,j,*ldc)] *= *beta;
            }
        } else {
            if (*beta == 0) {
                for (int j=1; j<=*n; ++j)
                    for (int i=j; i<=*n; ++i)
                        c[IDX2F(i,j,*ldc)] = 0;
            } else {
                for (int j=1; j<=*n; ++j)
                    for (int i=j; i<=*n; ++i)
                        c[IDX2F(i,j,*ldc)] *= *beta;
            }
        }
        return false;
    }

    return true;
}

F77_syr2k(s, float) {
    if (!syr2k_check(__func__, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
        return;
    _b2c_syr2k(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasSsyr2k
#else
            &clblasSsyr2k
#endif
            );
}

F77_syr2k(d, double) {
    if (!syr2k_check(__func__, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
        return;
    _b2c_syr2k(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasDsyr2k
#else
            &clblasDsyr2k
#endif
            );
}

F77_syr2k(c, float _Complex) {
    if (!syr2k_check(__func__, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
        return;
    _b2c_syr2k(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
            cu(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasCsyr2k
#else
            &clblasCsyr2k
#endif
            );
}

F77_syr2k(z, double _Complex) {
    if (!syr2k_check(__func__, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
        return;
    _b2c_syr2k(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
            cu(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasZsyr2k
#else
            &clblasZsyr2k
#endif
            );
}
