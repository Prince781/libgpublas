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

#if USE_CUDA
extern cublasHandle_t b2c_cublas_handle;
#else
extern cl_command_queue opencl_cmd_queue;
#endif

template <typename T>
#if USE_CUDA
using syr2k_t = cublasStatus_t (*)(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *,
            T *, int);
#else
using syrk_t = clblasStatus (*)(clblasOrder order,
            clblasUplo uplo,
            clblasTranspose transAB,
            size_t N, size_t K,
            T alpha,
            const cl_mem A, size_t offA, size_t lda,
            T beta,
            cl_mem C, size_t offC, size_t ldc,
            cl_uint numCommandQueues, cl_command_queue *commandQueues, 
            cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events);
#endif

template <typename S, typename T>
void _b2c_syrk(const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const S alpha,
        const T *a, const int lda,
        const S beta,
        T *c, const int ldc,
        syrk_t<S> syrk_func)
{
    gpuptr<const T> gpu_a(a, size(0, lda, k, sizeof *a));
    gpuptr<T> gpu_c(c, size(0, ldc, n, sizeof *c));

    call_kernel(
#if USE_CUDA
        syrk_func(b2c_handle,
                cuplo, ctrans,
                n, k,
                &alpha,
                gpu_a, lda,
                &beta,
                gpu_c, ldc)
#else
        syrk_func(clblasColumnMajor,
            clb(uplo), clb(trans),
            n, k,
            alpha,
            gpu_a, 0, lda,
            beta,
            gpu_c, 0, ldc,
            1, &opencl_cmd_queue, 0, NULL, NULL)
#endif
    );
}

template <typename T>
bool syrk_check(const char *func_name,
                 char *uplo,
                 char *trans,
                 int *n, int *k,
                 T *alpha,
                 T *a, int *lda,
                 T *beta,
                 T *c, int *ldc) {
    int nrowa = -1;
    int info;
    int upper;

    if (runtime_blas_lsame(trans, "N"))
        nrowa = *n;
    else
        nrowa = *k;
    upper = runtime_blas_lsame(uplo, "U");
    info = 0;

    if (!upper && !runtime_blas_lsame(uplo, "L"))
        info = 1;
    else if (!runtime_blas_lsame(trans, "N") &&
             !runtime_blas_lsame(trans, "T") &&
             !runtime_blas_lsame(trans, "C"))
        info = 2;
    else if (*n < 0)
        info = 3;
    else if (*k < 0)
        info = 4;
    else if (*lda < std::max(1, nrowa))
        info = 7;
    else if (*ldc < std::max(1, *n))
        info = 10;
    
    if (info != 0) {
        runtime_blas_xerbla(func_name, info);
        return false;
    }

    // quick return if possible
    if (*n == 0 || ((*alpha == 0 || *k == 0) && *beta == 1))
        return false;
    
    if (*alpha == 0) {
        if (upper) {
            if (*beta == 0) {
                for (int j=1; j<=*n; j++)
                    for (int i=1; i<=j; i++)
                        c[IDX2F(i, j, *ldc)] = 0;
            } else {
                for (int i=1; i<=*n; i++)
                    for (int j=1; j<=i; j++)
                        c[IDX2F(i, j, *ldc)] *= *beta;
            }
        } else {
            if (*beta == 0) {
                for (int j=1; j<=*n; j++)
                    for (int i=j; i<=*n; i++)
                        c[IDX2F(i, j, *ldc)] = 0;
            } else {
                for (int j=1; j<=*n; j++)
                    for (int i=j; i<=*n; i++)
                        c[IDX2F(i, j, *ldc)] *= *beta;
            }
        }
    }
    return true;
}

F77_syrk(s, float) {
    if (!syrk_check(__func__, uplo, trans, n, k, alpha, a, lda, beta, c, ldc))
        return;
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            a, *lda,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasSsyrk
#else
            &clblasSsyrk
#endif
    );
}

F77_syrk(d, double) {
    if (!syrk_check(__func__, uplo, trans, n, k, alpha, a, lda, beta, c, ldc))
        return;
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            a, *lda,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasDsyrk
#else
            &clblasDsyrk
#endif
    );
}

F77_syrk(c, float _Complex) {
    if (!syrk_check(__func__, uplo, trans, n, k, alpha, a, lda, beta, c, ldc))
        return;
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu2(*alpha),
            cmplx_ptr(a), *lda,
            cu2(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasCsyrk
#else
            &clblasCsyrk
#endif
    );
}

F77_syrk(z, double _Complex) {
    if (!syrk_check(__func__, uplo, trans, n, k, alpha, a, lda, beta, c, ldc))
        return;
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu2(*alpha),
            cmplx_ptr(a), *lda,
            cu2(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasZsyrk
#else
            &clblasZsyrk
#endif
    );
}
