#if USE_CUDA
#include <cublas_v2.h>
#endif
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
using symm_t = cublasStatus_t symm_func(cublasHandle_t,
            cublasSideMode_t,
            cublasFillMode_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int);
#else
using symm_t = clblasStatus (*)(clblasOrder order, 
        clblasSide side, 
        clblasUplo uplo, 
        size_t M, size_t N, 
        S alpha, 
        const cl_mem A, size_t offa, size_t lda, 
        const cl_mem B, size_t offb, size_t ldb, 
        S beta, 
        cl_mem C, size_t offc, size_t ldc, 
        cl_uint numCommandQueues, cl_command_queue *commandQueues, 
        cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events);
#endif

template <typename T, typename S>
void _b2c_symm(const CBLAS_SIDE side,
        const CBLAS_UPLO uplo,
        const int m, const int n,
        const S alpha,
        const T *a, const int lda,
        const T *b, const int ldb,
        const S beta,
        T *c, const int ldc,
        symm_t<T,S> symm_func)
{
    gpuptr<const T> gpu_a(a, size(0, lda, side == CblasLeft ? m : n, sizeof *a));
    gpuptr<const T> gpu_b(b, size(0, ldb, n, sizeof *b));
    gpuptr<T> gpu_c(c, size(0, ldc, n, sizeof *c));

    call_kernel(
#if USE_CUDA
        symm_func(b2c_cublas_handle,
                cu(side), cu(uplo),
                m, n,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                &beta,
                gpu_c, ldc)
#else
        symm_func(clblasColumnMajor, clb(side), clb(uplo),
            m, n,
            alpha,
            gpu_a, 0, lda,
            gpu_b, 0, ldb,
            beta,
            gpu_c, 0, ldc,
            1, &opencl_cmd_queue, 0, NULL, NULL)
#endif
    );
}

#define symm_check()\
do {\
    int nrowa = runtime_blas_lsame(side, "L") ? *m : *n;\
    int upper = runtime_blas_lsame(uplo, "U");\
    int info = 0;\
    \
    if (!runtime_blas_lsame(side, "L") && !runtime_blas_lsame(side, "R"))\
        info = 1;\
    else if (!upper && !runtime_blas_lsame(uplo, "L"))\
        info = 2;\
    else if (*m < 0)\
        info = 3;\
    else if (*n < 0)\
        info = 4;\
    else if (*lda < std::max(1, nrowa))\
        info = 7;\
    else if (*ldb < std::max(1, *m))\
        info = 9;\
    else if (*ldc < std::max(1, *m))\
        info = 12;\
    \
    if (info != 0) {\
        runtime_blas_xerbla(__func__, info);\
        return;\
    }\
    \
    if (*m == 0 || *n == 0 || (*alpha == 0 && *beta == 1))\
        return;\
    \
    if (*alpha == 0) {\
        if (*beta == 0) {\
            for (int j=1; j<=*n; ++j)\
                for (int i=1; i<=*m; ++i)\
                    c[IDX2F(i,j,*ldc)] = 0;\
        } else {\
            for (int j=1; j<=*n; ++j)\
                for (int i=1; i<=*m; ++i)\
                    c[IDX2F(i,j,*ldc)] *= *beta;\
        }\
        return;\
    }\
} while (0)

F77_symm(s, float) {
    symm_check();
    _b2c_symm(c_side(*side), c_uplo(*uplo),
            *m, *n, 
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasSsymm
#else
            &clblasSsymm
#endif
            );
}

F77_symm(d, double) {
    symm_check();
    _b2c_symm(c_side(*side), c_uplo(*uplo),
            *m, *n, 
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasDsymm
#else
            &clblasDsymm
#endif
            );
}

F77_symm(c, float _Complex) {
    symm_check();
    _b2c_symm(c_side(*side), c_uplo(*uplo),
            *m, *n, 
#if USE_CUDA
            cu(*alpha),
#else
            cu2(*alpha),
#endif
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
#if USE_CUDA
            cu(*beta),
#else
            cu2(*beta),
#endif
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasCsymm
#else
            &clblasCsymm
#endif
            );
}

F77_symm(z, double _Complex) {
    symm_check();
    _b2c_symm(c_side(*side), c_uplo(*uplo),
            *m, *n, 
            cu(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
            cu(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasZsymm
#else
            &clblasZsymm
#endif
            );
}
