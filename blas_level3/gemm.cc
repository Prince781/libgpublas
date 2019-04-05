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
using gemm_t = cublasStatus_t (*)(cublasHandle_t,
            cublasOperation_t transa, cublasOperation_t transb,
            int, int, int,
            const S *,
            const T *, int,
            const T *, int,
            const S *,
            T *, int);
#else
using gemm_t = clblasStatus (*)(clblasOrder order, clblasTranspose transA, clblasTranspose transB,
                size_t M, size_t N, size_t K, S alpha, const cl_mem A, size_t offA, size_t lda,
                const cl_mem B, size_t offB, size_t ldb, S beta, cl_mem C, size_t offC, size_t ldc,
                cl_uint numCommandQueues, cl_command_queue* commandQueues, cl_uint numEventsInWaitList,
                const cl_event* eventWaitList, cl_event* events);
#endif

template <typename T>
static inline int compute_size(const CBLAS_TRANSPOSE transa,
        const T *a, const int lda, const int dim1, const int dim2)
{
    if (transa == CblasNoTrans)
        return size(0, lda, dim1, sizeof *a);

    return size(0, lda, dim2, sizeof(*a));
}

template <typename T, typename S>
void _b2c_gemm(const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb,
        const int m, const int n, const int k,
        const S alpha,
        const T *a, const int lda,
        const T *b, const int ldb,
        const S beta,
        T *c, const int ldc,
        gemm_t<T,S> gemm_func)
{
    gpuptr<const T> gpu_a(a, compute_size(transa, a, lda, k, m));
    gpuptr<const T> gpu_b(b, compute_size(transb, b, ldb, n, k));
    gpuptr<T> gpu_c(c, size(0, ldc, n, sizeof(*c)));


    call_kernel(
#if USE_CUDA
        gemm_func(b2c_cublas_handle,
                cu(transa), cu(transb),
                m, n, k,
                &alpha,
                gpu_a, lda,
                gpu_b, ldb,
                &beta,
                gpu_c, ldc)
#else
        gemm_func(clblasColumnMajor, clb(transa), clb(transb),
            m, n, k,
            alpha,
            gpu_a, 0, lda,
            gpu_b, 0, ldb,
            beta,
            gpu_c, 0, ldc,
            1, &opencl_cmd_queue, 0, NULL, NULL)
#endif
    );
}

// Fortran wrappers

#define gemm_check() \
do {\
    bool nota = runtime_blas_lsame(transa, "N");\
    bool notb = runtime_blas_lsame(transb, "N");\
    int info = 0;\
\
    int nrowa = nota ? *m : *k;\
    int nrowb = notb ? *k : *n;\
\
    if (!nota && !runtime_blas_lsame(transa, "C") && !runtime_blas_lsame(transa, "T"))\
        info = 1;\
    else if (!notb && !runtime_blas_lsame(transb, "C") && !runtime_blas_lsame(transb, "T"))\
        info = 2;\
    else if (*m < 0)\
        info = 3;\
    else if (*n < 0)\
        info = 4;\
    else if (*k < 0)\
        info = 5;\
    else if (*lda < std::max(1,nrowa))\
        info = 8;\
    else if (*ldb < std::max(1,nrowb))\
        info = 10;\
    else if (*ldc < std::max(1,*m))\
        info = 13;\
\
    if (info != 0){ \
        runtime_blas_xerbla(__func__, info);\
        return;\
    }\
\
    if (*m == 0 || *n == 0 || ((*alpha == 0 || *k == 0) && *beta == 1))\
        return;\
\
    if (*alpha == 0 || *k == 0) {\
        for (int j=1; j<=*n; ++j)\
            for (int i=1; i<=*m; ++i)\
                c[IDX2F(i,j,*ldc)] *= *beta;\
        return;\
    }\
} while (0)

#define gemm_perf_check(fname)\
do {\
    typeof(fname) *f = (typeof(fname) *) runtime_blas_func(__func__);\
\
    if (*m == 0 || *n == 0 || *k == 0 || *lda < 512 || *ldb < 512 || *ldc < 512)\
        (*f)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);\
} while (0)\

F77_gemm(s, float) {
    gemm_check();
    gemm_perf_check(sgemm_);
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasSgemm
#else
            &clblasSgemm
#endif
            );
}

F77_gemm(d, double) {
    gemm_check();
    gemm_perf_check(dgemm_);
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            *alpha,
            a, *lda,
            b, *ldb,
            *beta,
            c, *ldc,
#if USE_CUDA
            &cublasDgemm
#else
            &clblasDgemm
#endif
            );
}

F77_gemm(c, float _Complex) {
    gemm_check();
    gemm_perf_check(cgemm_);
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            cu(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
            cu(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasCgemm
#else
            &clblasCgemm
#endif
            );
}

F77_gemm(z, double _Complex) {
    gemm_check();
    gemm_perf_check(zgemm_);
    _b2c_gemm(c_trans(*transa),
            c_trans(*transb),
            *m, *n, *k,
            cu(*alpha),
            cmplx_ptr(a), *lda,
            cmplx_ptr(b), *ldb,
            cu(*beta),
            cmplx_ptr(c), *ldc,
#if USE_CUDA
            &cublasZgemm
#else
            &clblasZgemm
#endif
            );
}
