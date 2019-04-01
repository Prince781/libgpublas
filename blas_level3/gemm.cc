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

template <typename T>
#if USE_CUDA
using gemm_t = cublasStatus_t (*)(cublasHandle_t,
            cublasOperation_t transa, cublasOperation_t transb,
            int, int, int,
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int);
#else
using gemm_t = CLBlastStatusCode (*)(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const T alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const T beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event);
#endif

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
        gemm_t<T> gemm_func)
{
    gpuptr<const T> gpu_a(a, compute_size(CblasColMajor, transa, a, lda, k, m));
    gpuptr<const T> gpu_b(b, compute_size(CblasColMajor, transb, b, ldb, n, k));
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
        gemm_func(CLBlastLayoutColMajor, clb(transa), clb(transb),
            m, n, k,
            alpha,
            gpu_a, 0, lda,
            gpu_b, 0, ldb,
            beta,
            gpu_c, 0, ldc,
            &opencl_cmd_queue, NULL)
#endif
    );
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
#if USE_CUDA
            &cublasSgemm
#else
            &CLBlastSgemm
#endif
            );
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
#if USE_CUDA
            &cublasDgemm
#else
            &CLBlastDgemm
#endif
            );
}

/*
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
*/
