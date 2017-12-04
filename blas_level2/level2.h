#ifndef BLAS2CUDA_LEVEL2_H
#define BLAS2CUDA_LEVEL2_H

#include <complex.h>
#include "../blas2cuda.h"
#include "../cblas.h"

template <typename T>
using geam_t = cublasStatus_t (*)(cublasHandle_t,
            cublasOperation_t, cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *,
            const T *, int,
            T *, int);

extern "C" {

/* BLAS Level 2 routines */

/* cblas_?gbmv - Matrix-vector product using a general band matrix */
#define DECLARE_CBLAS__GBMV(prefix, type)                \
void cblas_##prefix##gbmv (const CBLAS_LAYOUT Layout,   \
        CBLAS_TRANSPOSE trans,                          \
        const int m, const int n,                       \
        const int kl, const int ku,                     \
        const type alpha,                               \
        const type *A, const int lda,                   \
        const type *x, const int incx,                  \
        const type beta,                                \
        type *y, const int incy)

DECLARE_CBLAS__GBMV(s, float);
DECLARE_CBLAS__GBMV(d, double);
DECLARE_CBLAS__GBMV(c, float _Complex);
DECLARE_CBLAS__GBMV(z, double _Complex);

/* cblas_?gemv - matrix-vector product using a general matrix */
#define DECLARE_CBLAS__GEMV(prefix, type)               \
void cblas_##prefix##gemv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_TRANSPOSE trans,                    \
        const int m, const int n,                       \
        const type alpha,                               \
        const type *A, const int lda,                   \
        const type *x, const int incx,                  \
        const type beta,                                \
        type *y, const int incy)

DECLARE_CBLAS__GEMV(s, float);
DECLARE_CBLAS__GEMV(d, double);
DECLARE_CBLAS__GEMV(c, float _Complex);
DECLARE_CBLAS__GEMV(z, double _Complex);

};

#endif