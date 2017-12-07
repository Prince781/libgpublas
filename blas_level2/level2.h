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


/* cblas_?ger - rank-1 update of a general matrix */
#define DECLARE_CBLAS__GER(prefix, type)                \
void cblas_##prefix##ger(const CBLAS_LAYOUT Layout,     \
        const int m, const int n,                       \
        const type alpha,                               \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *a, const int lda)

DECLARE_CBLAS__GER(s, float);
DECLARE_CBLAS__GER(d, double);

#define DECLARE_CBLAS__GERC(prefix, type)               \
void cblas_##prefix##gerc(const CBLAS_LAYOUT Layout,    \
        const int m, const int n,                       \
        const type *alpha,                              \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *a, const int lda)

DECLARE_CBLAS__GERC(c, float _Complex); 
DECLARE_CBLAS__GERC(z, double _Complex); 

#define DECLARE_CBLAS__GERU(prefix, type)               \
void cblas_##prefix##geru(const CBLAS_LAYOUT Layout,    \
        const int m, const int n,                       \
        const type *alpha,                              \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *a, const int lda)

DECLARE_CBLAS__GERU(c, float _Complex);
DECLARE_CBLAS__GERU(z, double _Complex);

/* ?hbmv - matrix-vector product using Hermitian band matrix */
#define DECLARE_CBLAS__HBMV(prefix, type)               \
void cblas_##prefix##hbmv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n, const int k,                       \
        const type *alpha,                              \
        const type *a, const int lda,                   \
        const type *x, const int incx,                  \
        const type *beta,                               \
        type *y, const int incy)

DECLARE_CBLAS__HBMV(c, float _Complex);
DECLARE_CBLAS__HBMV(z, double _Complex);

/* ?hemv - matrix-vector product using Hermitian matrix */

/* ?her - rank-1 update of Hermitian matrix */

/* ?her2 - rank-2 update of Hermitian matrix */

/* ?hpmv - matrix-vector product using Hermitian packed matrix */
#define DECLARE_CBLAS__HPMV(prefix, type)               \
void cblas_##prefix##hpmv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type *alpha,                              \
        const type *ap,                                 \
        const type *x, const int incx,                  \
        const type *beta,                               \
        type *y, const int incy)

DECLARE_CBLAS__HPMV(c, float _Complex);
DECLARE_CBLAS__HPMV(z, double _Complex);

/* ?hpr */

/* ?hpr2 */

/* ?sbmv */
#define DECLARE_CBLAS__SBMV(prefix, type)               \
void cblas_##prefix##sbmv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n, const int k,                       \
        const type alpha,                              \
        const type *a, const int lda,                   \
        const type *x, const int incx,                  \
        const type beta,                               \
        type *y, const int incy)

DECLARE_CBLAS__SBMV(s, float);
DECLARE_CBLAS__SBMV(d, double);

/* ?spmv */
#define DECLARE_CBLAS__SPMV(prefix, type)               \
void cblas_##prefix##spmv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *ap,                                 \
        const type *x, const int incx,                  \
        const type beta,                                \
        type *y, const int incy)

DECLARE_CBLAS__SPMV(s, float);
DECLARE_CBLAS__SPMV(d, double);

};

#endif
