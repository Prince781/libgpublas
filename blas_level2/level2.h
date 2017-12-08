#ifndef BLAS2CUDA_LEVEL2_H
#define BLAS2CUDA_LEVEL2_H

#include <complex.h>
#include "../blas2cuda.h"
#include "../cblas.h"
#include "../conversions.h"

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
#define DECLARE_CBLAS__HEMV(prefix, type)               \
void cblas_##prefix##hemv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type *alpha,                              \
        const type *a, const int lda,                   \
        const type *x, const int incx,                  \
        const type *beta,                               \
        type *y, const int incy)

DECLARE_CBLAS__HEMV(c, float _Complex);
DECLARE_CBLAS__HEMV(z, double _Complex);

/* ?her - rank-1 update of Hermitian matrix */
#define DECLARE_CBLAS__HER(prefix, rtype, type)         \
void cblas_##prefix##her(const CBLAS_LAYOUT Layout,     \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const rtype alpha,                              \
        const type *x, const int incx,                  \
        type *a, const int lda)

DECLARE_CBLAS__HER(c, float, float _Complex);
DECLARE_CBLAS__HER(z, double, double _Complex);

/* ?her2 - rank-2 update of Hermitian matrix */
#define DECLARE_CBLAS__HER2(prefix, type)               \
void cblas_##prefix##her2(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *a, const int lda)

DECLARE_CBLAS__HER2(c, float _Complex);
DECLARE_CBLAS__HER2(z, double _Complex);

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

/* ?hpr - rank-1 update of a Hermitian packed matrix */
#define DECLARE_CBLAS__HPR(prefix, stype, type)         \
void cblas_##prefix##hpr(const CBLAS_LAYOUT Layout,     \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const stype alpha,                              \
        const type *x, const int incx,                  \
        type *ap)

DECLARE_CBLAS__HPR(c, float, float _Complex);
DECLARE_CBLAS__HPR(z, double, double _Complex);

/* ?hpr2 - rank-2 update of a Hermitian packed matrix */
#define DECLARE_CBLAS__HPR2(prefix, type)               \
void cblas_##prefix##hpr2(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *ap)

DECLARE_CBLAS__HPR2(c, float _Complex);
DECLARE_CBLAS__HPR2(z, double _Complex);

/* ?sbmv - matrix-vector product using a symmetric band matrix */
#define DECLARE_CBLAS__SBMV(prefix, type)               \
void cblas_##prefix##sbmv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n, const int k,                       \
        const type alpha,                               \
        const type *a, const int lda,                   \
        const type *x, const int incx,                  \
        const type beta,                                \
        type *y, const int incy)

DECLARE_CBLAS__SBMV(s, float);
DECLARE_CBLAS__SBMV(d, double);

/* ?spmv - matrix-vector product using a symmetric packed matrix */
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

/* ?spr - rank-1 update of a symmetric packed matrix */
#define DECLARE_CBLAS__SPR(prefix, type)                \
void cblas_##prefix##spr(const CBLAS_LAYOUT Layout,     \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *x, const int incx,                  \
        type *ap)

DECLARE_CBLAS__SPR(s, float);
DECLARE_CBLAS__SPR(d, double);

/* ?spr2 - rank-2 update of a symmetric packed matrix */
#define DECLARE_CBLAS__SPR2(prefix, type)               \
void cblas_##prefix##spr2(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *ap)

DECLARE_CBLAS__SPR2(s, float);
DECLARE_CBLAS__SPR2(d, double);

/* ?symv - matrix-vector product for symmetric matrix */
#define DECLARE_CBLAS__SYMV(prefix, type)               \
void cblas_##prefix##symv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *a, const int lda,                   \
        const type *x, const int incx,                  \
        const type beta,                                \
        type *y, const int incy)


DECLARE_CBLAS__SYMV(s, float);
DECLARE_CBLAS__SYMV(d, double);

/* ?syr - rank-1 update of a symmetric matrix */
#define DECLARE_CBLAS__SYR(prefix, type)                \
void cblas_##prefix##syr (const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *x, const int incx,                  \
        type *a, const int lda)


DECLARE_CBLAS__SYR(s, float);
DECLARE_CBLAS__SYR(d, double);

/* ?syr2 - rank-2 update of symmetric matrix */
#define DECLARE_CBLAS__SYR2(prefix, type)               \
void cblas_##prefix##syr2 (const CBLAS_LAYOUT Layout,   \
        const CBLAS_UPLO uplo,                          \
        const int n,                                    \
        const type alpha,                               \
        const type *x, const int incx,                  \
        const type *y, const int incy,                  \
        type *a, const int lda)

DECLARE_CBLAS__SYR2(s, float);
DECLARE_CBLAS__SYR2(d, double);

/* ?tbmv - matrix-vector product using a triangular band matrix */
#define DECLARE_CBLAS__TBMV(prefix, type)               \
void cblas_##prefix##tbmv (const CBLAS_LAYOUT Layout,   \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const CBLAS_DIAG diag,                          \
        const int n, const int k,                       \
        const type *a, const int lda,                   \
        type *x, const int incx)

DECLARE_CBLAS__TBMV(s, float);
DECLARE_CBLAS__TBMV(d, double);
DECLARE_CBLAS__TBMV(c, float _Complex);
DECLARE_CBLAS__TBMV(z, double _Complex);

/* ?tbsv - solve a system of linear equations whose coefficients are in a
 * triangular band matrix */
#define DECLARE_CBLAS__TBSV(prefix, type)               \
void cblas_##prefix##tbsv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const CBLAS_DIAG diag,                          \
        const int n, const int k,                       \
        const type *a, const int lda,                   \
        type *x, const int incx)

DECLARE_CBLAS__TBSV(s, float);
DECLARE_CBLAS__TBSV(d, double);
DECLARE_CBLAS__TBSV(c, float _Complex);
DECLARE_CBLAS__TBSV(z, double _Complex);

/* ?tpmv - matrix-vector product using a triangular band matrix */
#define DECLARE_CBLAS__TPMV(prefix, type)               \
void cblas_##prefix##tpmv (const CBLAS_LAYOUT Layout,   \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const CBLAS_DIAG diag,                          \
        const int n, const int k,                       \
        const type *ap,                                 \
        type *x, const int incx)

DECLARE_CBLAS__TPMV(s, float);
DECLARE_CBLAS__TPMV(d, double);
DECLARE_CBLAS__TPMV(c, float _Complex);
DECLARE_CBLAS__TPMV(z, double _Complex);

/* ?tpsv - solves a system of linear equations whose coefficients are in a
 * triangular packed matrix */
#define DECLARE_CBLAS__TPSV(prefix, type)               \
void cblas_##prefix##tpsv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const CBLAS_DIAG diag,                          \
        const int n,                                    \
        const type *ap,                                 \
        type *x, const int incx)

DECLARE_CBLAS__TPSV(s, float);
DECLARE_CBLAS__TPSV(d, double);
DECLARE_CBLAS__TPSV(c, float _Complex);
DECLARE_CBLAS__TPSV(z, double _Complex);

/* ?trmv - compute  a matrix-vector product using a triangular matrix */
#define DECLARE_CBLAS__TRMV(prefix, type)               \
void cblas_##prefix##trmv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const CBLAS_DIAG diag,                          \
        const int n,                                    \
        const type *a, const int lda,                   \
        type *x, const int incx)

DECLARE_CBLAS__TRMV(s, float);
DECLARE_CBLAS__TRMV(d, double);
DECLARE_CBLAS__TRMV(c, float _Complex);
DECLARE_CBLAS__TRMV(z, double _Complex);

/* ?trsv - solve a system of linear equations whose coefficients are in a
 * triangular matrix */
#define DECLARE_CBLAS__TRSV(prefix, type)               \
void cblas_##prefix##trsv(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const CBLAS_DIAG diag,                          \
        const int n,                                    \
        const type *a, const int lda,                   \
        type *x, const int incx)

DECLARE_CBLAS__TRSV(s, float);
DECLARE_CBLAS__TRSV(d, double);
DECLARE_CBLAS__TRSV(c, float _Complex);
DECLARE_CBLAS__TRSV(z, double _Complex);

};

#endif
