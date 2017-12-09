#ifndef CBLAS_H
#define CBLAS_H

#include <stdlib.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define size(amt, n,stride,sz) (((amt) + (n) * abs(stride)) * (sz))

/* taken from cblas.h */

/*
 * Enumerated and derived types
 */
#ifdef WeirdNEC
   #define CBLAS_INDEX long
#else
    #define CBLAS_INDEX int
#endif

typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */

static inline int get_lda(CBLAS_LAYOUT layout, int rows, int cols) {
    if (layout == CblasRowMajor)
        return rows;
    return cols;
}

#ifdef __cplusplus
extern "C" {
#endif

/* BLAS Level 1 routines */

/* cblas_?asum - sum of vector magnitudes (functions) */
#define DECLARE_CBLAS__ASUM(prefix, type)    \
type cblas_##prefix##asum (const int n, const type *x, const int incx)

DECLARE_CBLAS__ASUM(s, float);
DECLARE_CBLAS__ASUM(sc, float _Complex);
DECLARE_CBLAS__ASUM(d, double);
DECLARE_CBLAS__ASUM(dz, double _Complex);

/* cblas_?axpy - scalar-vector product (routines) */
#define DECLARE_CBLAS__AXPY(prefix, type)   \
void cblas_##prefix##axpy (const int n,     \
        const type a,                       \
        const type *x,                      \
        const int incx,                     \
        type *y,                            \
        const int incy)

DECLARE_CBLAS__AXPY(s, float);
DECLARE_CBLAS__AXPY(d, double);
DECLARE_CBLAS__AXPY(c, float _Complex);
DECLARE_CBLAS__AXPY(z, double _Complex);

/* cblas_?copy - copy vector (routines) */
#define DECLARE_CBLAS__COPY(prefix, type)   \
void cblas_##prefix##copy (const int n,     \
        const type *x,                      \
        const int incx,                     \
        type *y,                            \
        const int incy)

DECLARE_CBLAS__COPY(s, float);
DECLARE_CBLAS__COPY(d, double);
DECLARE_CBLAS__COPY(c, float _Complex);
DECLARE_CBLAS__COPY(z, double _Complex);

/* cblas_?dot - vector-vector dot product */
#define DECLARE_CBLAS__DOT(prefix, type)    \
type cblas_##prefix##dot (const int n,      \
        const type *x,                      \
        const int incx,                     \
        const type *y,                      \
        const int incy)

DECLARE_CBLAS__DOT(s, float);
DECLARE_CBLAS__DOT(d, double);

/* cblas_?sdot - vector-vector dot product with double precision */
/* (not implemented by cuBLAS)
float cblas_sdsdot (const int n, 
        const float sb,
        const float *sx,
        const int incx,
        const float *sy,
        const int incy);

double cblas_dsdot (const int n,
        const float *sx,
        const int incx,
        const float *sy,
        const int incy);
*/

/* cblas_?dotc - dot product of a conjugated vector with another vector */
#define DECLARE_CBLAS__DOTC(prefix, type)   \
void cblas_##prefix##dotc_sub (const int n, \
        const type *x,                      \
        const int incx,                     \
        const type *y,                      \
        const int incy,                     \
        type *dotc)

DECLARE_CBLAS__DOTC(c, float _Complex);
DECLARE_CBLAS__DOTC(z, double _Complex);

/* cblas_?dotu - vector-vector dot product */
#define DECLARE_CBLAS__DOTU(prefix, type)   \
void cblas_##prefix##dotu_sub (const int n, \
        const type *x,                      \
        const int incx,                     \
        const type *y,                      \
        const int incy,                     \
        type *dotu)

DECLARE_CBLAS__DOTU(c, float _Complex);
DECLARE_CBLAS__DOTU(z, double _Complex);

/* cblas_?nrm2 - Euclidean norm of a vector */
#define DECLARE_CBLAS__NRM2(prefix, type)   \
type cblas_##prefix##_nrm2 (const int n,    \
        const type *x,                      \
        const int incx)

DECLARE_CBLAS__NRM2(s, float);
DECLARE_CBLAS__NRM2(d, double);
DECLARE_CBLAS__NRM2(sc, float _Complex);
DECLARE_CBLAS__NRM2(dz, double _Complex);

/* cblas_?rot - performs rotation of points in the plane */
#define DECLARE_CBLAS__ROT(prefix, type, type2)    \
void cblas_##prefix##rot (const int n,              \
        type *x,                                    \
        const int incx,                             \
        type *y,                                    \
        const int incy,                             \
        const type2 c,                              \
        const type2 s)

DECLARE_CBLAS__ROT(s, float, float);
DECLARE_CBLAS__ROT(d, double, double);
DECLARE_CBLAS__ROT(cs, float _Complex, float);
DECLARE_CBLAS__ROT(zd, double _Complex, double);

/* cblas_?rotg - computes the parameters for a Givens rotation */
void cblas_srotg (float *a, float *b, float *c, float *s);
void cblas_drotg (double *a, double *b, double *c, double *s);
void cblas_crotg (float _Complex *a, const float _Complex *b, float *c, float _Complex *s);
void cblas_zrotg (double _Complex *a, const double _Complex *b, double *c, double _Complex *s);

/* cblas_?rotm - performs modified Givens rotation of points in the plane */
void cblas_srotm (const int n,
        float *x, const int incx, 
        float *y, const int incy, 
        const float *param);
void cblas_drotm (const int n, 
        double *x, const int incx, 
        double *y, const int incy, 
        const double *param);

/* cblas_?rotmg - computes the parameters for a modified Givens rotation */
void cblas_srotmg (float *d1, 
        float *d2, 
        float *x1, 
        const float y1, 
        float *param);
void cblas_drotmg (double *d1, 
        double *d2, 
        double *x1, 
        const double y1, 
        double *param);

/* cblas_?scal - computes the product of a vector by a scalar */
#define DECLARE_CBLAS__SCAL(prefix, vtype, stype)   \
void cblas_##prefix##scal (const int n,             \
        const stype a,                              \
        vtype *x,                                   \
        const int incx)

DECLARE_CBLAS__SCAL(s, float, float);
DECLARE_CBLAS__SCAL(d, double, double);
DECLARE_CBLAS__SCAL(c, float _Complex, float _Complex);
DECLARE_CBLAS__SCAL(z, double _Complex, double _Complex);
DECLARE_CBLAS__SCAL(cs, float _Complex, float);
DECLARE_CBLAS__SCAL(zd, double _Complex, double);

/* cblas_?swap - swaps a vector with another vector */
#define DECLARE_CBLAS__SWAP(prefix, type)           \
void cblas_##prefix##swap (const int n,             \
        type *x,                                    \
        const int incx,                             \
        type *y,                                    \
        const int incy)

DECLARE_CBLAS__SWAP(s, float);
DECLARE_CBLAS__SWAP(d, double);
DECLARE_CBLAS__SWAP(c, float _Complex);
DECLARE_CBLAS__SWAP(z, double _Complex);

/* cblas_i?amax - finds the index of the element with maximum value */
#define DECLARE_CBLAS_I_AMAX(prefix, type)          \
CBLAS_INDEX cblas_i##prefix##amax (const int n,     \
        const type *x,                              \
        const int incx)

DECLARE_CBLAS_I_AMAX(s, float);
DECLARE_CBLAS_I_AMAX(d, double);
DECLARE_CBLAS_I_AMAX(c, float _Complex);
DECLARE_CBLAS_I_AMAX(z, double _Complex);

/* cblas_i?amin - finds the index of the element with the smallest absolute
 * value */
#define DECLARE_CBLAS_I_AMIN(prefix, type)          \
CBLAS_INDEX cblas_i##prefix##amin (const int n,     \
        const type *x,                              \
        const int incx)

DECLARE_CBLAS_I_AMIN(s, float);
DECLARE_CBLAS_I_AMIN(d, double);
DECLARE_CBLAS_I_AMIN(c, float _Complex);
DECLARE_CBLAS_I_AMIN(z, double _Complex);


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


/* Level 3 BLAS Routines */

/* ?gemm - matrix-matrix product with general matrices */
#define DECLARE_CBLAS__GEMM(prefix, T)                  \
void cblas_##prefix##gemm(const CBLAS_LAYOUT Layout,    \
        const CBLAS_TRANSPOSE transa,                   \
        const CBLAS_TRANSPOSE transb,                   \
        const int m, const int n, const int k,          \
        const T alpha,                                  \
        const T *a, const int lda,                      \
        const T *b, const int ldb,                      \
        const T beta,                                   \
        T *c, const int ldc)

DECLARE_CBLAS__GEMM(s, float);
DECLARE_CBLAS__GEMM(d, double);
DECLARE_CBLAS__GEMM(c, float _Complex);
DECLARE_CBLAS__GEMM(z, double _Complex);


/* ?hemm - matrix-matrix product with general matrices */
#define DECLARE_CBLAS__HEMM(prefix, T)                  \
void cblas_##prefix##hemm(const CBLAS_LAYOUT Layout,    \
        const CBLAS_SIDE side,                          \
        const CBLAS_UPLO uplo,                          \
        const int m, const int n,                       \
        const T *alpha,                                 \
        const T *a, const int lda,                      \
        const T *b, const int ldb,                      \
        const T *beta,                                  \
        T *c, const int ldc)

DECLARE_CBLAS__HEMM(c, float _Complex);
DECLARE_CBLAS__HEMM(z, double _Complex);

/* ?herk - Hermitian rank-k update */
#define DECLARE_CBLAS__HERK(prefix, S, T)               \
void cblas_##prefix##herk(const CBLAS_LAYOUT Layout,    \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE trans,                    \
        const int n, const int k,                       \
        const S alpha,                                  \
        const T *a, const int lda,                      \
        const S beta,                                   \
        T *c, const int ldc)

DECLARE_CBLAS__HERK(c, float, float _Complex);
DECLARE_CBLAS__HERK(z, float, double _Complex);

/* ?her2k */

/* ?symm - matrix-matrix product where one input is symmetric */
#define DECLARE_CBLAS__SYMM(prefix, T)                  \
void cblas_##prefix##symm(const CBLAS_LAYOUT Layout,    \
        const CBLAS_SIDE side,                          \
        const CBLAS_UPLO uplo,                          \
        const int m, const int n,                       \
        const T alpha,                                  \
        const T *a, const int lda,                      \
        const T *b, const int ldb,                      \
        const T beta,                                   \
        T *c, const int ldc)

DECLARE_CBLAS__SYMM(s, float);
DECLARE_CBLAS__SYMM(d, double);
DECLARE_CBLAS__SYMM(c, float _Complex);
DECLARE_CBLAS__SYMM(z, double _Complex);

/* ?syrk */

/* ?syr2k */

/* ?trmm */

/* ?trsm - solves a triangular matrix equation */
#define DECLARE_CBLAS__TRSM(prefix, T)                  \
void cblas_##prefix##trsm(const CBLAS_LAYOUT Layout,    \
        const CBLAS_SIDE side,                          \
        const CBLAS_UPLO uplo,                          \
        const CBLAS_TRANSPOSE transa,                   \
        const CBLAS_DIAG diag,                          \
        const int m, const int n,                       \
        const T alpha,                                  \
        const T *a, const int lda,                      \
        T *b, const int ldb)

DECLARE_CBLAS__TRSM(s, float);
DECLARE_CBLAS__TRSM(d, double);
DECLARE_CBLAS__TRSM(c, float _Complex);
DECLARE_CBLAS__TRSM(z, double _Complex);

#ifdef __cplusplus
};
#endif


#endif
