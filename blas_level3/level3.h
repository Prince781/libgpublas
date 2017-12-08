#ifndef BLAS2CUDA_LEVEL3_H
#define BLAS2CUDA_LEVEL3_H
#include <complex.h>
#include "../blas2cuda.h"
#include "../cblas.h"
#include "../conversions.h"

void assign_dims(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
        int& rows_ref, int& cols_ref,
        const int ld,
        const int rows, const int cols);

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

#endif
