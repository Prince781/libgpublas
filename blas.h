#ifndef BLAS_H_H
#define BLAS_H_H

#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>

/* BLAS Level 1 */
#define F77_asum(prefix, T)                         \
void prefix##asum_(const int *n,                    \
        T *sx, const int *incx)

F77_asum(s, float);
F77_asum(sc, float _Complex);
F77_asum(d, double);
F77_asum(dz, double _Complex);

#define F77_axpy(prefix, T)                         \
void prefix##axpy_(const int *n,                    \
        const T *sa,                                \
        T *sx,                                      \
        int *incx,                                  \
        T *sy,                                      \
        int *incy)

F77_axpy(s, float);
F77_axpy(d, double);
F77_axpy(c, float _Complex);
F77_axpy(z, double _Complex);

#define F77_nrm2(prefix, T)                         \
void prefix##nrm2_(int *n,                          \
        T *x,                                       \
        int *incx)

#define F77_copy(prefix, T)                         \
void prefix##copy_(int *n,                          \
        T *sx,                                      \
        int *incx,                                  \
        T *sy,                                      \
        int *incy)

#define F77_dot(prefix, T)                          \
void prefix##dot_(int *n,                           \
        T *sb,                                      \
        T *sx,                                      \
        int *incx,                                  \
        T *sy,                                      \
        int *incy)

#define F77_rot(prefix, T)                          \
void prefix##rot_(int *n,                           \
        T *sb,                                      \
        T *sx,                                      \
        int *incx,                                  \
        T *sy,                                      \
        int *incy)

#define F77_scal(prefix, S, T)                      \
void prefix##scal_(int *n,                          \
        S *sa,                                      \
        T *sx,                                      \
        int *incx)

#define F77_swap(prefix, T)                         \
void prefix##swap_(int *n,                          \
        T *sx,                                      \
        int *incx,                                  \
        T *sy,                                      \
        int *incy)


/* BLAS Level 2 */
#define F77_gbmv(prefix, T)                         \
void prefix##gbmv_(char *trans,                     \
        int *m, int *n, int *kl, int *ku,           \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *x, int *incx,                            \
        T *beta,                                    \
        T *y, int *incy)

#define F77_gemv(prefix, T)                         \
void prefix##gemv_(char *trans,                     \
        int *m, int *n,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *x, int *incx,                            \
        T *beta,                                    \
        T *y, int *incy)

#define F77_ger(prefix, T)                          \
void prefix##ger_(int *m, int *n,                   \
        T *alpha,                                   \
        T *x, int *incx,                            \
        T *y, int *incy,                            \
        T *a, int *lda)

#define F77_sbmv(prefix, T)                         \
void prefix##sbmv_(char *uplo,                      \
        int *n, int *k,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *x, int *incx,                            \
        T *beta,                                    \
        T *y, int *incy)

#define F77_spmv(prefix, T)                         \
void prefix##spmv_(char *uplo,                      \
        int *n,                                     \
        T *alpha,                                   \
        T *ap,                                      \
        T *x, int *incx,                            \
        T *beta,                                    \
        T *y, int *incy)

#define F77_spr(prefix, T)                          \
void prefix##spr_(char *uplo,                       \
        int *n,                                     \
        T *alpha,                                   \
        T *x, int *incx,                            \
        T *ap)

#define F77_spr2(prefix, T)                         \
void prefix##spr2_(char *uplo,                      \
        int *n,                                     \
        T *alpha,                                   \
        T *x, int *incx,                            \
        T *y, int *incy,                            \
        T *ap)

#define F77_symv(prefix, T)                         \
void prefix##symv_(char *uplo,                      \
        int *n,                                     \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *x, int *incx,                            \
        T *beta,                                    \
        T *y, int *incy)

#define F77_syr(prefix, T)                          \
void prefix##syr_(char *uplo,                       \
        int *n,                                     \
        T *alpha,                                   \
        T *x, int *incx,                            \
        T *a, int *lda)

#define F77_syr2(prefix, T)                         \
void prefix##syr2_(char *uplo,                      \
        int *n,                                     \
        T *alpha,                                   \
        T *x, int *incx,                            \
        T *y, int *incy,                            \
        T *a, int *lda)

#define F77_tbmv(prefix, T)                         \
void prefix##tbmv_(char *uplo,                      \
        char *trans, char *diag,                    \
        int *n, int *k,                             \
        T *a, int *lda,                             \
        T *x, int *incx)

#define F77_tbsv(prefix, T)                         \
void prefix##tbsv_(char *uplo,                      \
        char *trans, char *diag,                    \
        int *n, int *k,                             \
        T *a, int *lda,                             \
        T *x, int *incx)

#define F77_tpmv(prefix, T)                         \
void prefix##tpmv_(char *uplo,                      \
        char *trans, char *diag,                    \
        int *n,                                     \
        T *ap,                                      \
        T *x, int *incx)

#define F77_tpsv(prefix, T)                         \
void prefix##tpsv_(char *uplo,                      \
        char *trans, char *diag,                    \
        int *n,                                     \
        T *ap,                                      \
        T *x, int *incx)

#define F77_trmv(prefix, T)                         \
void prefix##trmv_(char *uplo,                      \
        char *trans, char *diag,                    \
        int *n,                                     \
        T *a, int *lda,                             \
        T *x, int *incx)

#define F77_trsv(prefix, T)                         \
void prefix##trsv_(char *uplo,                      \
        char *trans, char *diag,                    \
        int *n,                                     \
        T *a, int *lda,                             \
        T *x, int *incx)

/* BLAS Level 3 */

#define F77_gemm(prefix, T)                         \
void prefix##gemm_(const char *transa,              \
        const char *transb,                         \
        const int *m, const int *n, const int *k,   \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb,                             \
        T *beta,                                    \
        T *c, int *ldc)

F77_gemm(s, float);
F77_gemm(d, double);
F77_gemm(c, float _Complex);
F77_gemm(z, double _Complex);

#define F77_hemm(prefix, T)                         \
void prefix##hemm_(char *side, char *uplo,          \
        int *m, int *n,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb,                             \
        T *beta,                                    \
        T *c, int *ldc)

#define F77_herk(prefix, S, T)                      \
void prefix##herk_(char *uplo, char *trans,         \
        int *n, int *k,                             \
        S *alpha,                                   \
        T *a, int *lda,                             \
        S *beta,                                    \
        T *c, int *ldc)

#define F77_her2k(prefix, S, T)                     \
void prefix##her2k_(char *uplo, char *trans,        \
        int *n, int *k,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb,                             \
        S *beta,                                    \
        T *c, int *ldc)

#define F77_symm(prefix, T)                         \
void prefix##symm_(char *side, char *uplo,          \
        int *m, int *n,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb,                             \
        T *beta,                                    \
        T *c, int *ldc)

#define F77_syrk(prefix, T)                         \
void prefix##syrk_(char *uplo, char *trans,         \
        int *n, int *k,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *beta,                                    \
        T *c, int *ldc)

#define F77_syr2k(prefix, T)                        \
void prefix##syr2k_(char *uplo, char *trans,        \
        int *n, int *k,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb,                             \
        T *beta,                                    \
        T *c, int *ldc)


#define F77_trmm(prefix, T)                         \
void prefix##trmm_(char *side,                      \
        char *uplo, char *transa, char *diag,       \
        int *m, int *n,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb)

#define F77_trsm(prefix, T)                         \
void prefix##trsm_(char *side, char *uplo,          \
        char *transa, char *diag,                   \
        int *m, int *n,                             \
        T *alpha,                                   \
        T *a, int *lda,                             \
        T *b, int *ldb)

void _b2c_xerbla(const char *routine, int arg_pos);

#ifdef __cplusplus
};
#endif

#endif
