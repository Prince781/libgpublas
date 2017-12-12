#include <stdio.h>
#include <stdlib.h>
#include "../cblas.h"
#include "fortran-compat.h"

static CBLAS_TRANSPOSE c_trans(char c) {
    switch (c) {
        case 'N':
        case 'n':
            return CblasNoTrans;
        case 'T':
        case 't':
            return CblasTrans;
        case 'C':
        case 'c':
            return CblasConjTrans;
        default:
            fprintf(stderr, "Invalid transpose op '%c'\n", c);
            abort();
            break;
    }
}

static CBLAS_SIDE c_side(char c) {
    switch (c) {
        case 'L':
        case 'l':
            return CblasLeft;
        case 'R':
        case 'r':
            return CblasRight;
        default:
            fprintf(stderr, "Invalid side op '%c'\n", c);
            abort();
            break;
    }
}

static CBLAS_UPLO c_uplo(char c) {
    switch (c) {
        case 'U':
        case 'u':
            return CblasUpper;
        case 'L':
        case 'l':
            return CblasLower;
        default:
            fprintf(stderr, "Invalid uplo op '%c'\n", c);
            abort();
            break;
    }
}

static CBLAS_DIAG c_diag(char c) {
    switch (c) {
        case 'U':
        case 'u':
            return CblasUnit;
        case 'N':
        case 'n':
            return CblasNonUnit;
        default:
            fprintf(stderr, "Invalid diag op '%c'\n", c);
            abort();
            break;
    }
}

/* Level 3 */
#define IMPL_F77_gemm(prefix, T)                                    \
F77_gemm(prefix, T) {                                               \
    cblas_##prefix##gemm(CblasColMajor,                             \
            c_trans(*transa), c_trans(*transb),                     \
            *m, *n, *k,                                             \
            *alpha,                                                 \
            a, *ldb,                                                \
            b, *ldb,                                                \
            *beta,                                                  \
            c, *ldc);                                               \
}

IMPL_F77_gemm(s, float)
IMPL_F77_gemm(d, double)
IMPL_F77_gemm(c, float _Complex)
IMPL_F77_gemm(z, double _Complex)

#define IMPL_F77_hemm(prefix, T)                                    \
F77_hemm(prefix, T) {                                               \
    cblas_##prefix##hemm(CblasColMajor,                             \
            c_side(*side), c_uplo(*uplo),                           \
            *m, *n,                                                 \
            alpha,                                                  \
            a, *lda,                                                \
            b, *ldb,                                                \
            beta,                                                   \
            c, *ldc);                                               \
}

IMPL_F77_hemm(c, float _Complex)
IMPL_F77_hemm(z, double _Complex)

#define IMPL_F77_herk(prefix, S, T)                                 \
F77_herk(prefix, S, T) {                                            \
    cblas_##prefix##herk(CblasColMajor,                             \
            c_uplo(*uplo), c_trans(*trans),                         \
            *n, *k,                                                 \
            *alpha,                                                 \
            a, *lda,                                                \
            *beta,                                                  \
            c, *ldc);                                               \
}

IMPL_F77_herk(c, float, float _Complex)
IMPL_F77_herk(z, double, double _Complex)

#define IMPL_F77_her2k(prefix, S, T)                                \
F77_her2k(prefix, S, T) {                                           \
    cblas_##prefix##her2k(CblasColMajor,                            \
            c_uplo(*uplo), c_trans(*trans),                         \
            *n, *k,                                                 \
            alpha,                                                  \
            a, *lda,                                                \
            b, *ldb,                                                \
            *beta,                                                  \
            c, *ldc);                                               \
}

IMPL_F77_her2k(c, float, float _Complex)
IMPL_F77_her2k(z, double, double _Complex)

#define IMPL_F77_symm(prefix, T)                                    \
F77_symm(prefix, T) {                                               \
    cblas_##prefix##symm(CblasColMajor,                             \
            c_side(*side), c_uplo(*uplo),                           \
            *m, *n,                                                 \
            *alpha,                                                 \
            a, *lda,                                                \
            b, *ldb,                                                \
            *beta,                                                  \
            c, *ldc);                                               \
}

IMPL_F77_symm(s, float)
IMPL_F77_symm(d, double)
IMPL_F77_symm(c, float _Complex)
IMPL_F77_symm(z, double _Complex)

#define IMPL_F77_syrk(prefix, T)                                    \
F77_syrk(prefix, T) {                                               \
    cblas_##prefix##syrk(CblasColMajor,                             \
            c_uplo(*uplo), c_trans(*trans),                         \
            *n, *k,                                                 \
            *alpha,                                                 \
            a, *lda,                                                \
            *beta,                                                  \
            c, *ldc);                                               \
}

IMPL_F77_syrk(s, float)
IMPL_F77_syrk(d, double)
IMPL_F77_syrk(c, float _Complex)
IMPL_F77_syrk(z, double _Complex)

#define IMPL_F77_syr2k(prefix, T)                                   \
F77_syr2k(prefix, T) {                                              \
    cblas_##prefix##syr2k(CblasColMajor,                            \
            c_uplo(*uplo), c_trans(*trans),                         \
            *n, *k,                                                 \
            *alpha,                                                 \
            a, *lda,                                                \
            b, *ldb,                                                \
            *beta,                                                  \
            c, *ldc);                                               \
}

IMPL_F77_syr2k(s, float)
IMPL_F77_syr2k(d, double)
IMPL_F77_syr2k(c, float _Complex)
IMPL_F77_syr2k(z, double _Complex)

#define IMPL_F77_trmm(prefix, T)                                    \
F77_trmm(prefix, T) {                                               \
    cblas_##prefix##trmm(CblasColMajor,                             \
            c_side(*side), c_uplo(*uplo),                           \
            c_trans(*transa), c_diag(*diag),                        \
            *m, *n,                                                 \
            *alpha,                                                 \
            a, *lda,                                                \
            b, *ldb);                                               \
}

IMPL_F77_trmm(s, float)
IMPL_F77_trmm(d, double)
IMPL_F77_trmm(c, float _Complex)
IMPL_F77_trmm(z, double _Complex)

#define IMPL_F77_trsm(prefix, T)                                    \
F77_trsm(prefix, T) {                                               \
    cblas_##prefix##trsm(CblasColMajor,                             \
            c_side(*side), c_uplo(*uplo),                           \
            c_trans(*transa), c_diag(*diag),                        \
            *m, *n,                                                 \
            *alpha,                                                 \
            a, *lda,                                                \
            b, *ldb);                                               \
}

IMPL_F77_trsm(s, float)
IMPL_F77_trsm(d, double)
IMPL_F77_trsm(c, float _Complex)
IMPL_F77_trsm(z, double _Complex)
