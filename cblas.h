#ifndef CBLAS_H
#define CBLAS_H

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define size(n,stride,sz) ((n) * (stride) * (sz))

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

#endif
