#ifndef BLAS2CUDA_LEVEL1_H
#define BLAS2CUDA_LEVEL1_H

#include "../blas2cuda.h"
#include <complex.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define size(n,stride,sz) (n * stride * sz)

extern "C" {

/* BLAS Level 1 routines */

/* cblas_?asum - sum of vector magnitudes (functions) */
float cblas_sasum (const int n, const float *x, const int incx);

float cblas_scasum (const int n, const void *x, const int incx);

/* cblas_?axpy - scalar-vector product (routines) */

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

/* ... TODO ... */

};

#endif
