#ifndef BLAS2CUDA_LEVEL1_H
#define BLAS2CUDA_LEVEL1_H

#include "../blas2cuda.h"

extern "C" {

/* BLAS Level 1 routines */

/* cblas_?asum - sum of vector magnitudes (functions) */
float cblas_sasum (const int n, const float *x, const int incx);

float cblas_scasum (const int n, const void *x, const int incx);

/* cblas_?axpy - scalar-vector product (routines) */

/* cblas_?copy - copy vector (routines) */
void cblas_scopy (const int n,
        const float *x, 
        const int incx,
        float *y,
        const int incy);

/* ... TODO ... */

};

#endif
