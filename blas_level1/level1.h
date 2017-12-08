#ifndef BLAS2CUDA_LEVEL1_H
#define BLAS2CUDA_LEVEL1_H

#include <ccomplex>
#include "../blas2cuda.h"
#include "../cblas.h"   /* this should come after the above */
#include "../conversions.h"

extern "C" {

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

};

#endif
