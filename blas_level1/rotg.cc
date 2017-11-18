#include "level1.h"

void cblas_srotg (float *a, float *b, float *c, float *s) {
    cublasSrotg(b2c_handle, a, b, c, s);
}

void cblas_drotg (double *a, double *b, double *c, double *s) {
    cublasDrotg(b2c_handle, a, b, c, s);
}

void cblas_crotg (float _Complex *a,
        const float _Complex *b, 
        float *c, float _Complex *s) {
    cuComplex b_tmp = *(const cuComplex *) b;
    cublasCrotg(b2c_handle, (cuComplex *) a, &b_tmp, c, (cuComplex *) s);
}

void cblas_zrotg (double _Complex *a, 
        const double _Complex *b, 
        double *c,
        double _Complex *s) {
    cuDoubleComplex b_tmp = *(const cuDoubleComplex *) b;
    cublasZrotg(b2c_handle, (cuDoubleComplex *) a, &b_tmp, c,
            (cuDoubleComplex *) s);
}

