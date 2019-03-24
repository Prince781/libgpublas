#include "level1.h"

void cblas_srotg (float *a, float *b, float *c, float *s) {
    call_kernel(cublasSrotg(b2c_handle, a, b, c, s));
}

void cblas_drotg (double *a, double *b, double *c, double *s) {
    call_kernel(cublasDrotg(b2c_handle, a, b, c, s));
}

void cblas_crotg (float _Complex *a,
        const float _Complex *b, 
        float *c, float _Complex *s) {
    cuComplex b_tmp = cu(*b);
    call_kernel(cublasCrotg(b2c_handle, (cuComplex *) a, &b_tmp, c, (cuComplex *) s));
}

void cblas_zrotg (double _Complex *a, 
        const double _Complex *b, 
        double *c,
        double _Complex *s) {
    cuDoubleComplex b_tmp = cu(*b);
    call_kernel(
        cublasZrotg(b2c_handle, (cuDoubleComplex *) a, &b_tmp, c,
                (cuDoubleComplex *) s)
    );
}

