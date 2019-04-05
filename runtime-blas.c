#include "runtime-blas.h"
#include <stdbool.h>
#include "common.h"

#include <ctype.h>
#include <stdio.h>


#if USE_CUDA
cublasHandle_t b2c_cublas_handle;
#endif

runtime_blas_error_t runtime_blas_init(void) {
#if USE_CUDA
    return cublasCreate(&b2c_cublas_handle);
#else
    return clblasSetup();
#endif
}

runtime_blas_error_t runtime_blas_fini(void) {
#if USE_CUDA
    return cublasDestroy(b2c_cublas_handle);
#else
    clblasTeardown();
    return CL_SUCCESS;
#endif
}

void runtime_blas_xerbla(const char *routine, int arg) {
    extern void xerbla_(const char *, int *);
    xerbla_(routine, (int[]){arg});
}

const char *func_name_to_f77(const char *func_name) {
    static char tbuf[64];

    snprintf(tbuf, sizeof tbuf, "%s", func_name);
    char *p = NULL;
    for (p = &tbuf[0]; *p; p++) {
        if (*p == '_')
            *p = ' ';
        else
            *p = toupper(*p);
    }

    return tbuf;
}
