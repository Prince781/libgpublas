#define _GNU_SOURCE
#include "runtime-blas.h"
#include <stdbool.h>
#include "common.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <dlfcn.h>


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
    xerbla_(func_name_to_f77(routine), (int[]){arg});
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


int runtime_blas_lsame(const char *side_p, const char *ch_p) {
    extern int lsame_(const char *cha, const char *chb);
    return lsame_(side_p, ch_p);
}

void *runtime_blas_func(const char *name) {
    void *fptr = dlsym(RTLD_NEXT, name);

    if (fptr == NULL) {
        writef(STDERR_FILENO, "blas2cuda: failed to lookup '%s': %s\n", name, dlerror());
        abort();
    }

    return fptr;
}
