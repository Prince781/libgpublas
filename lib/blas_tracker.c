#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "callinfo.h"
#include "obj_tracker.h"
#include "../cblas.h"

char *blas_tracker_libname;
void *blas_lib_handle;

static void print_objtrack_info(const void *ptr) {
    const struct objinfo *info;

    if (!blas_tracker_libname) {
        fprintf(stderr, "OBJTRACKER_BLASLIB is undefined. See OBJTRACKER_HELP\n");
    } else if (!blas_lib_handle) {
        fprintf(stderr, "library handle is NULL.\n");
    }

    if ((info = obj_tracker_objinfo((void *) ptr))) {
        obj_tracker_print_info(OBJPRINT_CALL, info);
    } else
        fprintf(stderr, "no objinfo for %p\n", ptr);
}

static void *get_real_blas_fun(const char *name) {
    void *sym;
    /* Level 3 */
    if (strcmp(name, "cblas_sgemm") == 0) {
        dlerror();
        if (!(sym = dlsym(blas_lib_handle, name))) {
            fprintf(stderr, "failed to get %s(): %s\n", name, dlerror());
        }
        return sym;
    }

    return NULL;
}

void blas_tracker_init(void) {
    if (!blas_tracker_libname) {
        fprintf(stderr, "blas_tracker_libname not set\n");
        abort();
    }

    dlerror();
    if (!(blas_lib_handle = dlopen(blas_tracker_libname, RTLD_LAZY))) {
        fprintf(stderr, "dlopen() : %s\n", dlerror());
        abort();
    } else
        printf("object tracker: got a handle to BLAS library %s\n", blas_tracker_libname);
}

void blas_tracker_fini(void) {
    if (dlclose(blas_lib_handle) != 0) {
        fprintf(stderr, "dlclose() : %s\n", dlerror());
        abort();
    }
}

/* Level 1 */
#define ASUM_TRACK(prefix, T)                   \
DECLARE_CBLAS__ASUM(prefix, T) {                \
    typeof(cblas_##prefix##asum) *fun;          \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        return fun(n, x, incx);                 \
    abort();                                    \
}

ASUM_TRACK(s, float)
ASUM_TRACK(sc, float _Complex)
ASUM_TRACK(d, double)
ASUM_TRACK(dz, double _Complex)

#define AXPY_TRACK(prefix, T)                   \
DECLARE_CBLAS__AXPY(prefix, T) {                \
    typeof(cblas_##prefix##axpy) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(n, a, x, incx, y, incy);            \
}

AXPY_TRACK(s, float)
AXPY_TRACK(d, double)
AXPY_TRACK(c, float _Complex)
AXPY_TRACK(z, double _Complex)

#define COPY_TRACK(prefix, type)                \
DECLARE_CBLAS__COPY(prefix, type) {             \
    typeof(cblas_##prefix##copy) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__))) {  \
        fun(n, x, incx, y, incy);               \
    }                                           \
}

COPY_TRACK(s, float)
COPY_TRACK(d, double)
COPY_TRACK(c, float _Complex)
COPY_TRACK(z, double _Complex)

#define DOT_TRACK(prefix, T)                    \
DECLARE_CBLAS__DOT(prefix, T) {                 \
    typeof(cblas_##prefix##dot) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        return fun(n, x, incx, y, incy);        \
    abort();                                    \
}

DOT_TRACK(s, float)
DOT_TRACK(d, double)

#define DOTC_TRACK(prefix, T)                   \
DECLARE_CBLAS__DOTC(prefix, T) {                \
    typeof(cblas_##prefix##dotc_sub) *fun;      \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__))) {  \
        fun(n, x, incx, y, incy, dotc);         \
    }                                           \
}

DOTC_TRACK(c, float _Complex)
DOTC_TRACK(z, double _Complex)

#define DOTU_TRACK(prefix, T)                   \
DECLARE_CBLAS__DOTU(prefix, T) {                \
    typeof(cblas_##prefix##dotu_sub) *fun;      \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(n, x, incx, y, incy, dotu);         \
}

DOTU_TRACK(c, float _Complex)
DOTU_TRACK(z, double _Complex)

#define NRM2_TRACK(prefix, T)                   \
DECLARE_CBLAS__NRM2(prefix, T) {                \
    typeof(cblas_##prefix##_nrm2) *fun;         \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        return fun(n, x, incx);                 \
    abort();                                    \
}

NRM2_TRACK(s, float)
NRM2_TRACK(d, double)
NRM2_TRACK(sc, float _Complex)
NRM2_TRACK(dz, double _Complex)

#define ROT_TRACK(prefix, S, T)                 \
DECLARE_CBLAS__ROT(prefix, S, T) {              \
    typeof(cblas_##prefix##rot) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(n, x, incx, y, incy, c, s);         \
}

ROT_TRACK(s, float, float)
ROT_TRACK(d, double, double)
ROT_TRACK(cs, float _Complex, float)
ROT_TRACK(zd, double _Complex, double)

/* NOTE: rotg omitted */

void cblas_srotm (const int n,
        float *x, const int incx, 
        float *y, const int incy, 
        const float *param) {
    typeof(cblas_srotm) *fun;
    print_objtrack_info(x);
    print_objtrack_info(y);
    if ((fun = get_real_blas_fun(__func__)))
        fun(n, x, incx, y, incy, param);
}

void cblas_drotm (const int n, 
        double *x, const int incx, 
        double *y, const int incy, 
        const double *param) {
    typeof(cblas_drotm) *fun;
    print_objtrack_info(x);
    print_objtrack_info(y);
    if ((fun = get_real_blas_fun(__func__)))
        fun(n, x, incx, y, incy, param);
}

/* NOTE: rotmg omitted */
#define SCAL_TRACK(prefix, S, T)                \
DECLARE_CBLAS__SCAL(prefix, S, T) {             \
    typeof(cblas_##prefix##scal) *fun;          \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(n, a, x, incx);                     \
}

SCAL_TRACK(s, float, float)
SCAL_TRACK(d, double, double)
SCAL_TRACK(c, float _Complex, float _Complex)
SCAL_TRACK(z, double _Complex, double _Complex)
SCAL_TRACK(cs, float _Complex, float)
SCAL_TRACK(zd, double _Complex, double)

#define SWAP_TRACK(prefix, T)                   \
DECLARE_CBLAS__SWAP(prefix, T) {                \
    typeof(cblas_##prefix##swap) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(n, x, incx, y, incy);               \
}

SWAP_TRACK(s, float)
SWAP_TRACK(d, double)
SWAP_TRACK(c, float _Complex)
SWAP_TRACK(z, double _Complex)

#define I_AMAX_TRACK(prefix, T)                 \
DECLARE_CBLAS_I_AMAX(prefix, T) {               \
    typeof(cblas_i##prefix##amax) *fun;         \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        return fun(n, x, incx);                 \
    abort();                                    \
}

I_AMAX_TRACK(s, float)
I_AMAX_TRACK(d, double)
I_AMAX_TRACK(c, float _Complex)
I_AMAX_TRACK(z, double _Complex)


#define I_AMIN_TRACK(prefix, T)                 \
DECLARE_CBLAS_I_AMIN(prefix, T) {               \
    typeof(cblas_i##prefix##amin) *fun;         \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        return fun(n, x, incx);                 \
    abort();                                    \
}

I_AMIN_TRACK(s, float)
I_AMIN_TRACK(d, double)
I_AMIN_TRACK(c, float _Complex)
I_AMIN_TRACK(z, double _Complex)

/* Level 2 */
#define GBMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__GBMV(prefix, T) {                \
    typeof(cblas_##prefix##gbmv) *fun;          \
    print_objtrack_info(A);                     \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, trans,                      \
                m, n,                           \
                kl, ku,                         \
                alpha,                          \
                A, lda,                         \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

GBMV_TRACK(s, float)
GBMV_TRACK(d, double)
GBMV_TRACK(c, float _Complex)
GBMV_TRACK(z, double _Complex)

#define GEMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__GEMV(prefix, T) {                \
    typeof(cblas_##prefix##gemv) *fun;          \
    print_objtrack_info(A);                     \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, trans,                      \
                m, n,                           \
                alpha,                          \
                A, lda,                         \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

GEMV_TRACK(s, float)
GEMV_TRACK(d, double)
GEMV_TRACK(c, float _Complex)
GEMV_TRACK(z, double _Complex)

#define GER_TRACK(prefix, T)                    \
DECLARE_CBLAS__GER(prefix, T) {                 \
    typeof(cblas_##prefix##ger) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, m, n,                       \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                a, lda);                        \
}

GER_TRACK(s, float)
GER_TRACK(d, double)

#define GERC_TRACK(prefix, T)                   \
DECLARE_CBLAS__GERC(prefix, T) {                \
    typeof(cblas_##prefix##gerc) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                m, n,                           \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                a, lda);                        \
}

GERC_TRACK(c, float _Complex)
GERC_TRACK(z, double _Complex)

#define GERU_TRACK(prefix, T)                   \
DECLARE_CBLAS__GERU(prefix, T) {                \
    typeof(cblas_##prefix##geru) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                m, n,                           \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                a, lda);                        \
}

GERU_TRACK(c, float _Complex)
GERU_TRACK(z, double _Complex)

#define HBMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__HBMV(prefix, T) {                \
    typeof(cblas_##prefix##hbmv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n, k,                           \
                alpha,                          \
                a, lda,                         \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

HBMV_TRACK(c, float _Complex)
HBMV_TRACK(z, double _Complex)

#define HEMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__HEMV(prefix, T) {                \
    typeof(cblas_##prefix##hemv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                a, lda,                         \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

HEMV_TRACK(c, float _Complex)
HEMV_TRACK(z, double _Complex)

#define HER_TRACK(prefix, S, T)                 \
DECLARE_CBLAS__HER(prefix, S, T) {              \
    typeof(cblas_##prefix##her) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                a, lda);                        \
}

HER_TRACK(c, float, float _Complex)
HER_TRACK(z, double, double _Complex)

#define HER2_TRACK(prefix, T)                   \
DECLARE_CBLAS__HER2(prefix, T) {                \
    typeof(cblas_##prefix##her2) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                a, lda);                        \
}

HER2_TRACK(c, float _Complex)
HER2_TRACK(z, double _Complex)

#define HPMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__HPMV(prefix, T) {                \
    typeof(cblas_##prefix##hpmv) *fun;          \
    print_objtrack_info(ap);                    \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                ap,                             \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

HPMV_TRACK(c, float _Complex)
HPMV_TRACK(z, double _Complex)

#define HPR_TRACK(prefix, S, T)                 \
DECLARE_CBLAS__HPR(prefix, S, T) {              \
    typeof(cblas_##prefix##hpr) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(ap);                    \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                ap);                            \
}

HPR_TRACK(c, float, float _Complex)
HPR_TRACK(z, double, double _Complex)

#define HPR2_TRACK(prefix, T)                   \
DECLARE_CBLAS__HPR2(prefix, T) {                \
    typeof(cblas_##prefix##hpr2) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(ap);                    \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                ap);                            \
}

HPR2_TRACK(c, float _Complex)
HPR2_TRACK(z, double _Complex)

#define SBMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__SBMV(prefix, T) {                \
    typeof(cblas_##prefix##sbmv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n, k,                           \
                alpha,                          \
                a, lda,                         \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

SBMV_TRACK(s, float)
SBMV_TRACK(d, double)

#define SPMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__SPMV(prefix, T) {                \
    typeof(cblas_##prefix##spmv) *fun;          \
    print_objtrack_info(ap);                    \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                ap,                             \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

SPMV_TRACK(s, float)
SPMV_TRACK(d, double)

#define SPR_TRACK(prefix, T)                    \
DECLARE_CBLAS__SPR(prefix, T) {                 \
    typeof(cblas_##prefix##spr) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(ap);                    \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                ap);                            \
}

SPR_TRACK(s, float)
SPR_TRACK(d, double)

#define SPR2_TRACK(prefix, T)                   \
DECLARE_CBLAS__SPR2(prefix, T) {                \
    typeof(cblas_##prefix##spr2) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(ap);                    \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                ap);                            \
}

SPR2_TRACK(s, float)
SPR2_TRACK(d, double)

#define SYMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__SYMV(prefix, T) {                \
    typeof(cblas_##prefix##symv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                a, lda,                         \
                x, incx,                        \
                beta,                           \
                y, incy);                       \
}

SYMV_TRACK(s, float)
SYMV_TRACK(d, double)

#define SYR_TRACK(prefix, T)                    \
DECLARE_CBLAS__SYR(prefix, T) {                 \
    typeof(cblas_##prefix##syr) *fun;           \
    print_objtrack_info(x);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                a, lda);                        \
}

SYR_TRACK(s, float)
SYR_TRACK(d, double)

#define SYR2_TRACK(prefix, T)                   \
DECLARE_CBLAS__SYR2(prefix, T) {                \
    typeof(cblas_##prefix##syr2) *fun;          \
    print_objtrack_info(x);                     \
    print_objtrack_info(y);                     \
    print_objtrack_info(a);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                n,                              \
                alpha,                          \
                x, incx,                        \
                y, incy,                        \
                a, lda);                        \
}

SYR2_TRACK(s, float)
SYR2_TRACK(d, double)

#define TBMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__TBMV(prefix, T) {                \
    typeof(cblas_##prefix##tbmv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                trans,                          \
                diag,                           \
                n, k,                           \
                a, lda,                         \
                x, incx);                       \
}

TBMV_TRACK(s, float)
TBMV_TRACK(d, double)
TBMV_TRACK(c, float _Complex)
TBMV_TRACK(z, double _Complex)

#define TBSV_TRACK(prefix, T)                   \
DECLARE_CBLAS__TBSV(prefix, T) {                \
    typeof(cblas_##prefix##tbsv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                trans,                          \
                diag,                           \
                n, k,                           \
                a, lda,                         \
                x, incx);                       \
}

TBSV_TRACK(s, float)
TBSV_TRACK(d, double)
TBSV_TRACK(c, float _Complex)
TBSV_TRACK(z, double _Complex)

#define TPMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__TPMV(prefix, T) {                \
    typeof(cblas_##prefix##tpmv) *fun;          \
    print_objtrack_info(ap);                    \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                trans,                          \
                diag,                           \
                n, k,                           \
                ap,                             \
                x, incx);                       \
}

TPMV_TRACK(s, float)
TPMV_TRACK(d, double)
TPMV_TRACK(c, float _Complex)
TPMV_TRACK(z, double _Complex)

#define TPSV_TRACK(prefix, T)                   \
DECLARE_CBLAS__TPSV(prefix, T) {                \
    typeof(cblas_##prefix##tpsv) *fun;          \
    print_objtrack_info(ap);                    \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                trans,                          \
                diag,                           \
                n,                              \
                ap,                             \
                x, incx);                       \
}

TPSV_TRACK(s, float)
TPSV_TRACK(d, double)
TPSV_TRACK(c, float _Complex)
TPSV_TRACK(z, double _Complex)

#define TRMV_TRACK(prefix, T)                   \
DECLARE_CBLAS__TRMV(prefix, T) {                \
    typeof(cblas_##prefix##trmv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                trans,                          \
                diag,                           \
                n,                              \
                a, lda,                         \
                x, incx);                       \
}

TRMV_TRACK(s, float)
TRMV_TRACK(d, double)
TRMV_TRACK(c, float _Complex)
TRMV_TRACK(z, double _Complex)

#define TRSV_TRACK(prefix, T)                   \
DECLARE_CBLAS__TRSV(prefix, T) {                \
    typeof(cblas_##prefix##trsv) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(x);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout,                             \
                uplo,                           \
                trans,                          \
                diag,                           \
                n,                              \
                a, lda,                         \
                x, incx);                       \
}

TRSV_TRACK(s, float)
TRSV_TRACK(d, double)
TRSV_TRACK(c, float _Complex)
TRSV_TRACK(z, double _Complex)



/* Level 3 */
#define GEMM_TRACK(prefix, T)                   \
DECLARE_CBLAS__GEMM(prefix, T) {                \
    typeof(cblas_##prefix##gemm) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, transa, transb,             \
                m, n, k,                        \
                alpha,                          \
                a, lda,                         \
                b, ldb,                         \
                beta,                           \
                c, ldc);                        \
}

GEMM_TRACK(s, float)
GEMM_TRACK(d, double)
GEMM_TRACK(c, float _Complex)
GEMM_TRACK(z, double _Complex)

#define HEMM_TRACK(prefix, T)                   \
DECLARE_CBLAS__HEMM(prefix, T) {                \
    typeof(cblas_##prefix##hemm) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, side, uplo,                 \
                m, n,                           \
                alpha,                          \
                a, lda,                         \
                b, ldb,                         \
                beta,                           \
                c, ldc);                        \
}

HEMM_TRACK(s, float)
HEMM_TRACK(d, double)
HEMM_TRACK(c, float _Complex)
HEMM_TRACK(z, double _Complex)

#define HERK_TRACK(prefix, S, T)                \
DECLARE_CBLAS__HERK(prefix, S, T) {             \
    typeof(cblas_##prefix##herk) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, uplo, trans,                \
                n, k,                           \
                alpha,                          \
                a, lda,                         \
                beta,                           \
                c, ldc);                        \
}

HERK_TRACK(c, float, float _Complex)
HERK_TRACK(z, double, double _Complex)

#define HER2K_TRACK(prefix, S, T)               \
DECLARE_CBLAS__HER2K(prefix, S, T) {            \
    typeof(cblas_##prefix##her2k) *fun;         \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, uplo, trans,                \
                n, k,                           \
                alpha,                          \
                a, lda,                         \
                b, ldb,                         \
                beta,                           \
                c, ldc);                        \
}

HER2K_TRACK(c, float, float _Complex)
HER2K_TRACK(z, double, double _Complex)

#define SYMM_TRACK(prefix, T)                   \
DECLARE_CBLAS__SYMM(prefix, T) {                \
    typeof(cblas_##prefix##symm) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, side, uplo,                 \
                m, n,                           \
                alpha,                          \
                a, lda,                         \
                b, ldb,                         \
                beta,                           \
                c, ldc);                        \
}

SYMM_TRACK(s, float)
SYMM_TRACK(d, double)
SYMM_TRACK(c, float _Complex)
SYMM_TRACK(z, double _Complex)


#define SYRK_TRACK(prefix, T)                   \
DECLARE_CBLAS__SYRK(prefix, T) {                \
    typeof(cblas_##prefix##syrk) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, uplo, trans,                \
                n, k,                           \
                alpha,                          \
                a, lda,                         \
                beta,                           \
                c, ldc);                        \
}

SYRK_TRACK(s, float)
SYRK_TRACK(d, double)
SYRK_TRACK(c, float _Complex)
SYRK_TRACK(z, double _Complex)


#define SYR2K_TRACK(prefix, T)                  \
DECLARE_CBLAS__SYR2K(prefix, T) {               \
    typeof(cblas_##prefix##syr2k) *fun;         \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    print_objtrack_info(c);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, uplo, trans,                \
                n, k,                           \
                alpha,                          \
                a, lda,                         \
                b, ldb,                         \
                beta,                           \
                c, ldc);                        \
}

SYR2K_TRACK(s, float)
SYR2K_TRACK(d, double)
SYR2K_TRACK(c, float _Complex)
SYR2K_TRACK(z, double _Complex)


#define TRMM_TRACK(prefix, T)                   \
DECLARE_CBLAS__TRMM(prefix, T) {                \
    typeof(cblas_##prefix##trmm) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, side, uplo, transa, diag,   \
                m, n,                           \
                alpha,                          \
                a, lda,                         \
                b, ldb);                        \
}

TRMM_TRACK(s, float)
TRMM_TRACK(d, double)
TRMM_TRACK(c, float _Complex)
TRMM_TRACK(z, double _Complex)


#define TRSM_TRACK(prefix, T)                   \
DECLARE_CBLAS__TRSM(prefix, T) {                \
    typeof(cblas_##prefix##trsm) *fun;          \
    print_objtrack_info(a);                     \
    print_objtrack_info(b);                     \
    if ((fun = get_real_blas_fun(__func__)))    \
        fun(Layout, side, uplo, transa, diag,   \
                m, n,                           \
                alpha,                          \
                a, lda,                         \
                b, ldb);                        \
}

TRSM_TRACK(s, float)
TRSM_TRACK(d, double)
TRSM_TRACK(c, float _Complex)
TRSM_TRACK(z, double _Complex)

