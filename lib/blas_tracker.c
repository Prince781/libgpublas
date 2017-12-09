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
    pid_t tid;

    if (!blas_tracker_libname) {
        fprintf(stderr, "OBJTRACKER_BLASLIB is undefined. See OBJTRACKER_HELP\n");
    } else if (!blas_lib_handle) {
        fprintf(stderr, "library handle is NULL.\n");
    }

    tid = syscall(SYS_gettid);

    if ((info = obj_tracker_objinfo((void *) ptr))) {
        printf("C [%p] reqsize=[%zu] ip=[0x%lx] tid=[%d]\n",
                info->ptr, info->ci.reqsize, info->ci.ip, tid);
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
        fprintf(stderr, "dlopen(%s) : %s\n", blas_tracker_libname, dlerror());
    } else
        printf("object tracker: got a handle to BLAS library %s\n", blas_tracker_libname);
}

void blas_tracker_fini(void) {
    if (dlclose(blas_lib_handle) != 0) {
        fprintf(stderr, "dlclose(%s) : %s\n", blas_tracker_libname, dlerror());
        abort();
    }
}

/* Level 1 */
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


/* Level 2 */

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

