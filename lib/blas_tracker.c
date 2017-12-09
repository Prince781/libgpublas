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
    typeof(cblas_scopy) *fun;                   \
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
#define GEMM_TRACK(prefix, type)                \
DECLARE_CBLAS__GEMM(prefix, type) {             \
    typeof(cblas_sgemm) *fun;                   \
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
