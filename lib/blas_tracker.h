#ifndef BLAS_TRACKER_H
#define BLAS_TRACKER_H

extern char *blas_tracker_libname;
extern void *blas_lib_handle;

void blas_tracker_init(void);

void blas_tracker_fini(void);

#endif
