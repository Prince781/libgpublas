#ifndef BLAS_TRACKER_H
#define BLAS_TRACKER_H

extern void **blas_lib_handles;
extern int num_handles;

void blas_tracker_init(void);

void blas_tracker_fini(void);

#endif
