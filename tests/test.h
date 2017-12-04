#ifndef TEST_H
#define TEST_H

/* test infrastructure */

#include <time.h>
#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

typedef int (*prologue_t)(int num);
typedef void (*blas_func_t)(void);
typedef int (*epilogue_t)(int num);

struct perf_info {
    struct timespec avg;
    struct timespec total;
};

/**
 * Run {@num} tests. Fills {@pinfo}.
 */
void run_test(int num, prologue_t prologue,
        blas_func_t blas_op,
        epilogue_t epilogue,
        struct perf_info *pinfo);

#endif
