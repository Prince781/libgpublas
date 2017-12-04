#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "test.h"

void run_test(int num,
        prologue_t prologue,
        blas_func_t blas_op,
        epilogue_t epilogue,
        struct perf_info *pinfo)
{
    struct timespec total = { 0, 0 };

    assert (num > 0);
    for (int i=0; i<num; ++i) {
        struct timespec start, end;
        struct timespec ts_diff;
        if (prologue(i) < 0) {
            fprintf(stderr, "sub-test %d failed\n", i);
            continue;
        }
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &start) == -1) {
            perror("clock_gettime");
            exit(1);
        }
        blas_op();
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &end) == -1) {
            perror("clock_gettime");
            exit(1);
        }
        if (epilogue(i) < 0) {
            fprintf(stderr, "sub-test %d failed\n", i);
            continue;
        }

        ts_diff.tv_sec = end.tv_sec;
        /* subtract nanoeconds */
        if (end.tv_nsec < start.tv_nsec) {
            ts_diff.tv_sec--;
            ts_diff.tv_nsec = 1000000000L + (end.tv_nsec - start.tv_nsec);
        } else 
            ts_diff.tv_nsec = end.tv_nsec - start.tv_nsec;

        /* subtract seconds */
        ts_diff.tv_sec -= start.tv_sec;

        /* add to total */
        total.tv_sec += ts_diff.tv_sec;
        if (total.tv_nsec + ts_diff.tv_nsec >= 1000000000L)
            ++total.tv_sec;
        total.tv_nsec = (total.tv_nsec + ts_diff.tv_nsec) % 1000000000L;
    }

    pinfo->total = total;
    total.tv_sec /= num;
    total.tv_nsec /= num;
    pinfo->avg = total;
}
