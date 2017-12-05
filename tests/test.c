#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
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
            fprintf(stderr, "sub-test %d failed\n", i+1);
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
            fprintf(stderr, "sub-test %d failed\n", i+1);
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

void print_help(const char *progname) {
    fprintf(stderr, "Usage: %s N\n", progname);
}

void get_N_or_fail(int argc, char *argv[], int *N) {
    int n;

    errno = 0;
    if (argc > 1) {
        n = (int) strtol(argv[1], NULL, 0);
        if (errno) {
            perror("strtol");
            exit(1);
        } else if (n < 1) {
            fprintf(stderr, "N must be positive\n");
            exit(1);
        }
    } else {
        fprintf(stderr, "Usage: %s N (N is size of element(s) of computation)\n", argv[0]);
        exit(1);
    }

    *N = n;
}

void print_results(const char *name, 
        int n, const struct perf_info *pinfo) {
    printf(" %10s[n=%d]: %ld s + %ld ns\n",
            name, n, pinfo->avg.tv_sec, pinfo->avg.tv_nsec);
}

void print_mat_f(const float *mat, int rows, int cols, FILE *fin) {
    for (int r=0; r<rows; ++r) {
        fprintf(fin, "|");
        for (int c=0; c<cols; ++c) {
            float e = mat[r*cols + c];
            fprintf(fin, " %f", e);
        }
        fprintf(fin, " |\n");
    }
}

void print_mat_d(const double *mat, int rows, int cols, FILE *fin) {
    for (int r=0; r<rows; ++r) {
        fprintf(fin, "|");
        for (int c=0; c<cols; ++c) {
            double e = mat[r*cols + c];
            fprintf(fin, " %lf", e);
        }
        fprintf(fin, " |\n");
    }
}

void print_mat_cf(const complex float *mat, int rows, int cols, FILE *fin) {
    for (int r=0; r<rows; ++r) {
        fprintf(fin, "|");
        for (int c=0; c<cols; ++c) {
            complex float e = mat[r*cols + c];
            fprintf(fin, " %f + %fi", crealf(e), cimagf(e));
        }
        fprintf(fin, " |\n");
    }
}

void print_mat_cd(const complex double *mat, int rows, int cols, FILE *fin) {
    for (int r=0; r<rows; ++r) {
        fprintf(fin, "|");
        for (int c=0; c<cols; ++c) {
            complex double e = mat[r*cols + c];
            fprintf(fin, " %lf + %lfi", creal(e), cimag(e));
        }
        fprintf(fin, " |\n");
    }
}
