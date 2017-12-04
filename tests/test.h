#ifndef TEST_H
#define TEST_H

/* test infrastructure */

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>
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

void print_help(const char *progname);

void get_N_or_fail(int argc, char *argv[], int *N);

void print_results(const char *name, int n, const struct perf_info *pinfo);

void print_mat_f(const float *mat, int rows, int cols, FILE *fin);
void print_mat_d(const double *mat, int rows, int cols, FILE *fin);
void print_mat_cf(const complex float *mat, int rows, int cols, FILE *fin);
void print_mat_cd(const complex double *mat, int rows, int cols, FILE *fin);

#define print_matrix(mat, rows, cols, fin) \
    _Generic((mat), \
            float *: print_mat_f,\
            double *: print_mat_d,\
            complex float *: print_mat_cf,\
            complex double *: print_mat_cd)(mat, rows, cols, fin)

#define print_vector(vec, len, fin) print_matrix(vec, len, 1, fin)

#define print_scalar(scal, fin) \
    _Generic((scal), \
            float: fprintf(fin, "%f", (scal)),\
            double: fprintf(fin, "%lf", (scal)),\
            complex float: fprintf(fin, "%f + %fi", crealf(scal), cimagf(scal)),\
            complex double: fprintf(fin, "%lf + %lfi", creal(scal), cimag(scal)))

#define def_gen(name,args,code,ret_t) \
    static inline ret_t name args { \
        code; \
    }

def_gen(max_i, (int a, int b), return a > b ? a : b;, int)
def_gen(max_l, (long a, long b), return a > b ? a : b;, long)

def_gen(min_i, (int a, int b), return b > a ? a : b;, int)
def_gen(min_l, (long a, long b), return b > a ? a : b;, long)

#define max(a,b) \
    _Generic((a), \
            int: max_i, \
            long: max_l, \
            float: fmaxf, \
            double: fmax)(a,b)

#define min(a,b) \
    _Generic((a), \
            int: min_i, \
            long: min_l, \
            float: fminf, \
            double: fmin)(a,b)

#endif
