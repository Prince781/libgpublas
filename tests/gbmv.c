#include <stdio.h>
#include <stdlib.h>
#include "test.h"

char outfname[100];
bool print_res = true;

int n;  /* cols for mat_A, rows for vec_x, vec_y */
int m;  /* rows for mat_A */
float *mat_A, *vec_x, *vec_y;
const int incx = 1, incy = 1;
const int kl = 2, ku = 2;
const float alpha = 1, beta = 1;

int prologue(int num) {
    int len_x, len_y;
    m = n;
    const int lda = m;

    len_x = n * incx;
    len_y = n * incy;

    if (!(vec_x = calloc(len_x, sizeof(*vec_x))))
        return -1;
    if (!(vec_y = calloc(len_y, sizeof(*vec_y)))) {
        free(vec_x);
        return -1;
    }
    if (!(mat_A = calloc(m, n * sizeof(*mat_A)))) {
        free(vec_x);
        free(vec_y);
        return -1;
    }

    /* initialize vec_x */
    for (int i = 0; i < len_x; i += incx) {
        vec_x[i] = i % len_x;
    }
    /* initialize vec_y */
    for (int i = 0; i < len_y; i += incy) {
        vec_y[i] = i % len_y;
    }
    /* initialize mat_A */
    for (int row = 0; row < m; ++row) {
        int k = kl - row;
        for (int col = max(0, row-kl); col < min(n, row+ku+1); col++) {
            mat_A[(k + col) + row * lda] = 1;
        }
    }

    return 0;
}

void test_gbmv(void) {
    cblas_sgbmv(CblasRowMajor, CblasNoTrans, 
            m, n, kl, ku, alpha, mat_A, n, 
            vec_x, incx,
            beta, vec_y, incy);
}

int epilogue(int num) {
    if (num == 9 && print_res) {
        FILE *fp;
        
        if (!(fp = fopen(outfname, "w"))) {
            perror("fopen");
            exit(1);
        }
        print_vector(vec_y, incy*n, fp);
        fclose(fp);
    }

    free(vec_x);
    free(vec_y);
    free(mat_A);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    parse_args(argc, argv, &n, &print_res);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(10, &prologue, &test_gbmv, &epilogue, &pinfo);
    print_perfinfo("GBMV", n, &pinfo);

    return 0;
}
