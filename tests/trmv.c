#include <stdio.h>
#include <stdlib.h>
#include "test.h"

char outfname[100];

int n;  /* cols/rows for mat_A, rows for vec_x */
float *mat_A, *vec_x;
const int incx = 1, incy = 1;

int prologue(int num) {
    int len_x;

    len_x = n * incx;

    if (!(mat_A = calloc(n, n * sizeof(*mat_A))))
        return -1;
    if (!(vec_x = calloc(len_x, sizeof(*vec_x)))) {
        free(mat_A);
        return -1;
    }

    /* initialize mat_A */
    for (int row = 0; row < n; ++row)
        for (int col = row; col < n; ++col)
            mat_A[row * n + col] = (row * n + col) % ((n * n) / 10);

    /* initialize vec_x */
    for (int i = 0; i < len_x; i += incx) {
        vec_x[i] = i % len_x;
    }
    return 0;
}

void test_trmv(void) {
    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans,
            CblasNonUnit /* unit triangle = all 1's */,
            n, mat_A, n,
            vec_x, incx);
}

int epilogue(int num) {
    if (num == 9) {
        FILE *fp;
        
        if (!(fp = fopen(outfname, "w"))) {
            perror("fopen");
            exit(1);
        }
        print_vector(vec_x, incx*n, fp);
        fclose(fp);
    }

    free(mat_A);
    free(vec_x);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    get_N_or_fail(argc, argv, &n);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(10, &prologue, &test_trmv, &epilogue, &pinfo);
    print_results("TRMV", n, &pinfo);

    return 0;
}
