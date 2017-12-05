#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "test.h"

char outfname[100];
int n;
complex float *x, *y;
const int incx = 1, incy = 1;
const float c = 1, s = 1;

int prologue(int num) {
    int len_x, len_y;

    len_x = n * incx;
    len_y = n * incy;

    if (!(x = malloc(len_x * sizeof(*x))))
        return -1;
    if (!(y = malloc(len_y * sizeof(*y)))) {
        free(x);
        return -1;
    }

    for (int i = 0; i < len_x; i += incx)
        x[i] = i + (i + 1) * I;
    for (int i = 0; i < len_y; i += incy)
        y[i] = i + (i + 1) * I;

    return 0;
}

void test_csrot(void) {
    cblas_csrot(n, x, incx, y, incy, c, s);
}

int epilogue(int num) {
    if (num == 9) {
        FILE *fp;
        
        if (!(fp = fopen(outfname, "w"))) {
            perror("fopen");
            exit(1);
        }
        print_vector(x, incx*n, fp);
        print_vector(y, incy*n, fp);
        fclose(fp);
    }

    free(x);
    free(y);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    get_N_or_fail(argc, argv, &n);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(10, &prologue, &test_csrot, &epilogue, &pinfo);
    print_results("CSROT", n, &pinfo);

    return 0;
}
