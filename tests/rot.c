#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "test.h"

int n;
complex float *x, *y;
int incx, incy;
const float c = 1, s = 1;

int prologue(int num) {
    int len_x, len_y;

    incx = 1;
    incy = 1;
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
    free(x);
    free(y);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    get_N_or_fail(argc, argv, &n);
    run_test(10, &prologue, &test_csrot, &epilogue, &pinfo);
    print_results("CSROT", n, &pinfo);

    return 0;
}
