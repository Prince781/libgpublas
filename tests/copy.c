#include <stdio.h>
#include <stdlib.h>
#include "test.h"

float *x, *y;
int n;
int incx, incy;

int prologue(int num) {
    int len_x, len_y;

    incx = n % 10;
    incy = n / 10;
    len_x = n * incx;
    len_y = n * incy;

    if (!(x = malloc(len_x * sizeof(*x))))
        return -1;
    if (!(y = malloc(len_y * sizeof(*y)))) {
        free(x);
        return -1;
    }

    for (int i = 0; i < len_x; i += incx)
        x[i] = i + 1;

    return 0;
}

void test_scopy(void) {
    cblas_scopy(n, x, incx, y, incy);
}

int epilogue(int num) {
    free(x);
    free(y);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    get_N_or_fail(argc, argv, &n);
    run_test(10, &prologue, &test_scopy, &epilogue, &pinfo);
    print_results("COPY", n, &pinfo);

    return 0;
}
