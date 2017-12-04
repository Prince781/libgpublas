#include <stdio.h>
#include <stdlib.h>
#include "test.h"

float *x, *y;
int n;
int incx, incy;

int prologue(int num) {
    int len_x, len_y;

    n = num;
    incx = num % 10;
    incy = num / 10;
    len_x = num * incx;
    len_y = num * incy;

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
    /* TODO: check for correctness */
    free(x);
    free(y);
    return 0;
}

int main() {
    struct perf_info pinfo;

    run_test(100, &prologue, &test_scopy, &epilogue, &pinfo);

    printf(" COPY: %ld s + %ld ns\n",
            pinfo.avg.tv_sec, pinfo.avg.tv_nsec);

    return 0;
}
