#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test.h"

char outfname[100];
bool print_res = true;

int n;
float *x, *y;
const int incx = 1, incy = 1;
float result;

int prologue(int num) {
    int len_x, len_y;

    len_x = n * incx;
    len_y = n * incy;

    if (!(x = calloc(len_x, sizeof(*x))))
        return -1;
    if (!(y = calloc(len_y, sizeof(*y)))) {
        free(x);
        return -1;
    }

    for (int i = 0; i < len_x; i += incx)
        x[i] = i + 1;
    for (int i = 0; i < len_y; i += incy)
        y[i] = i + 1;

    return 0;
}

void test_dsdot(void) {
    result = cblas_dsdot(n, x, incx, y, incy);
}

int epilogue(int num) {
    if (num == 9 && print_res) {
        FILE *fp;
        
        if (!(fp = fopen(outfname, "w"))) {
            perror("fopen");
            exit(1);
        }
        print_scalar(result, fp);
        fclose(fp);
    }

    free(x);
    free(y);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    parse_args(argc, argv, &n, &print_res);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(N_TESTS, &prologue, &test_dsdot, &epilogue, &pinfo);
    print_perfinfo("DSDOT", n, &pinfo);

    return 0;
}
