#include <stdio.h>
#include <stdlib.h>
#include "test.h"

char outfname[100];
bool print_res = true;

int m;   /* rows (m), columns (n) */
const int n = 1;
double *mat_A /* [m x m] */, *mat_B /* m x n */;
const double alpha = 1;

int prologue(int num) {
    if (!(mat_A = calloc(m, m * sizeof(*mat_A))))
        return -1;
    if (!(mat_B = calloc(m, n * sizeof(*mat_B)))) {
        free(mat_A);
        return -1;
    }

    /* fill mat_A */
    for (int row = 0; row < m; ++row)
        for (int col = row; col < m; ++col) {
            mat_A[row * m + col] = (row * m + col) % 10;
        }

    /* fill mat_B */
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            mat_B[row * n + col] = (row * n + col) % 10;
        }

    return 0;
}

void test_trsm(void) {
    /* FIXME: segfault when m > 100 */
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
            CblasNonUnit,
            m, n,
            alpha,
            mat_A, m,
            mat_B, m);
}

int epilogue(int num) {
    if (num == 9 && print_res) {
        FILE *fp;
        
        if (!(fp = fopen(outfname, "w"))) {
            perror("fopen");
            exit(1);
        }
        print_matrix(mat_B, m, n, fp);
        fclose(fp);
    }

    free(mat_A);
    free(mat_B);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    parse_args(argc, argv, &m, &print_res);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(10, &prologue, &test_trsm, &epilogue, &pinfo);
    print_perfinfo("TRSM", m, &pinfo);

    return 0;
}
