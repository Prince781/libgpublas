#include <stdio.h>
#include <stdlib.h>
#include "test.h"

char outfname[100];
bool print_res = true;

int n, m;
complex float *mat_A /* [m x m] */, *mat_B /* [m x n] */, *mat_C /* [m x n] */;
const complex float alpha = 1, beta = 0;

int prologue(int num) {
    m = n;

    if (!(mat_A = calloc(m, m * sizeof(*mat_A))))
        return -1;
    if (!(mat_B = calloc(m, n * sizeof(*mat_B)))) {
        free(mat_A);
        return -1;
    }
    if (!(mat_C = calloc(m, n * sizeof(*mat_C)))) {
        free(mat_A);
        free(mat_B);
        return -1;
    }

    /* fill mat_A (hermitian) */
    for (int row = 0; row < m; ++row)
        for (int col = row; col < m; ++col) {
            if (col == row)
                mat_A[idx(row, col, m, m)] = col;
            else {
                mat_A[idx(row, col, m, m)] = col + row * I;
                mat_A[idx(col, row, m, m)] = col - row * I;
            }
        }
    /* fill mat_B */
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col)
            mat_B[idx(row, col, m, n)] = (row * n + col) % n + (row % 10) * I;

    return 0;

}

void test_hemm(void) {
    /* C := alpha * A * B */
    cblas_chemm(CblasRowMajor, CblasLeft, CblasUpper,
            m, n,
            &alpha, 
            mat_A, m,
            mat_B, m,
            &beta,
            mat_C, m);
}

int epilogue(int num) {
    if (num == 9 && print_res) {
        FILE *fp;
        
        if (!(fp = fopen(outfname, "w"))) {
            perror("fopen");
            exit(1);
        }
        print_matrix(mat_C, m, n, fp);
        fclose(fp);
    }

    free(mat_A);
    free(mat_B);
    free(mat_C);
    return 0;
}

int main(int argc, char *argv[]) {
    struct perf_info pinfo;

    parse_args(argc, argv, &n, &print_res);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(N_TESTS, &prologue, &test_hemm, &epilogue, &pinfo);
    print_perfinfo("HEMM", n, &pinfo);

    return 0;
}
