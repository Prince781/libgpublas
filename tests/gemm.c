#include <stdio.h>
#include <stdlib.h>
#include "test.h"

char outfname[100];

int n, m, k;
float *mat_A /* [m x k] */, *mat_B /* [k x n] */, *mat_C /* [m x n] */;
const float alpha = 1, beta = 1;

int prologue(int num) {
    m = n;
    k = n;

    if (!(mat_A = calloc(m, k * sizeof(*mat_A))))
        return -1;
    if (!(mat_B = calloc(k, n * sizeof(*mat_B)))) {
        free(mat_A);
        return -1;
    }
    if (!(mat_C = calloc(m, n * sizeof(*mat_C)))) {
        free(mat_A);
        free(mat_B);
        return -1;
    }

    /* fill mat_A */
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            mat_A[row * k + col] = (row * k + col) / k;
    /* fill mat_B */
    for (int row = 0; row < k; ++row)
        for (int col = 0; col < n; ++col)
            mat_B[row * n + col] = (row * n + col) % n;

    return 0;
}

void test_gemm(void) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            alpha, 
            mat_A, k,
            mat_B, n,
            beta,
            mat_C, m);
}

int epilogue(int num) {
    if (num == 9) {
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

    get_N_or_fail(argc, argv, &n);
    snprintf(outfname, sizeof outfname, "%s.out", argv[0]);
    run_test(10, &prologue, &test_gemm, &epilogue, &pinfo);
    print_results("GEMM", n, &pinfo);

    return 0;
}
