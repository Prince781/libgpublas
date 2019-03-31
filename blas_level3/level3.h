#ifndef BLAS2CUDA_LEVEL3_H
#define BLAS2CUDA_LEVEL3_H
#include <complex.h>

static inline void assign_dims(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
        int& rows_ref, int& cols_ref,
        const int ld,
        const int rows, const int cols) {
    if (Layout == CblasColMajor) {
        if (trans == CblasNoTrans) {
            cols_ref = ld;
            rows_ref = cols;
        } else {
            cols_ref = ld;
            rows_ref = rows;
        }
    } else {
        if (trans == CblasNoTrans) {
            rows_ref = ld;
            cols_ref = rows;
        } else {
            rows_ref = ld;
            cols_ref = cols;
        }
    }
}

#endif
