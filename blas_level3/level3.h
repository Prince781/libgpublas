#ifndef BLAS2CUDA_LEVEL3_H
#define BLAS2CUDA_LEVEL3_H
#include <complex.h>
#include "../blas2cuda.h"
#include "../cblas.h"
#include "../conversions.h"

void assign_dims(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
        int& rows_ref, int& cols_ref,
        const int ld,
        const int rows, const int cols);

#endif
