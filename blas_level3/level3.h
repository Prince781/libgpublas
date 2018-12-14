#ifndef gpublas_LEVEL3_H
#define gpublas_LEVEL3_H
#include <complex.h>
#include "../gpublas.h"
#include "../cblas.h"
#include "../conversions.h"

void assign_dims(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
        int& rows_ref, int& cols_ref,
        const int ld,
        const int rows, const int cols);

#endif
