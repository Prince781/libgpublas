#include "level1.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void cblas_scopy (const int n,
        const float *x,
        const int incx, 
        float *y, 
        const int incy)
{
}
