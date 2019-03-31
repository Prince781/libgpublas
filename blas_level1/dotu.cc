#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level1.h"
#include "../blas2cuda.h"

template <typename T>
static void _cblas_dotu_sub(const int n,
        const T *x,
        const int incx,
        const T *y,
        const int incy,
        T *dotu,
        cublasStatus_t dotu_func(cublasHandle_t, int, const T *, int, const T *, int, T *))
{
    const T *gpu_x, *gpu_y;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info;
    extern cublasHandle_t b2c_handle;

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);
    gpu_y = (T *) b2c_place_on_gpu((void *) y, size_y, &y_info, 
            (void *) gpu_x, x_info,
            NULL);

    call_kernel(dotu_func(b2c_handle, n, gpu_x, incx, gpu_y, incy, dotu));

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__DOTU(c, float _Complex) {
    _cblas_dotu_sub(n, (const cuComplex *) x, incx,
            (const cuComplex *) y, incy, (cuComplex *) dotu,
            &cublasCdotu);
}

DECLARE_CBLAS__DOTU(z, double _Complex) {
    _cblas_dotu_sub(n, (const cuDoubleComplex *) x, incx,
            (const cuDoubleComplex *) y, incy, (cuDoubleComplex *) dotu,
            &cublasZdotu);
}
