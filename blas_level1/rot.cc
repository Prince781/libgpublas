#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level1.h"
#include "../blas2cuda.h"

template <typename T, typename T2>
static void _cblas_rot (const int n,
        T *x, const int incx,
        T *y, const int incy,
        const T2 *c, const T2 *s,
        cublasStatus_t rot_func(cublasHandle_t, int, 
            T *, int, 
            T *, int, 
            const T2 *, const T2 *))
{
    T *gpu_x, *gpu_y;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info;
    extern cublasHandle_t b2c_handle;

    gpu_x = (T *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);
    gpu_y = (T *) b2c_place_on_gpu((void *) y, size_y, &y_info, 
            (void *) gpu_x, x_info,
            NULL);

    call_kernel(rot_func(b2c_handle, n, gpu_x, incx, gpu_y, incy, c, s));

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!x_info)
        b2c_copy_from_gpu(x, gpu_x, size_x);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__ROT(s, float, float) {
    _cblas_rot(n, x, incx, y, incy, &c, &s, &cublasSrot);
}

DECLARE_CBLAS__ROT(d, double, double) {
    _cblas_rot(n, x, incx, y, incy, &c, &s, &cublasDrot);
}

DECLARE_CBLAS__ROT(cs, float _Complex, float) {
    _cblas_rot(n, (cuComplex *) x, incx, (cuComplex *) y, incy, &c, &s,
            &cublasCsrot);
}

DECLARE_CBLAS__ROT(zd, double _Complex, double) {
    _cblas_rot(n, (cuDoubleComplex *) x, incx, (cuDoubleComplex *) y, incy, &c, &s,
            &cublasZdrot);
}
