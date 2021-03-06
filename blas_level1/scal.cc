#include <cublas_v2.h>
#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level1.h"
#include "../blas2cuda.h"

extern cublasHandle_t b2c_handle;

template <typename A, typename V>
void _cblas_scal (const int n,
        const A a,
        V *x,
        const int incx,
        cublasStatus_t scal_func(cublasHandle_t, int, 
            const A *, 
            V *, int))
{
    V *gpu_x;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *x_info;

    gpu_x = (V *) b2c_place_on_gpu((void *) x, size_x, &x_info, NULL);

    call_kernel(scal_func(b2c_handle, n, &a, gpu_x, incx));

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!x_info)
        b2c_copy_from_gpu(x, gpu_x, size_x);

    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__SCAL(s, float, float) {
    _cblas_scal(n, a, x, incx, &cublasSscal);
}

DECLARE_CBLAS__SCAL(d, double, double) {
    _cblas_scal(n, a, x, incx, &cublasDscal);
}

DECLARE_CBLAS__SCAL(c, float _Complex, float _Complex) {
    _cblas_scal(n, cu(a), (cuComplex *) x, incx, &cublasCscal);
}

DECLARE_CBLAS__SCAL(z, double _Complex, double _Complex) {
    _cblas_scal(n, cu(a), (cuDoubleComplex *) x, incx, &cublasZscal);
}

DECLARE_CBLAS__SCAL(cs, float _Complex, float) {
    _cblas_scal(n, a, (cuComplex *)x, incx, &cublasCsscal);
}

DECLARE_CBLAS__SCAL(zd, double _Complex, double) {
    _cblas_scal(n, a, (cuDoubleComplex *) x, incx, &cublasZdscal);
}

