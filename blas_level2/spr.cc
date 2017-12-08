#include "level2.h"

template <typename T>
void _cblas_spr(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n,
        const T alpha,
        const T *x, const int incx,
        T *a,
        cublasStatus_t spr_func(cublasHandle_t,
            cublasFillMode_t,
            int, const T *,
            const T*, int,
            T *),
        geam_t<T> geam_func,
        T geam_alpha = 1.0, T geam_beta = 0.0)
{
    T *gpu_a;
    const T *gpu_x;
    const int size_a = size(0, n, n+1, sizeof(*a))/2;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const struct objinfo *a_info, *x_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = n;
        cols_a = (n+1)/2;
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, rows_a, geam_func);
    } else {
        cols_a = n;
        rows_a = (n+1)/2;
        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);

    spr_func(b2c_handle, fillmode, 
            n, &alpha,
            gpu_x, incx,
            gpu_a);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!a_info)
        b2c_copy_from_gpu(a, gpu_a, size_a);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
}

DECLARE_CBLAS__SPR(s, float) {
    _cblas_spr(Layout, uplo,
            n, alpha, 
            x, incx,
            ap,
            &cublasSspr,
            &cublasSgeam);
}

DECLARE_CBLAS__SPR(d, double) {
    _cblas_spr(Layout, uplo,
            n, alpha, 
            x, incx,
            ap,
            &cublasDspr,
            &cublasDgeam);
}
