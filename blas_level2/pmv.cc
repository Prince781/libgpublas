#include "level2.h"

template <typename T>
static void _cblas_hpmv(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n, 
        const T alpha,
        const T *a,
        const T *x, const int incx,
        const T beta,
        T *y, const int incy,
        cublasStatus_t hpmv_func(cublasHandle_t, cublasFillMode_t,
            int, const T *,
            const T *,
            const T *, int,
            const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const cublasFillMode_t fillmode = cu(uplo);
    const T *gpu_a, *gpu_x;
    T *gpu_y;
    const int size_a = size(0, n, n+1, sizeof(*a))/2;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info, *a_info;
    int rows_a, cols_a;

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = n;
        cols_a = (n+1)/2;
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, rows_a, geam_func);
    } else {
        rows_a = n;
        cols_a = (n+1)/2;
        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);
    gpu_y = (T *) b2c_place_on_gpu((void *) y, size_y, &y_info,
            (void *) gpu_a, a_info,
            (void *) gpu_x, x_info,
            NULL);

    hpmv_func(b2c_handle, fillmode,
            n,
            &alpha,
            gpu_a, 
            gpu_x, incx,
            &beta,
            gpu_y, incy);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__HPMV(c, float _Complex) {
    _cblas_hpmv(Layout, uplo, n,
            *(cuComplex *) alpha,
            (cuComplex *) ap,
            (cuComplex *) x, incx,
            *(cuComplex *) beta,
            (cuComplex *) y, incy,
            &cublasChpmv,
            &cublasCgeam);
}

DECLARE_CBLAS__HPMV(z, float _Complex) {
    _cblas_hpmv(Layout, uplo, n,
            *(cuDoubleComplex *) alpha,
            (cuDoubleComplex *) ap,
            (cuDoubleComplex *) x, incx,
            *(cuDoubleComplex *) beta,
            (cuDoubleComplex *) y, incy,
            &cublasZhpmv,
            &cublasZgeam);
}

DECLARE_CBLAS__SPMV(s, float) {
    _cblas_hpmv(Layout, uplo, n,
            alpha,
            ap,
            x, incx,
            beta,
            y, incy,
            &cublasSspmv,
            &cublasSgeam);
}

DECLARE_CBLAS__SPMV(d, double) {
    _cblas_hpmv(Layout, uplo, n,
            alpha,
            ap,
            x, incx,
            beta,
            y, incy,
            &cublasDspmv,
            &cublasDgeam);
}

