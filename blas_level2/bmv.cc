#include "level2.h"

template <typename T>
static void _cblas_hbmv(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n, const int k,
        const T alpha,
        const T *a, const int lda,
        const T *x, const int incx,
        const T beta,
        T *y, const int incy,
        cublasStatus_t hbmv_func(cublasHandle_t,
            cublasFillMode_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const cublasFillMode_t fillmode = cu(uplo); 
    const T *gpu_a, *gpu_x;
    T *gpu_y;
    const int size_a = size(1, n - 1, lda, sizeof(*a));
    const int size_x = size(1, n - 1, incx, sizeof(*x));
    const int size_y = size(1, n - 1, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info, *a_info;
    int rows_a, cols_a;

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = lda;
        cols_a = n;
        gpu_a = transpose(a, size_a, &rows_a, &cols_a, lda, geam_func);
    } else {
        gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    }

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_a, a_info,
            NULL);
    gpu_y = (T *) b2c_place_on_gpu((void *) y, size_y, &y_info,
            (void *) gpu_a, a_info,
            (void *) gpu_x, x_info,
            NULL);

    call_kernel(
        hbmv_func(b2c_handle, fillmode,
                n, k,
                &alpha,
                gpu_a, lda,
                gpu_x, incx,
                &beta,
                gpu_y, incy)
    );

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__HBMV(c, float _Complex) {
    _cblas_hbmv(Layout, uplo,
            n, k,
            *(cuComplex *) alpha,
            (cuComplex *) a, lda,
            (cuComplex *) x, incx,
            *(cuComplex *) beta,
            (cuComplex *) y, incy,
            &cublasChbmv,
            &cublasCgeam);
}

DECLARE_CBLAS__HBMV(z, double _Complex) {
    _cblas_hbmv(Layout, uplo,
            n, k,
            *(cuDoubleComplex *) alpha,
            (cuDoubleComplex *) a, lda,
            (cuDoubleComplex *) x, incx,
            *(cuDoubleComplex *) beta,
            (cuDoubleComplex *) y, incy,
            &cublasZhbmv,
            &cublasZgeam);
}


DECLARE_CBLAS__SBMV(s, float) {
    _cblas_hbmv(Layout, uplo,
            n, k,
            alpha,
            a, lda,
            x, incx,
            beta,
            y, incy,
            &cublasSsbmv,
            &cublasSgeam);
}


DECLARE_CBLAS__SBMV(d, double) {
    _cblas_hbmv(Layout, uplo,
            n, k,
            alpha,
            a, lda,
            x, incx,
            beta,
            y, incy,
            &cublasDsbmv,
            &cublasDgeam);

}
