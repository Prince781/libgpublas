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
    const cublasFillMode_t fillmode = (cublasFillMode_t) (uplo - CblasUpper);
    const T *gpu_a, *gpu_x;
    T *gpu_y;
    const int size_a = size(n, lda, sizeof(*a));
    const int size_x = size(n, incx, sizeof(*x));
    const int size_y = size(n, incy, sizeof(*y));
    const struct objinfo *x_info, *y_info, *a_info;
    int rows_a, cols_a;

    if (Layout == CblasRowMajor) {
        T *gpu_a_trans;

        a_info = NULL;
        rows_a = n;
        cols_a = lda;

        gpu_a_trans = (T *) b2c_copy_to_gpu((void *) a, size_a);
        
        /* transpose A */
        geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                rows_a, cols_a,
                &alpha,
                gpu_a_trans, lda,
                0,
                0, 0,
                gpu_a_trans, lda);
        
        if (cudaPeekAtLastError() != cudaSuccess)
            b2c_fatal_error(cudaGetLastError(), __func__);

        gpu_a = gpu_a_trans;
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

    hbmv_func(b2c_handle, fillmode,
            n, k,
            &alpha,
            a, lda,
            x, incx,
            &beta,
            y, incy);

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
