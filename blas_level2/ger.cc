#include "level2.h"

template <typename T>
static void _cblas_ger(const CBLAS_LAYOUT Layout,
        const int m, const int n,
        const T alpha,
        const T *x, const int incx,
        const T *y, const int incy,
        T *a, const int lda,
        cublasStatus_t ger_func(cublasHandle_t,
            int, int,
            const T *,
            const T *, int,
            const T *, int,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_x, *gpu_y;
    T *gpu_a;
    const int size_x = size(n, incx, sizeof(*x));
    const int size_y = size(n, incy, sizeof(*y));
    const int size_a = size(n, lda, sizeof(*a));
    const struct objinfo *x_info, *y_info, *a_info;
    int rows_a, cols_a;

    if (Layout == CblasRowMajor) {
        T *gpu_a_trans;

        a_info = NULL;
        rows_a = n;
        cols_a = m;

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
    gpu_y = (const T *) b2c_place_on_gpu((void *) y, size_y, &y_info,
            (void *) gpu_a, a_info,
            (void *) gpu_x, x_info,
            NULL);

    ger_func(b2c_handle, m, n,
            &alpha, 
            gpu_x, incx,
            gpu_y, incy,
            gpu_a, lda);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!a_info)
        b2c_copy_from_gpu(a, gpu_a, size_a);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__GER(s, float) {
    _cblas_ger(Layout, m, n,
            alpha,
            x, incx,
            y, incy,
            a, lda,
            &cublasSger,
            &cublasSgeam);
}

DECLARE_CBLAS__GER(d, double) {
    _cblas_ger(Layout, m, n,
            alpha,
            x, incx,
            y, incy,
            a, lda,
            &cublasDger,
            &cublasDgeam);
}

DECLARE_CBLAS__GERC(c, float _Complex) {
    _cblas_ger(Layout, m, n,
            *(cuComplex *) alpha,
            (cuComplex *) x, incx,
            (cuComplex *) y, incy,
            (cuComplex *) a, lda,
            &cublasCgerc,
            &cublasCgeam);
}

DECLARE_CBLAS__GERC(z, double _Complex) {
    _cblas_ger(Layout, m, n,
            *(cuDoubleComplex *) alpha,
            (cuDoubleComplex *) x, incx,
            (cuDoubleComplex *) y, incy,
            (cuDoubleComplex *) a, lda,
            &cublasZgerc,
            &cublasZgeam);
}

DECLARE_CBLAS__GERU(c, float _Complex) {
    _cblas_ger(Layout, m, n,
            *(cuComplex *) alpha,
            (cuComplex *) x, incx,
            (cuComplex *) y, incy,
            (cuComplex *) a, lda,
            &cublasCgeru,
            &cublasCgeam);
}

DECLARE_CBLAS__GERU(z, double _Complex) {
    _cblas_ger(Layout, m, n,
            *(cuDoubleComplex *) alpha,
            (cuDoubleComplex *) x, incx,
            (cuDoubleComplex *) y, incy,
            (cuDoubleComplex *) a, lda,
            &cublasZgeru,
            &cublasZgeam);
}
