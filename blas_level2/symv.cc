#include "level2.h"

template <typename T>
void _cblas_symv (const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n,
        const T alpha,
        const T *a, const int lda,
        const T *x, const int incx,
        const T beta,
        T *y, const int incy,
        cublasStatus_t symv_func(cublasHandle_t,
            cublasFillMode_t,
            int, const T *,
            const T *, int,
            const T *, int, const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_a, *gpu_x;
    T *gpu_y;
    const int size_a = size(0, n, lda, sizeof(*a));
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *a_info, *x_info, *y_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);

    if (Layout == CblasRowMajor) {
        a_info = NULL;
        rows_a = lda;
        cols_a = n;

        gpu_a = transpose(a, size_a, &rows_a, &cols_a, rows_a, geam_func);
    } else {
        rows_a = lda;
        cols_a = n;
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
        symv_func(b2c_handle, fillmode,
                cols_a, &alpha,
                gpu_a, rows_a,
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

DECLARE_CBLAS__SYMV(s, float) {
    _cblas_symv(Layout, uplo, n,
            alpha,
            a, lda,
            x, incx,
            beta,
            y, incy,
            &cublasSsymv,
            &cublasSgeam);
}

DECLARE_CBLAS__SYMV(d, double) {
    _cblas_symv(Layout, uplo, n,
            alpha,
            a, lda,
            x, incx,
            beta,
            y, incy,
            &cublasDsymv,
            &cublasDgeam);
}
