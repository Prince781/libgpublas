#include "level2.h"

template <typename T>
void _cblas_gemv (const CBLAS_LAYOUT Layout,
        CBLAS_TRANSPOSE trans,
        const int m, const int n,
        const T alpha,
        const T *A, const int lda,
        const T *x, const int incx,
        const T beta,
        T *y, const int incy,
        cublasStatus_t gemv_func(cublasHandle_t,
            cublasOperation_t,
            int, int, 
            const T *,
            const T *, int,
            const T *, int,
            const T *,
            T *, int),
        geam_t<T> geam_func)
{
    const T *gpu_A, *gpu_x;
    T *gpu_y;
    const int size_A = size(0, n, lda, sizeof(*A));
    const int size_x = size(1, (trans == CblasNoTrans ? n : m) - 1, incx, sizeof(*x));
    const int size_y = size(1, (trans == CblasNoTrans ? n : m) - 1, incy, sizeof(*y));
    const struct objinfo *A_info, *x_info, *y_info;
    int rows_A, cols_A;
    cublasOperation_t op = cu(trans);

    if (Layout == CblasRowMajor && trans == CblasConjTrans) {
        /* create a new buffer that is the transpose matrix of A*/
        A_info = NULL;
        rows_A = m;
        cols_A = n;

        gpu_A = transpose(A, size_A, &rows_A, &cols_A, rows_A, geam_func);
    } else {
        gpu_A = (const T *) b2c_place_on_gpu((void *) A, size_A, &A_info, NULL);
    }

    gpu_x = (const T *) b2c_place_on_gpu((void *) x, size_x, &x_info,
            (void *) gpu_A, A_info,
            NULL);
    gpu_y = (T *) b2c_place_on_gpu(NULL, size_y, &y_info,
            (void *) gpu_A, A_info,
            (void *) gpu_x, x_info,
            NULL);

    call_kernel(
        gemv_func(b2c_handle, op,
                m, n,
                &alpha, 
                gpu_A, lda,
                gpu_x, incx,
                &beta,
                gpu_y, incy)
    );

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!y_info)
        b2c_copy_from_gpu(y, gpu_y, size_y);

    b2c_cleanup_gpu_ptr((void *) gpu_A, A_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);

}

DECLARE_CBLAS__GEMV(s, float) {
    _cblas_gemv(Layout, trans, 
            m, n, 
            alpha, 
            A, lda,
            x, incx,
            beta,
            y, incy, &cublasSgemv, &cublasSgeam);
}

DECLARE_CBLAS__GEMV(d, double) {
    _cblas_gemv(Layout, trans, 
            m, n, 
            alpha, 
            A, lda,
            x, incx,
            beta,
            y, incy, &cublasDgemv, &cublasDgeam);
}

DECLARE_CBLAS__GEMV(c, float _Complex) {
    _cblas_gemv(Layout, trans, 
            m, n, 
            cu(alpha), 
            (cuComplex *) A, lda,
            (cuComplex *) x, incx,
            cu(beta),
            (cuComplex *) y, incy, &cublasCgemv, &cublasCgeam);
}

DECLARE_CBLAS__GEMV(z, double _Complex) {
    _cblas_gemv(Layout, trans, 
            m, n, 
            cu(alpha), 
            (cuDoubleComplex *) A, lda,
            (cuDoubleComplex *) x, incx,
            cu(beta),
            (cuDoubleComplex *) y, incy, &cublasZgemv, &cublasZgeam);
}
