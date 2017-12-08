#include "level2.h"

template <typename T>
void _cblas_syr2(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n,
        const T alpha,
        const T *x, const int incx,
        const T *y, const int incy,
        T *a, const int lda,
        cublasStatus_t syr2_func(cublasHandle_t,
            cublasFillMode_t,
            int,
            const T *, 
            const T *, int,
            const T *, int, 
            T *, int),
        geam_t<T> geam_func,
        T geam_alpha = 1.0, T geam_beta = 0.0)
{
    T *gpu_a;
    const T *gpu_x, *gpu_y;
    const int size_a = size(0, n, lda, sizeof(*a));
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *a_info, *x_info, *y_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);

    if (Layout == CblasRowMajor) {
        T *gpu_a_trans;

        a_info = NULL;
        rows_a = n;
        cols_a = lda;

        gpu_a_trans = (T *) b2c_copy_to_gpu((void *) a, size_a);
        
        /* transpose A */
        geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                rows_a, cols_a,
                &geam_alpha,
                gpu_a_trans, lda,
                &geam_beta,
                NULL, 0,
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


    syr2_func(b2c_handle, fillmode,
            n, &alpha,
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

DECLARE_CBLAS__SYR2(s, float) {
    _cblas_syr2(Layout, uplo, n,
            alpha,
            x, incx,
            y, incy,
            a, lda,
            &cublasSsyr2,
            &cublasSgeam);
}

DECLARE_CBLAS__SYR2(d, double) {
    _cblas_syr2(Layout, uplo, n,
            alpha,
            x, incx,
            y, incy,
            a, lda,
            &cublasDsyr2,
            &cublasDgeam);

}
