#include "level2.h"

template <typename T>
void _cblas_hpr2(const CBLAS_LAYOUT Layout,
        const CBLAS_UPLO uplo,
        const int n,
        const T alpha,
        const T *x, const int incx,
        const T *y, const int incy,
        T *a,
        cublasStatus_t hpr2_func(cublasHandle_t,
            cublasFillMode_t,
            int, const T *,
            const T *, int,
            const T *, int,
            T *),
        geam_t<T> geam_func,
        T geam_alpha, T geam_beta)
{
    T *gpu_a;
    const T *gpu_x, *gpu_y;
    const int size_a = size(0, n, n+1, sizeof(*a))/2;
    const int size_x = size(1, n-1, incx, sizeof(*x));
    const int size_y = size(1, n-1, incy, sizeof(*y));
    const struct objinfo *a_info, *x_info, *y_info;
    int rows_a, cols_a;
    const cublasFillMode_t fillmode = cu(uplo);

    if (Layout == CblasRowMajor) {
        T *gpu_a_trans;

        a_info = NULL;
        rows_a = n;
        cols_a = (n+1)/2;

        gpu_a_trans = (T *) b2c_copy_to_gpu((void *) a, size_a);
        
        /* transpose A */
        geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                rows_a, cols_a,
                &geam_alpha,
                gpu_a_trans, rows_a,
                &geam_beta,
                NULL, 0,
                gpu_a_trans, rows_a);
        
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

    hpr2_func(b2c_handle, fillmode, 
            n, &alpha,
            gpu_x, incx,
            gpu_y, incy,
            gpu_a);

    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    if (!a_info)
        b2c_copy_from_gpu(a, gpu_a, size_a);

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_x, x_info);
    b2c_cleanup_gpu_ptr((void *) gpu_y, y_info);
}

DECLARE_CBLAS__HPR2(c, float _Complex) {
    float _Complex _alpha = 1.0;
    float _Complex _beta = 1.0;

    _cblas_hpr2(Layout, uplo,
            n, cu(alpha), 
            (cuComplex *) x, incx,
            (cuComplex *) y, incy,
            (cuComplex *) ap,
            &cublasChpr2,
            &cublasCgeam,
            cu(_alpha),
            cu(_beta));
}

DECLARE_CBLAS__HPR2(z, double _Complex) {
    double _Complex _alpha = 1.0;
    double _Complex _beta = 1.0;

    _cblas_hpr2(Layout, uplo,
            n, cu(alpha), 
            (cuDoubleComplex *) x, incx,
            (cuDoubleComplex *) y, incy,
            (cuDoubleComplex *) ap,
            &cublasZhpr2,
            &cublasZgeam,
            cu(_alpha),
            cu(_beta));
}
