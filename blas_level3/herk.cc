#include "../common.h"
#include "../cblas.h"
#include "../blas.h"
#include "../conversions.h"
#include "level3.h"
#include "../blas2cuda.h"
#include "../runtime-blas.h"

#if USE_CUDA
extern cublasHandle_t b2c_cublas_handle;
#else
extern cl_command_queue opencl_cmd_queue;
#endif

template <typename S, typename T>
void _b2c_herk(const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const S alpha,
        const T *a, const int lda,
        const S beta,
        T *c, const int ldc,
        cublasStatus_t herk_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            int, int,
            const S *,
            const T *, int,
            const S *,
            T *, int))
{
    const T *gpu_a;
    T *gpu_c;
    int rows_a, cols_a,
        rows_c, cols_c;
    int size_a, size_c;
    cublasFillMode_t cuplo = cu(uplo);
    cublasOperation_t ctrans = cu(trans);
    const struct objinfo *a_info, *c_info;

    assign_dims(CblasColMajor, trans, rows_a, cols_a, lda, k, n);

    size_a = size(0, rows_a, cols_a, sizeof(*a));
    
    rows_c = ldc;
    cols_c = n;
    size_c = size(0, rows_c, cols_c, sizeof(*c));

    gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
            (void *) gpu_a, &a_info,
            NULL);

    call_kernel(
        herk_func(b2c_handle,
                cuplo, ctrans,
                n, k,
                &alpha,
                gpu_a, lda,
                &beta,
                gpu_c, ldc)
    );

    
    runtime_fatal_errmsg(cudaGetLastError(), __func__);

    if (!c_info) {
        b2c_copy_from_gpu(c, gpu_c, size_c);
    }

    b2c_cleanup_gpu_ptr((void *) gpu_a, a_info);
    b2c_cleanup_gpu_ptr((void *) gpu_c, c_info);
}


F77_herk(c, float, float _Complex) {
    _b2c_herk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            (cuComplex *) a, *lda,
            *beta,
            (cuComplex *) c, *ldc,
            &cublasCherk);
}

F77_herk(z, double, double _Complex) {
    _b2c_herk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            (cuDoubleComplex *) a, *lda,
            *beta,
            (cuDoubleComplex *) c, *ldc,
            &cublasZherk);
}
