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


template <typename T>
void _b2c_syrk(const CBLAS_UPLO uplo,
        const CBLAS_TRANSPOSE trans,
        const int n, const int k,
        const T alpha,
        const T *a, const int lda,
        const T beta,
        T *c, const int ldc,
        cublasStatus_t syrk_func(cublasHandle_t,
            cublasFillMode_t, cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *,
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

    rows_c = ldc;
    cols_c = n;
    if (trans == CblasNoTrans) {
        rows_a = lda;
        cols_a = k;
    } else {
        rows_a = lda;
        cols_a = n;
    }


    size_a = size(0, rows_a, cols_a, sizeof(*a));
    size_c = size(0, rows_c, cols_c, sizeof(*c));

    gpu_a = (T *) b2c_place_on_gpu((void *) a, size_a, &a_info, NULL);
    gpu_c = (T *) b2c_place_on_gpu((void *) c, size_c, &c_info, 
            (void *) gpu_a, &a_info,
            NULL);

    call_kernel(
        syrk_func(b2c_handle,
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

F77_syrk(s, float) {
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            a, *lda,
            *beta,
            c, *ldc,
            &cublasSsyrk);
}

F77_syrk(d, double) {
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            *alpha,
            a, *lda,
            *beta,
            c, *ldc,
            &cublasDsyrk);
}

F77_syrk(c, float _Complex) {
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu(*alpha),
            (cuComplex *) a, *lda,
            cu(*beta),
            (cuComplex *) c, *ldc,
            &cublasCsyrk);
}

F77_syrk(z, double _Complex) {
    _b2c_syrk(c_uplo(*uplo), c_trans(*trans),
            *n, *k,
            cu(*alpha),
            (cuDoubleComplex *) a, *lda,
            cu(*beta),
            (cuDoubleComplex *) c, *ldc,
            &cublasZsyrk);
}
