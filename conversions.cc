#include <complex.h>
#include "blas2cuda.h"
#include "conversions.h"

class scalar {
    double _Complex value;
    scalar(double i, double j = 0.0) : value(i + I * j) {}

    operator float() const { return creal(value); }
    operator double() const { return creal(value); }
    operator float _Complex() const { return ((float) creal(value)) + I * ((float) cimag(value)); }
    operator double _Complex() const { return value; }
};


template <typename T>
T *transpose(const T *host_a, int size_a, int *rows_a, int *cols_a, int lda, geam_t<T> geam)
{
    T *gpu_a_trans;
    T alpha = scalar(1.0);
    T beta = scalar(0.0);
    // int temp;

    gpu_a_trans = (T *) b2c_copy_to_gpu((void *) host_a, size_a);
    
    /* transpose A */
    geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            *rows_a, *cols_a,
            &alpha,
            gpu_a_trans, lda,
            0,
            &beta, 0,
            gpu_a_trans, lda);
    
    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    // swap
    /*
    temp = *cols_a;
    *cols_a = *rows_a;
    *rows_a = temp;
    */

    return gpu_a_trans;
}

template <typename T>
void transpose_in(T *gpu_a, int size_a, int *rows_a, int *cols_a, int lda, geam_t<T> geam)
{
    T alpha = scalar(1.0);
    T beta = scalar(0.0);
    // int temp;

    /* transpose A */
    geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            *rows_a, *cols_a,
            &alpha,
            gpu_a, lda,
            0,
            &beta, 0,
            gpu_a, lda);
    
    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    // swap
    /*
    temp = *cols_a;
    *cols_a = *rows_a;
    *rows_a = temp;
    */

    return gpu_a;
}


