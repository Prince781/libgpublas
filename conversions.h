#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <complex.h>
#include <cublas_api.h>
#include "cblas.h"

static inline cuComplex cu(float _Complex f) {
    return (cuComplex) { .x = crealf(f), .y = cimagf(f) };
}

static inline cuComplex cu(float r, float i) {
    return (cuComplex) { .x = r, .y = i };
}

static inline cuDoubleComplex cu(double _Complex d) {
    return (cuDoubleComplex) { .x = creal(d), .y = cimag(d) };
}

static inline cuDoubleComplex cu(double r, double i) {
    return (cuDoubleComplex) { .x = r, .y = i };
}

static inline cublasOperation_t cu(CBLAS_TRANSPOSE trans) {
    switch (trans) {
        case CblasNoTrans:
            return CUBLAS_OP_N;
        case CblasTrans:
            return CUBLAS_OP_T;
        case CblasConjTrans:
            return CUBLAS_OP_C;
        default:
            fprintf(stderr, "Invalid value for enum: %d\n", trans);
            abort();
            break;
    }
}

static inline cublasFillMode_t cu(CBLAS_UPLO uplo) {
    switch (uplo) {
        case CblasUpper:
            return CUBLAS_FILL_MODE_UPPER;
        case CblasLower:
            return CUBLAS_FILL_MODE_LOWER;
        default:
            fprintf(stderr, "Invalid value for enum: %d\n", uplo);
            abort();
            break;
    }
}

static inline cublasDiagType_t cu(CBLAS_DIAG diag) {
    switch (diag) {
        case CblasNonUnit:
            return CUBLAS_DIAG_NON_UNIT;
        case CblasUnit:
            return CUBLAS_DIAG_UNIT;
        default:
            fprintf(stderr, "Invalid value for enum: %d\n", diag);
            abort();
            break;
    }
}

static inline cublasSideMode_t cu(CBLAS_SIDE side) {
    switch (side) {
        case CblasLeft:
            return CUBLAS_SIDE_LEFT;
        case CblasRight:
            return CUBLAS_SIDE_RIGHT;
        default:
            fprintf(stderr, "Invalid value for enum: %d\n", side);
            abort();
            break;
    }
}

template <typename T>
using geam_t = cublasStatus_t (*)(cublasHandle_t,
            cublasOperation_t, cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *,
            const T *, int,
            T *, int);

class scalar {
private:
    double _Complex value;
public:
    scalar(double i, double j = 0.0) : value(i + I * j) {}

    operator float() const { return creal(value); }
    operator double() const { return creal(value); }
    operator float _Complex() const { return ((float) creal(value)) + I * ((float) cimag(value)); }
    operator double _Complex() const { return value; }
    operator cuComplex() const { return cu((float _Complex) *this); }
    operator cuDoubleComplex() const { return cu((double _Complex) *this); }
};


template <typename T>
T *transpose(const T *host_a, int size_a, int *rows_a, int *cols_a, int lda, geam_t<T> geam_func)
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
            &beta,
            gpu_a_trans, lda,
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
void transpose_in(T *gpu_a, int size_a, int *rows_a, int *cols_a, int lda, geam_t<T> geam_func)
{
    T alpha = scalar(1.0);
    T beta = scalar(0.0);
    // int temp;

    /* transpose A */
    geam_func(b2c_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            *rows_a, *cols_a,
            &alpha,
            gpu_a, lda,
            &beta,
            gpu_a, lda,
            gpu_a, lda);
    
    if (cudaPeekAtLastError() != cudaSuccess)
        b2c_fatal_error(cudaGetLastError(), __func__);

    // swap
    /*
    temp = *cols_a;
    *cols_a = *rows_a;
    *rows_a = temp;

    return gpu_a;
    */
}

#endif
