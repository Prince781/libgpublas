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

template <typename T>
using geam_t = cublasStatus_t (*)(cublasHandle_t,
            cublasOperation_t, cublasOperation_t,
            int, int,
            const T *,
            const T *, int,
            const T *,
            const T *, int,
            T *, int);

template <typename T>
T *transpose(const T *host_a, int size_a, int *rows_a, int *cols_a, int lda, geam_t<T> geam);

#endif
