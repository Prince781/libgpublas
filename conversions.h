#ifndef CONVERSIONS_H
#define CONVERSIONS_H

// need to include runtime.h and cblas.h

#ifdef __cplusplus
#include <ccomplex>
#else
#include <complex.h>
#endif
#include "cblas.h"
#include "common.h"
#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline CBLAS_TRANSPOSE c_trans(char c) {
    switch (c) {
        case 'N':
        case 'n':
            return CblasNoTrans;
        case 'T':
        case 't':
            return CblasTrans;
        case 'C':
        case 'c':
            return CblasConjTrans;
        default:
            writef(STDERR_FILENO, "Invalid transpose op '%c'\n", c);
            abort();
            break;
    }
}

static inline CBLAS_SIDE c_side(char c) {
    switch (c) {
        case 'L':
        case 'l':
            return CblasLeft;
        case 'R':
        case 'r':
            return CblasRight;
        default:
            writef(STDERR_FILENO, "Invalid side op '%c'\n", c);
            abort();
            break;
    }
}

static inline CBLAS_UPLO c_uplo(char c) {
    switch (c) {
        case 'U':
        case 'u':
            return CblasUpper;
        case 'L':
        case 'l':
            return CblasLower;
        default:
            writef(STDERR_FILENO, "Invalid uplo op '%c'\n", c);
            abort();
            break;
    }
}

static inline CBLAS_DIAG c_diag(char c) {
    switch (c) {
        case 'U':
        case 'u':
            return CblasUnit;
        case 'N':
        case 'n':
            return CblasNonUnit;
        default:
            writef(STDERR_FILENO, "Invalid diag op '%c'\n", c);
            abort();
            break;
    }
}

#ifdef __cplusplus
};      // extern "C"
#endif


#ifdef __cplusplus

#if USE_CUDA
#include <cublas_api.h>

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

#elif USE_OPENCL
#include <clBLAS.h>

static inline clblasTranspose clb(CBLAS_TRANSPOSE trans) {
    switch (trans) {
        case CblasNoTrans:
            return clblasNoTrans;
        case CblasTrans:
            return clblasTrans;
        case CblasConjTrans:
            return clblasConjTrans;
        default:
            fprintf(stderr, "Invalid value for CBLAS_TRANSPOSE: %d\n", trans);
            abort();
            break;
    }
}

static inline clblasUplo clb(CBLAS_UPLO uplo) {
    switch (uplo) {
        case CblasUpper:
            return clblasUpper;
        case CblasLower:
            return clblasLower;
        default:
            fprintf(stderr, "Invalid value for CBLAS_UPLO: %d\n", uplo);
            abort();
            break;
    }
}

static inline clblasDiag clb(CBLAS_DIAG diag) {
    switch (diag) {
        case CblasNonUnit:
            return clblasNonUnit;
        case CblasUnit:
            return clblasUnit;
        default:
            fprintf(stderr, "Invalid value for CBLAS_DIAG: %d\n", diag);
            abort();
            break;
    }
}

static inline clblasSide clb(CBLAS_SIDE side) {
    switch (side) {
        case CblasLeft:
            return clblasLeft;
        case CblasRight:
            return clblasRight;
        default:
            fprintf(stderr, "Invalid value for CBLAS_SIDE: %d\n", side);
            abort();
            break;
    }
}

#endif // elif USE_OPENCL 

class scalar {
private:
    double _Complex value;
public:
    scalar(double i, double j = 0.0) : value(i + I * j) {}

    operator float() const { return creal(value); }
    operator double() const { return creal(value); }
    operator float _Complex() const { return ((float) creal(value)) + I * ((float) cimag(value)); }
    operator double _Complex() const { return value; }
#if USE_CUDA
    operator cuComplex() const { return cu((float _Complex) *this); }
    operator cuDoubleComplex() const { return cu((double _Complex) *this); }
#endif
};

#endif // __cplusplus

#endif
