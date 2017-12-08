#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <complex.h>
#include <cublas_v2.h>

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

#endif
