#ifndef BLAS2CUDA_H
#define BLAS2CUDA_H

#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

struct b2c_options {
    bool debug_execfail;
    bool debug_exec;
    bool trace_copy;
};

extern struct b2c_options b2c_options;
extern bool b2c_must_synchronize;

#endif
