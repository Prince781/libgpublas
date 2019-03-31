#ifndef BLAS2CUDA_H
#define BLAS2CUDA_H

#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#include "lib/obj_tracker.h"

struct b2c_options {
    bool debug_execfail;
    bool debug_exec;
    bool trace_copy;
};

extern struct b2c_options b2c_options;
extern bool b2c_must_synchronize;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new GPU buffer and copies the CPU buffer to it.
 * Returns the GPU buffer, which may be NULL on failure.
 * Free with cudaFree().
 */
void *b2c_copy_to_gpu(const void *hostbuf, size_t size);

/**
 * Copies an existing GPU buffer to an existing CPU buffer.
 */
void b2c_copy_from_gpu(void *hostbuf, const void *gpubuf, size_t size);

/**
 * If the buffer is on the CPU, copies it to the GPU and returns a GPU pointer.
 * {*info_in} is set to NULL. If the buffer is NULL, returns a new GPU buffer
 * and {*info_in} is set to NULL.
 * If the buffer is shared, returns the same pointer. {*info_in} points to the
 * object tracking information for this buffer.
 * If there is a fatal error, this function will abort.
 * The remaining arguments are each a pair (gpubuf, gpubuf_info) to perform cleanup on
 * in the event of an error. The list is terminated by a NULL pointer for a
 * gpubuf.
 */
void *b2c_place_on_gpu(void *hostbuf, 
                       size_t size,
                       const struct objinfo **info_in,
                       void *gpubuf2,
                       ...);

/**
 * If {info} is NULL, the GPU buffer will be freed with cudaFree(). Otherwise
 * nothing will happen.
 * If there is a fatal error, this function will abort.
 */
void b2c_cleanup_gpu_ptr(void *gpubuf, const struct objinfo *info);

#ifdef __cplusplus
};
#endif

#endif
