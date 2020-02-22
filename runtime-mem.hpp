#pragma once
#include "runtime.h"
#include "common.h"
#include "lib/obj_tracker.h"
#include <assert.h>
#include <type_traits>
#include <iostream>

extern size_t b2c_hits, b2c_misses;

#if USE_OPENCL
extern cl_command_queue opencl_cmd_queue;
extern cl_context opencl_ctx;
#endif

/**
 * RAII for GPU buffers.
 */
template <typename T, bool is_const = std::is_same<T, const T>::value>
class gpuptr {
private:
    T *host_ptr;
    size_t size;
#if USE_CUDA
    T *gpu_ptr;
#else
    cl_mem gpu_ptr;
#endif
    bool grabbed;
public:
    const struct objinfo *o_info;

#if USE_OPENCL
    cl_mem_flags get_mem_flags() { return is_const ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE; }
#endif

    gpuptr(T *host_ptr, size_t size) : host_ptr(host_ptr), size(size), gpu_ptr(0), grabbed(false), o_info(0) {
        runtime_error_t err;
        objtracker_guard guard;

        if (size == 0) {
            size_t dummy_size = 32 * sizeof *host_ptr;  /* just pick some arbitrary size */
#if USE_CUDA
            err = runtime_malloc((void **)&this->gpu_ptr, dummy_size);
#elif USE_OPENCL
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags(), dummy_size, NULL, &err);
#endif
            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to allocate %zu B on device: %s\n",
                        dummy_size, runtime_error_string(err));
                abort();
            }
            return;
        }

        if (!host_ptr) {
            // host_ptr is NULL, so create a brand new buffer
#if USE_CUDA
            err = runtime_malloc((void **)&this->gpu_ptr, size);
#else
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags(), size, NULL, &err);
#endif

            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to allocate %zu B on device: %s\n",
                        size, runtime_error_string(err));
                abort();
            }
        } else if ((this->o_info = obj_tracker_objinfo_subptr((void *)host_ptr))) {
            // host_ptr is already shared with GPU
#if USE_CUDA
            this->gpu_ptr = host_ptr;
#else
            // create a buffer that is backed by SVM
            assert(this->o_info->ptr == host_ptr && "cannot create CL buffer within CL buffer");
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags() | CL_MEM_USE_HOST_PTR, size, (void *)host_ptr, &err);
            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to create a buffer backed by %p: %s\n",
                        host_ptr, runtime_error_string(err));
                abort();
            }
#endif
            b2c_hits++;
        } else {
            // copy host_ptr contents over to GPU
#if USE_CUDA
            err = runtime_malloc((void **)&this->gpu_ptr, size);
#else
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags() | CL_MEM_COPY_HOST_PTR, size, (void *)host_ptr, &err);
#endif

            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to allocate %zu B on device: %s\n",
                        size, runtime_error_string(err));
                abort();
            }

#if USE_CUDA
            err = runtime_memcpy_htod((void *)this->gpu_ptr, host_ptr, size);
            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to copy %zu B from %p (CPU) ---> %p (GPU): %s\n",
                        size, host_ptr, this->gpu_ptr, runtime_error_string(err));
                abort();
            }
#endif

//            if (b2c_options.trace_copy) {
//                writef(STDOUT_FILENO, "blas2cuda: copy %zu B from %p (CPU) ---> %p (GPU)\n",
//                        size, host_ptr, this->gpu_ptr);
//            }
//
            b2c_misses++;
        }
    }

private:
    void cleanup_unmanaged() {
        if (!is_const) {
            runtime_error_t err;

            // copy the GPU buffer back to host
            if (this->grabbed) {
    #if USE_CUDA
                err = runtime_memcpy_dtoh(this->host_ptr, this->gpu_ptr, this->size);
    #else
                err = clEnqueueReadBuffer(opencl_cmd_queue, this->gpu_ptr, CL_TRUE, 0, this->size, (void *) this->host_ptr, 0, NULL, NULL);
    #endif
                if (runtime_is_error(err)) {
                    writef(STDERR_FILENO, "blas2cuda: failed to copy %zu B from %p (GPU) ---> %p (CPU): %s\n", 
                            this->size, this->gpu_ptr, this->host_ptr, runtime_error_string(err));
                    abort();
                }
            }
        }
    }

public:
    ~gpuptr() {
        runtime_error_t err = RUNTIME_ERROR_SUCCESS;
        objtracker_guard guard;

        if (!this->o_info) {
            if (this->size > 0)
                this->cleanup_unmanaged();
            // free the temporary GPU buffer
#if USE_CUDA
            err = runtime_free((void *) this->gpu_ptr);
#else
            err = clReleaseMemObject(this->gpu_ptr);
#endif
            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to free GPU buffer %p: %s\n",
                        this->gpu_ptr, runtime_error_string(err));
                abort();
            }
        } else
            // this is a managed object, so all we have to do is map it again
            err = runtime_svm_map((void *)this->host_ptr, this->o_info->size);

        if (runtime_is_error(err)) {
            writef(STDERR_FILENO, "blas2cuda: %s: failed to cleanup %p: %s\n", __func__,
                    this->host_ptr, runtime_error_name(err));
            abort();
        }
    }

    // TODO: return const cl_mem if underlying buffer is constant
#if USE_CUDA
    operator T*() {
#else
    operator cl_mem() {
#endif
        this->grabbed = true;
        return this->gpu_ptr;
    }
};
