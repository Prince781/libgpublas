#pragma once
#include "runtime.h"
#include "common.h"
#include "lib/obj_tracker.h"
#include <assert.h>
#include <type_traits>
#include <iostream>

#if USE_OPENCL
extern cl_command_queue opencl_cmd_queue;
#endif

/**
 * 
 */
template <class T>
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
    // if the type is const
    template <class U, std::enable_if_t<std::is_same<U, const U>::value, int> = 0>
    cl_mem_flags get_mem_flags() { return CL_MEM_READ_ONLY; }

    // if the type is non-const
    template <class U, std::enable_if_t<!std::is_same<U, const U>::value, int> = 0>
    cl_mem_flags get_mem_flags() { return CL_MEM_READ_WRITE; }

#endif

    gpuptr(T *host_ptr, size_t size) : host_ptr(host_ptr), size(size), gpu_ptr(0), grabbed(false) {
        runtime_error_t err;
        extern size_t b2c_hits, b2c_misses;
#if USE_OPENCL
        extern cl_context opencl_ctx;
#endif
        objtracker_guard guard;

        if (size == 0) {
            size_t dummy_size = 32 * sizeof *host_ptr;  /* just pick some arbitrary size */
#if USE_CUDA
            err = runtime_malloc((void **)&this->gpu_ptr, dummy_size);
#elif USE_OPENCL
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags<T>(), dummy_size, NULL, &err);
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
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags<T>(), size, NULL, &err);
#endif

            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to allocate %zu B on device: %s\n",
                        size, runtime_error_string(err));
                abort();
            }
        } else if ((this->o_info = obj_tracker_objinfo_subptr((void *)host_ptr))) {
            // host_ptr is already shared with GPU
            assert(this->o_info->ptr == host_ptr);
            err = runtime_svm_unmap((void *)host_ptr);
            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to unmap %p from host: %s\n",
                        host_ptr, runtime_error_string(err));
                abort();
            }
#if USE_CUDA
            this->gpu_ptr = host_ptr;
#else
            // create a buffer that is backed by SVM
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags<T>() | CL_MEM_USE_HOST_PTR, size, (void *)host_ptr, &err);
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
            this->gpu_ptr = clCreateBuffer(opencl_ctx, this->get_mem_flags<T>() | CL_MEM_COPY_HOST_PTR, size, (void *)host_ptr, &err);
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

    // if the type is const, do nothing
    template <class U, std::enable_if_t<std::is_same<U, const U>::value, int> = 0>
    void cleanup_unmanaged() { }

    // if the type is non-const
    template <class U, std::enable_if_t<!std::is_same<U, const U>::value, int> = 0>
    void cleanup_unmanaged() {
        runtime_error_t err;

        // copy the GPU buffer back to host
        if (this->grabbed) {
#if USE_CUDA
            err = runtime_memcpy_dtoh(this->host_ptr, this->gpu_ptr, this->size);
#else
            err = clEnqueueReadBuffer(opencl_cmd_queue, this->gpu_ptr, CL_TRUE, 0, this->size, this->host_ptr, 0, NULL, NULL);
#endif
            if (runtime_is_error(err)) {
                writef(STDERR_FILENO, "blas2cuda: failed to copy %zu B from %p (GPU) ---> %p (CPU): %s\n", 
                        this->size, this->gpu_ptr, this->host_ptr, runtime_error_string(err));
                abort();
            }
        }
    }

    ~gpuptr() {
        runtime_error_t err = RUNTIME_ERROR_SUCCESS;
        objtracker_guard guard;

        if (!this->o_info) {
            this->cleanup_unmanaged<T>();
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

#if USE_CUDA
    operator T*() {
#else
    operator cl_mem() {
#endif
        this->grabbed = true;
        return this->gpu_ptr;
    }
};
