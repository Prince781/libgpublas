#include "runtime.h"
#include "common.h"
#include <stdbool.h>

#if USE_OPENCL
#include "clext.h"

cl_context opencl_ctx;
cl_command_queue opencl_cmd_queue;
bool opencl_finegrained;

struct opencl_device {
    cl_device_id id;
    char *name;
    cl_device_type type;
    cl_device_svm_capabilities svm_capabilities;
};

struct opencl_platform {
    cl_platform_id id;
    char *name;
    cl_uint num_devices;
    struct opencl_device *devices;
};

struct opencl_platform *opencl_platforms;
cl_uint num_platforms;

struct opencl_platform *runtime_get_platforms(cl_int *err_in, cl_uint *nplatforms_in) {
    struct opencl_platform *platforms = NULL;
    cl_uint num_platforms = 0;
    cl_platform_id *platform_ids = NULL;
    cl_device_id *device_ids = NULL;

    if ((*err_in = clGetPlatformIDs(0, NULL, &num_platforms)) != CL_SUCCESS)
        goto end;
    platforms = calloc(num_platforms, sizeof *platforms);
    platform_ids = calloc(num_platforms, sizeof *platform_ids);
    if ((*err_in = clGetPlatformIDs(num_platforms, platform_ids, NULL)) != CL_SUCCESS)
        goto end;

    for (cl_uint p = 0; p < num_platforms; p++) {
        size_t platform_name_sz;
        platforms[p].id = platform_ids[p];

        if ((*err_in = clGetDeviceIDs(platforms[p].id, CL_DEVICE_TYPE_ALL, 0, NULL, &platforms[p].num_devices)) != CL_SUCCESS)
            goto end;
        platforms[p].devices = calloc(platforms[p].num_devices, sizeof *platforms[p].devices);
        device_ids = realloc(device_ids, platforms[p].num_devices * sizeof *device_ids);
        if ((*err_in = clGetDeviceIDs(platforms[p].id, CL_DEVICE_TYPE_ALL, platforms[p].num_devices, device_ids, NULL)) != CL_SUCCESS)
            goto end;

        if ((*err_in = clGetPlatformInfo(platforms[p].id, CL_PLATFORM_NAME, 0, NULL, &platform_name_sz)) != CL_SUCCESS)
            goto end;
        platforms[p].name = calloc(1, platform_name_sz);
        if ((*err_in = clGetPlatformInfo(platforms[p].id, CL_PLATFORM_NAME, platform_name_sz, platforms[p].name, NULL)) != CL_SUCCESS)
            goto end;
        writef(STDOUT_FILENO, "Platform [%u] = %s\n", p, platforms[p].name);

        for (cl_uint d = 0; d < platforms[p].num_devices; ++d) {
            size_t device_name_sz;

            platforms[p].devices[d].id = device_ids[d];

            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 0, NULL, &device_name_sz)) != CL_SUCCESS)
                goto end;
            platforms[p].devices[d].name = calloc(1, device_name_sz);
            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 
                            device_name_sz, 
                            platforms[p].devices[d].name, 
                            NULL)) != CL_SUCCESS)
                goto end;

            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_TYPE, 
                            sizeof platforms[p].devices[d].type, 
                            &platforms[p].devices[d].type, 
                            NULL)) != CL_SUCCESS)
                continue;

            writef(STDOUT_FILENO, "  Device [%u] = %s (%s)\n", d, platforms[p].devices[d].name, clDeviceTypeGetString(platforms[p].devices[d].type));
            writef(STDOUT_FILENO, " SVM capabilities:\n");

            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_SVM_CAPABILITIES, 
                            sizeof platforms[p].devices[d].svm_capabilities, 
                            &platforms[p].devices[d].svm_capabilities, 
                            NULL)) != CL_SUCCESS)
                continue;

            writef(STDOUT_FILENO, "    Course-grained buffer?: %s\n", platforms[p].devices[d].svm_capabilities & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? "yes" : "no");
            writef(STDOUT_FILENO, "    Fine-grained buffer?: %s\n", platforms[p].devices[d].svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? "yes" : "no");
            writef(STDOUT_FILENO, "    Fine-grained system?: %s\n", platforms[p].devices[d].svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? "yes" : "no");
            writef(STDOUT_FILENO, "    Atomics?: %s\n", platforms[p].devices[d].svm_capabilities & CL_DEVICE_SVM_ATOMICS ? "yes" : "no");
        }
    }

end:
    *nplatforms_in = num_platforms;
    free(platform_ids);
    platform_ids = NULL;
    free(device_ids);
    device_ids = NULL;

    return platforms;
}

void opencl_platform_cleanup(struct opencl_platform platform) {
    for (cl_uint d = 0; d < platform.num_devices; d++)
        free(platform.devices[d].name);
    free(platform.devices);
    free(platform.name);
}

#endif

runtime_error_t runtime_init(runtime_init_info_t info) {
#if USE_CUDA
    return cudaSuccess;
#else
    runtime_error_t err;
    opencl_platforms = runtime_get_platforms(&err, &num_platforms);

    if (runtime_is_error(err)) {
        writef(STDERR_FILENO, "blas2cuda: %s: failed to get OpenCL runtimes - %s\n", __func__, runtime_error_string(err));
        abort();
    }

    // create a context for a platform and device
    opencl_ctx = clCreateContext(
            (cl_context_properties[]){
                CL_CONTEXT_PLATFORM,
                (cl_context_properties) opencl_platforms[info.platform].id,
                0
            }, 1,
            &opencl_platforms[info.platform].devices[info.device].id, NULL,
            NULL, &err);

    if (runtime_is_error(err))
        return err;

    // create a command queue for the device
    opencl_cmd_queue = clCreateCommandQueueWithProperties(
            opencl_ctx, opencl_platforms[info.platform].devices[info.device].id,
            (cl_queue_properties[]) { 0 },
            &err);

    return err;
#endif
}

runtime_error_t runtime_fini(void) {
#if USE_CUDA
    return cudaSuccess;
#else
    for (cl_uint p = 0; p < num_platforms; p++)
        opencl_platform_cleanup(opencl_platforms[p]);
    free(opencl_platforms);
    opencl_platforms = NULL;
    num_platforms = 0;
#endif
    return RUNTIME_ERROR_SUCCESS;
}

void runtime_fatal_errmsg(runtime_error_t error_code, const char *domain) {
    if (runtime_is_error(error_code)) {
        writef(STDERR_FILENO, "libgpublas fatal error: %s - %s\n", domain, runtime_error_name(error_code));
        abort();
    }
}

runtime_error_t runtime_memcpy_htod(void *gpubuf, const void *hostbuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMemcpy(gpubuf, hostbuf, size, cudaMemcpyHostToDevice);
#else
    err = clEnqueueSVMMemcpy(opencl_cmd_queue,
            CL_TRUE, /* block */
            gpubuf, hostbuf, size,
            0, NULL, NULL);
#endif
    return err;
}

runtime_error_t runtime_memcpy_dtoh(void *hostbuf, const void *gpubuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMemcpy(hostbuf, gpubuf, size, cudaMemcpyDeviceToHost);
#else
    err = clEnqueueSVMMemcpy(opencl_cmd_queue,
            CL_TRUE, /* block */
            hostbuf, gpubuf, size,
            0, NULL, NULL);
#endif
    return err;
}

runtime_error_t runtime_malloc(void **gpubuf_in, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMalloc(gpubuf_in, size);
#else
    err = runtime_malloc_shared(gpubuf_in, size);
#endif
    return err;
}

runtime_error_t runtime_malloc_shared(void **sharedbuf_in, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaMallocManaged(sharedbuf_in, size, cudaMemAttachGlobal);
#else
    *sharedbuf_in = clSVMAlloc(opencl_ctx, CL_MEM_READ_WRITE, size, 0);
    err = RUNTIME_ERROR_SUCCESS;
#endif
    return err;
}

runtime_error_t runtime_free(void *gpubuf) {
    runtime_error_t err;
#if USE_CUDA
    err = cudaFree(gpubuf);
#else
    clSVMFree(opencl_ctx, gpubuf);
    err = RUNTIME_ERROR_SUCCESS;
#endif
    return err;
}

runtime_error_t runtime_svm_map(void *sharedbuf, size_t size) {
    runtime_error_t err;
#if USE_CUDA
    // sharedbuf was allocated with cudaMallocManaged, so we do nothing
    err = RUNTIME_ERROR_SUCCESS;
#else
    err = opencl_finegrained ? RUNTIME_ERROR_SUCCESS :
        clEnqueueSVMMap(opencl_cmd_queue, 
            CL_TRUE /* block until this completes */, 
            CL_MEM_READ_WRITE, sharedbuf, size, 0, NULL, NULL);
#endif
    return err;
}

runtime_error_t runtime_svm_unmap(void *sharedbuf) {
    runtime_error_t err;
#if USE_CUDA
    // sharedbuf was allocated with cudaMallocManaged, so we do nothing
    err = RUNTIME_ERROR_SUCCESS;
#else
    err = opencl_finegrained ? RUNTIME_ERROR_SUCCESS :
        clEnqueueSVMUnmap(opencl_cmd_queue,
            sharedbuf,
            0, NULL, NULL);
#endif
    return err;
}
