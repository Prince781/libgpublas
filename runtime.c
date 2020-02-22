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
    bool is_valid;
};

struct opencl_platform {
    cl_platform_id id;
    char *name;
    cl_uint num_devices;
    struct opencl_device *devices;
    bool is_valid;
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

    // get platform info
    for (cl_uint p = 0; p < num_platforms; p++) {
        struct opencl_platform *const curr_platform = &platforms[p];
        size_t platform_name_sz;

        curr_platform->id = platform_ids[p];

        if ((*err_in = clGetDeviceIDs(curr_platform->id, CL_DEVICE_TYPE_ALL, 0, NULL, &curr_platform->num_devices)) != CL_SUCCESS)
            goto end;
        curr_platform->devices = calloc(curr_platform->num_devices, sizeof *curr_platform->devices);
        device_ids = realloc(device_ids, curr_platform->num_devices * sizeof *device_ids);
        if ((*err_in = clGetDeviceIDs(curr_platform->id, CL_DEVICE_TYPE_ALL, curr_platform->num_devices, device_ids, NULL)) != CL_SUCCESS)
            continue;

        if ((*err_in = clGetPlatformInfo(curr_platform->id, CL_PLATFORM_NAME, 0, NULL, &platform_name_sz)) != CL_SUCCESS)
            continue;
        curr_platform->name = calloc(1, platform_name_sz);
        if ((*err_in = clGetPlatformInfo(curr_platform->id, CL_PLATFORM_NAME, platform_name_sz, curr_platform->name, NULL)) != CL_SUCCESS)
            continue;
        writef(STDOUT_FILENO, "Platform [%u] = %s\n", p, curr_platform->name);

        // get device info
        for (cl_uint d = 0; d < curr_platform->num_devices; ++d) {
            struct opencl_device *const curr_device = &curr_platform->devices[d];
            size_t device_name_sz;

            curr_device->id = device_ids[d];

            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 0, NULL, &device_name_sz)) != CL_SUCCESS)
                continue;
            curr_device->name = calloc(1, device_name_sz);
            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 
                            device_name_sz, 
                            curr_device->name, 
                            NULL)) != CL_SUCCESS)
                continue;

            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_TYPE, 
                            sizeof curr_device->type, 
                            &curr_device->type, 
                            NULL)) != CL_SUCCESS)
                continue;

            writef(STDOUT_FILENO, "  Device [%u] = %s (%s)\n", d, curr_device->name, clDeviceTypeGetString(curr_device->type));
            writef(STDOUT_FILENO, "   SVM capabilities:\n");

            if ((*err_in = clGetDeviceInfo(device_ids[d], CL_DEVICE_SVM_CAPABILITIES, 
                            sizeof curr_device->svm_capabilities, 
                            &curr_device->svm_capabilities, 
                            NULL)) != CL_SUCCESS)
                continue;
            
            curr_device->is_valid = true;

            writef(STDOUT_FILENO, "    Course-grained buffer?: %s\n", curr_device->svm_capabilities & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? "yes" : "no");
            writef(STDOUT_FILENO, "    Fine-grained buffer?: %s\n", curr_device->svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? "yes" : "no");
            writef(STDOUT_FILENO, "    Fine-grained system?: %s\n", curr_device->svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? "yes" : "no");
            writef(STDOUT_FILENO, "    Atomics?: %s\n", curr_device->svm_capabilities & CL_DEVICE_SVM_ATOMICS ? "yes" : "no");
        }

        curr_platform->is_valid = true;
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
    for (cl_uint d = 0; platform.devices && d < platform.num_devices; d++)
        free(platform.devices[d].name);
    free(platform.devices);
    free(platform.name);
}

#endif /* USE_OPENCL */

runtime_error_t runtime_init(runtime_init_info_t info) {
#if USE_CUDA
    return cudaSuccess;
#else
    runtime_error_t err;
    bool have_platform = false;
    opencl_platforms = runtime_get_platforms(&err, &num_platforms);

    for (cl_uint p = 0; p < num_platforms; ++p)
        if (opencl_platforms[p].is_valid) {
            have_platform = true;
            break;
        }

    if (!have_platform) {
        writef(STDERR_FILENO, "blas2cuda: %s: failed to get OpenCL platforms - %s\n", __func__, runtime_error_string(err));
        abort();
    }

    // create a context for a platform and device
    struct opencl_platform *selected_platform = NULL;
    struct opencl_device *selected_device = NULL;
    for (cl_uint p = 0; p < num_platforms && !selected_device; p++) {
        struct opencl_platform *const curr_platform = &opencl_platforms[p];
        if (!curr_platform->is_valid)
            continue;
        for (cl_uint d = 0; d < curr_platform->num_devices && !selected_device; d++) {
            struct opencl_device *const curr_device = &curr_platform->devices[d];
            if (!curr_device->is_valid)
                continue;
            opencl_ctx = clCreateContext(
                    (cl_context_properties[]){
                        CL_CONTEXT_PLATFORM,
                        (cl_context_properties) curr_platform->id,
                        0
                    }, 1,
                    &curr_device->id, NULL,
                    NULL, &err);
            if (runtime_is_error(err))
                return err;
            // we will break now that opencl_ctx has been initialized
            selected_device = curr_device;
            selected_platform = curr_platform;
            if (p != info.platform)
                writef(STDERR_FILENO, "blas2cuda: %s: WARNING: could not select platform #%d (%s)\n", __func__, info.platform, curr_platform->name);
            if (d != info.device)
                writef(STDERR_FILENO, "blas2cuda: %s: WARNING: could not select device #%d (%s)\n", __func__, info.device, curr_device->name);
        }
    }

    if (!selected_device) {
        writef(STDERR_FILENO, "blas2cuda: %s: FATAL: could not select a device\n", __func__);
        return err;
    }

    // create a command queue for the device
    opencl_cmd_queue = clCreateCommandQueueWithProperties(
            opencl_ctx, selected_device->id,
            (cl_queue_properties[]) { 0 },
            &err);
    
    if (!runtime_is_error(err))
        writef(STDOUT_FILENO, "blas2cuda: %s: selected %s [%s]\n", 
               __func__, selected_platform->name, selected_device->name);

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
