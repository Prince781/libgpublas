#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#include "clext.h"
#include "util.h"

static void checkCL(const char *function, cl_int err) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s(): error - %s\n", function, clGetErrorString(err));
    }
}

int main(int argc, char *argv[]) {
    cl_uint platformIdCount = 0;
    cl_platform_id *platformIds = NULL;
    cl_uint deviceIdCount = 0;
    cl_device_id *deviceIds = NULL;

    size_t sz = 1024;
    void *ptr = NULL;

    cl_int error = 0;

    clGetPlatformIDs(0, NULL, &platformIdCount);
    platformIds = calloc(platformIdCount, sizeof *platformIds);
    clGetPlatformIDs(platformIdCount, platformIds, NULL);

    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
    deviceIds = calloc(deviceIdCount, sizeof *deviceIds);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds, NULL);

    for (cl_uint i = 0; i < platformIdCount; ++i) {
        char platform_name[1024];

        clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, sizeof platform_name, platform_name, NULL);
        printf("Platform [%u] = %s\n", i, platform_name);
    }

    for (cl_uint i = 0; i < deviceIdCount; ++i) {
        char device_name[1024];
        cl_device_type device_type;

        clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, sizeof device_name, device_name, NULL);
        clGetDeviceInfo(deviceIds[i], CL_DEVICE_TYPE, sizeof device_type, &device_type, NULL);
        printf("Device [%u] = %s (%s)\n", i, device_name, clDeviceTypeGetString(device_type));
    }

    // create an OpenCL context
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platformIds[0],
        0
    };

    cl_context ctx = clCreateContext(
            contextProperties, deviceIdCount,
            deviceIds, NULL,
            NULL, &error);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // create a command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(
            ctx, deviceIds[0],
            (cl_queue_properties[]) { 0 },
            &error);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueueWithProperties(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // create shared virtual memory
    ptr = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);

    // make it accessible (assuming we don't have fine-grained buffer access
    error = clEnqueueSVMMap(queue, 
            CL_TRUE /* block until this completes */, 
            CL_MEM_READ_WRITE, ptr, sz, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    *(int *)ptr = 10;

    printf("*(int *)ptr = %d\n", *(int *)ptr);

    error = clEnqueueSVMUnmap(queue,
            ptr,
            0, NULL, NULL);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clSVMAlloc(): error - %s\n", clGetErrorString(error));
        goto end;
    }

end:
    clSVMFree(ctx, ptr);
    checkCL("clReleaseContext", clReleaseContext(ctx));
    free(platformIds);
    free(deviceIds);
    return 0;
}
