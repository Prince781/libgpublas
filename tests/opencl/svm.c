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

int read_int(const char *prompt, int min, int max) {
    int val = 0;
    int res = 0;
    do {
        printf("%s [%d-%d]: ", prompt, min, max);
        res = scanf("%d", &val);
    } while (res != EOF && res != 1 && !(val >= min && val <= max));
    return val;
}

int main(int argc, char *argv[]) {
    cl_uint platformIdCount = 0;
    cl_platform_id *platformIds = NULL;
    cl_uint *deviceIdCounts = NULL;
    cl_device_id **deviceIds = NULL;
    int selected_platform = 0;
    int selected_device = 0;

    size_t sz = 1024;
    void *ptr = NULL;

    cl_int error = 0;

    clGetPlatformIDs(0, NULL, &platformIdCount);
    platformIds = calloc(platformIdCount, sizeof *platformIds);
    clGetPlatformIDs(platformIdCount, platformIds, NULL);
    deviceIdCounts = calloc(platformIdCount, sizeof *deviceIdCounts);
    deviceIds = calloc(platformIdCount, sizeof *deviceIds);

    for (cl_uint i = 0; i < platformIdCount; ++i) {
        char platform_name[1024];
        clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCounts[i]);
        deviceIds[i] = calloc(deviceIdCounts[i], sizeof *deviceIds[i]);
        clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCounts[i], deviceIds[i], NULL);

        clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, sizeof platform_name, platform_name, NULL);
        printf("Platform [%u] = %s\n", i, platform_name);

        for (cl_uint d = 0; d < deviceIdCounts[i]; ++d) {
            char device_name[1024];
            cl_device_type device_type;

            clGetDeviceInfo(deviceIds[i][d], CL_DEVICE_NAME, sizeof device_name, device_name, NULL);
            clGetDeviceInfo(deviceIds[i][d], CL_DEVICE_TYPE, sizeof device_type, &device_type, NULL);
            printf("\tDevice [%u] = %s (%s)\n", d, device_name, clDeviceTypeGetString(device_type));
        }
    }

    if (platformIdCount == 0) {
        printf("No platforms found.\n");
        return 0;
    } else {
        selected_platform = read_int("Select a platform", 0, platformIdCount-1);
    }

    if (deviceIdCounts[selected_platform] == 0) {
        printf("No devices for that platform\n");
        return 0;
    } else {
        selected_device = read_int("Select a device", 0, deviceIdCounts[selected_platform]-1);
    }

    // create an OpenCL context
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platformIds[selected_platform],
        0
    };

    cl_context ctx = clCreateContext(
            contextProperties, deviceIdCounts[selected_platform],
            deviceIds[selected_platform], NULL,
            NULL, &error);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // create a command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(
            ctx, deviceIds[selected_platform][selected_device],
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
