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

static void print_mat(const char *name, float *m, int len) {
    const int mx = 14;  // max to show on
    printf("%s = ", name);
    for (int i=0; i<len; ++i) {
        if (i == 0) printf("{");
        else if (i == len - 1) printf("%8.2f }\n", m[i]);
        else if (i <= mx/2 || i >= len-mx/2) printf("%8.2f, ", m[i]);
        else if (i == mx/2+1) printf("   ...   ");
    }
}

int main(int argc, char *argv[]) {
    cl_uint platformIdCount = 0;
    cl_platform_id *platformIds = NULL;
    cl_uint deviceIdCount = 0;
    cl_device_id *deviceIds = NULL;

    cl_int error = 0;

    char **prog_lines = NULL;
    size_t nlines = 0;

    cl_program program;
    const char *prog_filename = "kernel.cl";

    cl_kernel kernel;

    float *svm_A = NULL, 
          *svm_B = NULL, 
          *svm_C = NULL;

    size_t sz = 32768 * sizeof(*svm_A);

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

        clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, sizeof device_name, device_name, NULL);
        printf("Device [%u] = %s\n", i, device_name);
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

    printf("loading and compiling `%s' ...\n", prog_filename);

    // load source
    nlines = load_file(prog_filename, &prog_lines);

    // create program
    program = clCreateProgramWithSource(ctx, nlines, (const char **) prog_lines, NULL, &error);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // compile and link program
    error = clBuildProgram(program, deviceIdCount, deviceIds, "-Werror", NULL, NULL);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram(): error - %s\n", clGetErrorString(error));
        fprintf(stderr, "compiler logs:\n\n");
        for (cl_uint d = 0; d < deviceIdCount; ++d) {
            char buf[4096];
            fprintf(stderr, "device #%d:\n", d);
            error = clGetProgramBuildInfo(program, deviceIds[d], CL_PROGRAM_BUILD_LOG, sizeof buf, buf, NULL);
            if (error != CL_SUCCESS)
                fprintf(stderr, "error getting build status: %s\n", clGetErrorString(error));
            else {
                buf[4095] = 0;
                fprintf(stderr, "%s\n", buf);
            }
        }
        goto end;
    }

    // create kernel
    kernel = clCreateKernel(program, "add", &error);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // initialize arguments 
    svm_A = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);
    svm_B = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);
    svm_C = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_A, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_B, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_C, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    for (size_t i=0; i < sz / sizeof(*svm_A); ++i) {
        svm_A[i] = (((i+1)%4) * ((i*100)%37) - (i%3)*(i%19))%100;
        svm_B[i] = (((i+1)%3) * ((i*100)%41) - (i%2)*(i%21))%100;
        svm_C[i] = 0;
    }

    printf("before kernel:\n");
    print_mat("A", svm_A, sz/sizeof(*svm_A));
    print_mat("B", svm_B, sz/sizeof(*svm_B));
    print_mat("C", svm_C, sz/sizeof(*svm_C));
    printf("\n");

    printf("operation: C = A + B\n\n");

    if ((error = clEnqueueSVMUnmap(queue, svm_A, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMUnmap(queue, svm_B, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMUnmap(queue, svm_C, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // setup arguments to kernel
    if ((error = clSetKernelArgSVMPointer(kernel, 0, svm_A)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArgSVMPointer(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clSetKernelArgSVMPointer(kernel, 1, svm_B)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArgSVMPointer(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clSetKernelArgSVMPointer(kernel, 2, svm_C)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArgSVMPointer(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // call kernel
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (size_t[]){ sz/sizeof(*svm_A) }, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    // print results
    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_A, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_B, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_C, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    printf("after kernel:\n");
    print_mat("A", svm_A, sz/sizeof(*svm_A));
    print_mat("B", svm_B, sz/sizeof(*svm_B));
    print_mat("C", svm_C, sz/sizeof(*svm_C));
    printf("\n\n");

    if ((error = clEnqueueSVMUnmap(queue, svm_A, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMUnmap(queue, svm_B, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    if ((error = clEnqueueSVMUnmap(queue, svm_C, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }

end:
    printf("cleaning up ...\n");
    clSVMFree(ctx, svm_A);
    clSVMFree(ctx, svm_B);
    clSVMFree(ctx, svm_C);
    checkCL("clReleaseContext", clReleaseContext(ctx));
    free(platformIds);
    free(deviceIds);
    for (size_t l = 0; l < nlines; ++l)
        free(prog_lines[l]);
    free(prog_lines);
    return 0;
}
