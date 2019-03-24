// SVM but using cl_mem buffers
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

static void copy_file(const char *filename1, const char *filename2) {
    FILE *file_in, *file_out;
    char buf[4096];
    size_t nbytes = 0;

    file_in = fopen(filename1, "r");
    file_out = fopen(filename2, "w");

    while (fread(buf, sizeof buf, 1, file_in) == 1)
        fwrite(buf, sizeof buf, 1, file_out);

    while ((nbytes = fread(buf, 1, sizeof buf, file_in)) > 0)
        fwrite(buf, nbytes, 1, file_out);

    fclose(file_in);
    fclose(file_out);
}

static void dump_maps(unsigned lineno) {
    char fname_buf[64];

    snprintf(fname_buf, sizeof fname_buf, "maps-ln%u.txt", lineno);
    copy_file("/proc/self/maps", fname_buf);

    snprintf(fname_buf, sizeof fname_buf, "smaps-ln%u.txt", lineno);
    copy_file("/proc/self/smaps", fname_buf);
}

int main(int argc, char *argv[]) {
    cl_uint platformIdCount = 0;
    cl_platform_id *platformIds = NULL;
    cl_uint *deviceIdCounts = NULL;
    cl_device_id **deviceIds = NULL;
    int selected_platform = 0;
    int selected_device = 0;

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
        printf("Platform [%u] = %s", i, platform_name);
        clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, sizeof platform_name, platform_name, NULL);
        printf(", Version: %s\n", platform_name);

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

    char **prog_lines = NULL;
    size_t nlines = 0;

    cl_program program = 0;
    const char *prog_filename = "kernel.cl";

    cl_kernel kernel = 0;

    float *svm_A = NULL, 
          *svm_B = NULL, 
          *svm_C = NULL;

    size_t sz = 32768 * sizeof(*svm_A);

    cl_mem dev_A = 0, 
           dev_B = 0, 
           dev_C = 0;


    dump_maps(__LINE__);

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
    error = clBuildProgram(program, 1, &deviceIds[selected_platform][selected_device], "-Werror", NULL, NULL);

    if (error != CL_SUCCESS) {
        char buf[4096];
        fprintf(stderr, "clBuildProgram(): error - %s\n", clGetErrorString(error));
        fprintf(stderr, "compiler logs:\n\n");
        fprintf(stderr, "device #%d:\n", selected_device);
        error = clGetProgramBuildInfo(program, deviceIds[selected_platform][selected_device], CL_PROGRAM_BUILD_LOG, sizeof buf, buf, NULL);
        if (error != CL_SUCCESS)
            fprintf(stderr, "error getting build status: %s\n", clGetErrorString(error));
        else {
            buf[4095] = 0;
            fprintf(stderr, "%s\n", buf);
        }
        goto end;
    }

    // create kernel
    kernel = clCreateKernel(program, "add", &error);

    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel(): error - %s\n", clGetErrorString(error));
        goto end;
    }

    dump_maps(__LINE__);

    // initialize arguments 
    svm_A = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);
    dump_maps(__LINE__);

    svm_B = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);
    dump_maps(__LINE__);

    svm_C = clSVMAlloc(ctx, CL_MEM_READ_WRITE, sz, 0);
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_A, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_B, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_C, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    for (size_t i=0; i < sz / sizeof(*svm_A); ++i) {
        svm_A[i] = (((i+1)%4) * ((i*100)%37) - (i%3)*(i%19))% 100;
        svm_B[i] = (((i+1)%3) * ((i*100)%41) - (i%2)*(i%21)) % 100;
        svm_C[i] = 0;
    }
    dump_maps(__LINE__);

    printf("before kernel:\n");
    print_mat("A", svm_A, sz/sizeof(*svm_A));
    print_mat("B", svm_B, sz/sizeof(*svm_B));
    print_mat("C", svm_C, sz/sizeof(*svm_C));
    printf("\n");

    printf("operation: C = A + B\n\n");

    dump_maps(__LINE__);

    if ((error = clEnqueueSVMUnmap(queue, svm_A, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMUnmap(queue, svm_B, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMUnmap(queue, svm_C, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    // setup arguments to kernel, creating cl_mem buffers
    dev_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sz, svm_A, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    dev_B = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sz, svm_B, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    dev_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sz, svm_C, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clSetKernelArg(kernel, 0, sizeof dev_A, &dev_A)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clSetKernelArg(kernel, 1, sizeof dev_B, &dev_B)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clSetKernelArg(kernel, 2, sizeof dev_C, &dev_C)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);


    // call kernel
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (size_t[]){ sz/sizeof(*svm_A) }, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    // print results
    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_A, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_B, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_C, sz, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMMap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    printf("after kernel:\n");
    print_mat("A", svm_A, sz/sizeof(*svm_A));
    print_mat("B", svm_B, sz/sizeof(*svm_B));
    print_mat("C", svm_C, sz/sizeof(*svm_C));
    printf("\n\n");

    if ((error = clEnqueueSVMUnmap(queue, svm_A, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMUnmap(queue, svm_B, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

    if ((error = clEnqueueSVMUnmap(queue, svm_C, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueSVMUnmap(): error - %s\n", clGetErrorString(error));
        goto end;
    }
    dump_maps(__LINE__);

end:
    printf("cleaning up ...\n");
    clReleaseMemObject(dev_A);
    clReleaseMemObject(dev_B);
    clReleaseMemObject(dev_C);
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
