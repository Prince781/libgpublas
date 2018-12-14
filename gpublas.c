#include "gpublas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#include <signal.h>

#define GPUBLAS_OPTIONS "GPUBLAS_OPTIONS"

#include "lib/oracle.h"

static void *alloc_managed(size_t request);
static void *calloc_managed(size_t nmemb, size_t size);
static void *realloc_managed(void *managed_ptr, size_t request);
static void free_managed(void *managed_ptr);
static size_t get_size_managed(void *managed_ptr);

static bool b2c_initialized = false;

struct objmngr gpublas_manager = {
    .ctor = alloc_managed,
    .cctor = calloc_managed,
    .realloc = realloc_managed,
    .dtor = free_managed,
    .get_size = get_size_managed
};

static size_t hits = 0;
static size_t misses = 0;
static size_t total_managed_mem = 0;    /* in bytes */

bool b2c_must_synchronize = false;

struct b2c_options b2c_options = { false, false, false };

#if USE_CUDA

static bool cublas_initialized = false;
cublasHandle_t b2c_handle;
void init_cublas(void);

#elif USE_OPENCL
static bool opencl_initialized = false;

cl_context cl_ctx;
cl_command_queue cl_queue;

cl_uint num_platforms = 0;
cl_platform_id *platformIds = NULL;
cl_uint *num_devices = NULL;
cl_device_id **deviceIds = NULL;

void init_opencl(void);
#endif

void b2c_print_help(void) {
    writef(STDERR_FILENO, 
            "gpublas options (set "GPUBLAS_OPTIONS"):\n"
            "   You can chain these options with a semicolon (;)\n"
            "   help            -- print help\n"
            "   debug_execfail  -- debug kernel failures\n"
            "   debug_exec      -- debug kernel invocations\n"
            "   trace_copy      -- trace copies between CPU and GPU\n"
            "   heuristic=<val> -- one of: 'random', 'true', 'false', or:\n"
            "                      'oracle:<filename>', where <filename> is\n"
            "                      the name of an object trace\n");
}

static void set_options(void) {
    /* TODO: use secure_getenv() ? */
    char *options = getenv(GPUBLAS_OPTIONS);
    char *saveptr = NULL;
    char *option = NULL;
    bool help = false;

    if (!options)
        return;

    option = strtok_r(options, ";", &saveptr);
    while (option != NULL) {
        if (strcmp(option, "help") == 0) {
            if (!help) {
                b2c_print_help();
                help = true;
            }
        } 
        else if (strcmp(option, "debug_execfail") == 0)
            b2c_options.debug_execfail = true;
        else if (strcmp(option, "debug_exec") == 0)
            b2c_options.debug_exec = true;
        else if (strcmp(option, "trace_copy") == 0)
            b2c_options.trace_copy = true;
        else if (strncmp(option, "heuristic=", 10) == 0) {
            char *hnum = strchr(option, '=');
            if (hnum) {
                bool set = false;

                hnum++;
                if (strncmp(hnum, "random", sizeof("random") - 1) == 0) {
                    hfunc = H_RANDOM;
                    set = true;
                } else if (strncmp(hnum, "true", sizeof("true") - 1) == 0) {
                    hfunc = H_TRUE;
                    set = true;
                } else if (strncmp(hnum, "false", sizeof("false") - 1) == 0) {
                    hfunc = H_FALSE;
                    set = true;
                } else if (strncmp(hnum, "oracle:", sizeof("oracle:") - 1) == 0) {
                    hfunc = H_ORACLE;
                    set = true;
                }

                if (set) {
                    writef(STDERR_FILENO, "gpublas: selecting heuristic %s\n", hnum);
                } else {
                    writef(STDERR_FILENO, "gpublas: unsupported heuristic '%s'\n", hnum);
                    abort();
                }

                if (hfunc == H_ORACLE) {
                    char *filename = strchr(hnum, ':') + 1;
                    
                    if (!oracle_load_file(filename)) {
                        writef(STDERR_FILENO, "gpublas:oracle: failed to load '%s':%m\n", filename);
                        abort();
                    }
                }
            }
        } else {
            writef(STDERR_FILENO, "gpublas: unknown option '%s'. Set "GPUBLAS_OPTIONS"=help.\n", option);
        }
        option = strtok_r(NULL, ";", &saveptr);
    }
}

#if USE_CUDA
void init_cublas(void) {
    static bool inside = false;
    if (!cublas_initialized && !inside) {
        inside = true;
        switch (cublasCreate(&b2c_handle)) {
            case CUBLAS_STATUS_SUCCESS:
                {
                    pid_t tid = syscall(SYS_gettid);
                    writef(STDOUT_FILENO, "gpublas: initialized cuBLAS on thread %d\n", tid);
                }
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                writef(STDERR_FILENO, "gpublas: failed to allocate resources\n");
                break;
            case CUBLAS_STATUS_NOT_INITIALIZED:
            default:
                writef(STDERR_FILENO, "gpublas: failed to initialize cuBLAS\n");
                abort();
                break;
        }

        /* get device properties */
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        for (int i=0; i < num_devices; i++) {
            struct cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            writef(STDOUT_FILENO, "CUDA device #%d {\n", i+1);
            writef(STDOUT_FILENO, "  name: %s\n", prop.name);
            writef(STDOUT_FILENO, "  total global memory: %zu\n", prop.totalGlobalMem);
            writef(STDOUT_FILENO, "  supports managed memory?: %s\n", prop.managedMemory ? "true" : "false");
            writef(STDOUT_FILENO, "  concurrent managed access?: %s\n", prop.concurrentManagedAccess ? "true" : "false");
            writef(STDOUT_FILENO, "}\n");

            b2c_must_synchronize |= !prop.concurrentManagedAccess;
        }

        cublas_initialized = true;
        inside = false;
    }
}
#elif USE_OPENCL
/* generated from https://github.com/WanghongLin/miscellaneous/blob/master/tools/clext.py */
const char* clGetErrorString(cl_int errorCode) {
	switch (errorCode) {
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
		case -69: return "CL_INVALID_PIPE_SIZE";
		case -70: return "CL_INVALID_DEVICE_QUEUE";
		case -71: return "CL_INVALID_SPEC_ID";
		case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
		case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
		case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
		case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
		case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
		case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
		case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
		case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
		case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
		case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
		case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
		case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
		default: return "CL_UNKNOWN_ERROR";
	}
}

void init_opencl(void) {
    static bool inside = false;

    if (inside || opencl_initialized)
        return;
    inside = true;
    obj_tracker_internal_enter();

    cl_int error = 0;

    // get platforms
    clGetPlatformIDs(0, NULL, &num_platforms);
    platformIds = calloc(num_platforms, sizeof *platformIds);
    deviceIds = calloc(num_platforms, sizeof *deviceIds);
    num_devices = calloc(num_platforms, sizeof *num_devices);
    clGetPlatformIDs(num_platforms, platformIds, NULL);

    for (cl_uint p = 0; p < num_platforms; ++p) {
        char platform_name[128];

        clGetPlatformInfo(platformIds[p], CL_PLATFORM_NAME, sizeof platform_name, platform_name, NULL);
        writef(STDOUT_FILENO, "Platform #%u: %s\n", p+1, platform_name);

        clGetDeviceIDs(platformIds[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices[p]);
        deviceIds[p] = calloc(num_devices[p], sizeof *deviceIds[p]);
        clGetDeviceIDs(platformIds[p], CL_DEVICE_TYPE_ALL, num_devices[p], deviceIds[p], NULL);
        for (cl_uint d = 0; d < num_devices[p]; ++d) {
            char device_name[128];

            clGetDeviceInfo(deviceIds[p][d], CL_DEVICE_NAME, sizeof device_name, device_name, NULL);
            writef(STDOUT_FILENO, "\t1. %s%s\n", device_name, (p == 0 && d == 0) ? " (selected)" : "");
        }
    }

    cl_ctx = clCreateContext(
            (cl_context_properties[]) {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties) platformIds[0],
                0
            }, 
            num_devices[0], deviceIds[0], NULL, NULL, &error);

    b2c_fatal_error(error, "clCreateContext()");

    obj_tracker_internal_leave();
    opencl_initialized = true;
    inside = false;
}
#endif

void *b2c_copy_to_gpu(const void *hostbuf, size_t size, b2c_runtime_error_t *err)
{
    void *gpubuf = NULL;

    obj_tracker_internal_enter();

#if USE_CUDA
    *err = cudaMalloc(&gpubuf, size);
#elif USE_OPENCL
    gpubuf = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, size, 0);
#endif

    if (!gpubuf) {
        obj_tracker_internal_leave();
        return NULL;
    }

#if USE_CUDA
    *err = cudaMemcpy(gpubuf, hostbuf, size, cudaMemcpyHostToDevice);
#elif USE_OPENCL
    memcpy(gpubuf, hostbuf, size);
#endif

    if (b2c_options.trace_copy)
        writef(STDOUT_FILENO, "gpublas: %s: %zu B : CPU ---> GPU\n", __func__, size);

    obj_tracker_internal_leave();
    return gpubuf;
}

void *b2c_copy_to_cpu(const void *gpubuf, size_t size, b2c_runtime_error_t *err)
{
    void *hostbuf = NULL;

    hostbuf = internal_malloc(size);

    if (hostbuf == NULL)
        return hostbuf;

#if USE_CUDA
    *err = cudaMemcpy(hostbuf, gpubuf, size, cudaMemcpyDeviceToHost);
#elif USE_OPENCL
    memcpy(hostbuf, gpubuf, size);
#endif

    if (b2c_options.trace_copy)
        writef(STDOUT_FILENO, "gpublas: %s: %zu B : GPU ---> CPU\n", __func__, size);

    return hostbuf;
}

void b2c_copy_from_gpu(void *hostbuf, const void *gpubuf, size_t size, b2c_runtime_error_t *err)
{
#if USE_CUDA
    err = cudaMemcpy(hostbuf, gpubuf, size, cudaMemcpyDeviceToHost);
#elif USE_OPENCL
    memcpy(hostbuf, gpubuf, size);
#endif

    if (b2c_options.trace_copy)
        writef(STDOUT_FILENO, "gpublas: %s: %zu B : GPU ---> CPU\n", __func__, size);
}

void *b2c_place_on_gpu(void *hostbuf, 
        size_t size,
        const struct objinfo **info_in,
        b2c_runtime_error_t *err,
        void *gpubuf2,
        ...)
{
    void *gpubuf;
    const struct objinfo *gpubuf2_info;

    if (!hostbuf) {
#if USE_CUDA
        *err = cudaMalloc(&gpubuf, size);
#elif USE_OPENCL
        gpubuf = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, size, 0);
#endif
    } else if ((*info_in = obj_tracker_objinfo(hostbuf))) {
        assert ((*info_in)->ptr == hostbuf);
        gpubuf = hostbuf;
        hits++;
    } else {
        gpubuf = b2c_copy_to_gpu(hostbuf, size, err);
        misses++;
    }

    if (!gpubuf || *err) {
        va_list ap;

        va_start(ap, gpubuf2);

        if (gpubuf2) {
            do {
                if (!(gpubuf2_info = va_arg(ap, const struct objinfo *)))
#if USE_CUDA
                    cudaFree(gpubuf2);
#elif USE_OPENCL
                    clSVMFree(cl_ctx, gpubuf2);
#endif
            } while ((gpubuf2 = va_arg(ap, void *)));
        }

        va_end(ap);

        writef(STDERR_FILENO, "gpublas: %s: failed to copy to GPU\n", __func__);
        b2c_fatal_error(*err, __func__);
        return NULL;
    }

    return gpubuf;
}

void b2c_cleanup_gpu_ptr(void *gpubuf, const struct objinfo *info, b2c_runtime_error_t *err)
{
    if (!info)
#if USE_CUDA
        *err = cudaFree(gpubuf);
#elif USE_OPENCL
    {
        *err = CL_SUCCESS;
        clSVMFree(cl_ctx, gpubuf);
    }
#endif
}


/* memory management */
static void *alloc_managed(size_t request)
{
#if USE_CUDA
    cudaError_t err;
#endif
    void *ptr;

    obj_tracker_internal_enter();
#if USE_CUDA
    err = cudaMallocManaged(&ptr, sizeof(size_t) + request, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        writef(STDERR_FILENO, "gpublas: %s @ %s, line %d: failed to allocate %zu B: %s - %s\n", 
                __func__, __FILE__, __LINE__, sizeof(size_t) + request, 
                cudaGetErrorName(err), cudaGetErrorString(err));
#elif USE_OPENCL
    ptr = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, sizeof(size_t) + request, 0);
    if (!ptr) {
        writef(STDERR_FILENO, "gpublas: %s @ %s, line %d: failed to allocate %zu B\n", 
                __func__, __FILE__, __LINE__, sizeof(size_t) + request);
#endif
        obj_tracker_internal_leave();
        abort();
    }

    total_managed_mem += sizeof(size_t) + request;
    *((size_t *)ptr) = request;
    obj_tracker_internal_leave();
    return ptr + sizeof(size_t);
}

static void *calloc_managed(size_t nmemb, size_t size) {
    void *ptr = alloc_managed(nmemb * size);
    memset(ptr, 0, nmemb * size);
    return ptr;
}

static void *realloc_managed(void *managed_ptr, size_t request)
{
    void *new_ptr;
    const size_t old_size = get_size_managed(managed_ptr);

    new_ptr = alloc_managed(request);

    memcpy(new_ptr, managed_ptr, old_size);
    free_managed(managed_ptr);
    return new_ptr;
}

static void free_managed(void *managed_ptr) {
    total_managed_mem -= sizeof(size_t) + get_size_managed(managed_ptr);
#if USE_CUDA
    cudaFree(managed_ptr - sizeof(size_t));
#elif USE_OPENCL
    clSVMFree(cl_ctx, managed_ptr - sizeof(size_t));
#endif
}

static size_t get_size_managed(void *managed_ptr) {
    return *(size_t *)(managed_ptr - sizeof(size_t));
}
/* memory management */

__attribute__((constructor))
void gpublas_init(void)
{
    static bool inside = false;
    pid_t tid;

    if (!b2c_initialized && !inside) {
        inside = true;
        tid = syscall(SYS_gettid);
        writef(STDOUT_FILENO, "gpublas: initializing on thread %d\n", tid);
        writef(STDOUT_FILENO, "gpublas: initializing cuBLAS...\n");
#if USE_CUDA
        init_cublas();
#elif USE_OPENCL
        init_opencl();
#endif
        writef(STDOUT_FILENO, "gpublas: initialized cuBLAS\n");

        /* initialize object tracker */
        obj_tracker_init(false);
        set_options();
        obj_tracker_set_tracking(true);
        /* add excluded regions */
        if (obj_tracker_find_excluded_regions("libcuda", "nvidia", NULL) < 0)
            writef(STDERR_FILENO, "gpublas: error while finding excluded regions: %m\n");

        writef(STDOUT_FILENO, "gpublas: initialized on thread %d\n", tid);
        b2c_initialized = true;
        inside = false;
    }
}

__attribute__((destructor))
void gpublas_fini(void)
{
    static bool inside = false;
    pid_t tid;

    if (!inside && b2c_initialized) {
        inside = true;
#if USE_CUDA
        if (cublas_initialized 
                && cublasDestroy(b2c_handle) == CUBLAS_STATUS_NOT_INITIALIZED)
            writef(STDERR_FILENO, "gpublas: failed to destroy. Not initialized\n");
#elif USE_OPENCL
        clReleaseCommandQueue(cl_queue);
        clReleaseContext(cl_ctx);
#endif

        tid = syscall(SYS_gettid);
        obj_tracker_fini();
        inside = false;
        b2c_initialized = false;

        int fd = open("statistics.csv", O_RDWR | O_CREAT, 0644);
        if (fd > -1) {
            writef(fd, "Hits, Misses\n");
            writef(fd, "%zu, %zu", hits, misses);
            close(fd);
        } else {
            writef(STDERR_FILENO, "gpublas: failed to write to statistics file: %s\n", strerror(errno));
        }

        writef(STDOUT_FILENO, "gpublas: decommissioned on thread %d\n", tid);
    }
}
