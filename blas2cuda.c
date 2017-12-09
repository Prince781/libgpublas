#include "blas2cuda.h"
#include "lib/callinfo.h"
#include "lib/obj_tracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

static bool init = false;

#if __cplusplus
extern "C" {
#endif
static void *alloc_managed(size_t request);
static void *calloc_managed(size_t nmemb, size_t size);
static void *realloc_managed(void *managed_ptr, size_t request);
static void free_managed(void *managed_ptr);
static size_t get_size_managed(void *managed_ptr);
#if __cplusplus
};
#endif

struct objmngr blas2cuda_manager = {
    .ctor = alloc_managed,
    .cctor = calloc_managed,
    .realloc = realloc_managed,
    .dtor = free_managed,
    .get_size = get_size_managed
};

cublasHandle_t b2c_handle;

static bool blas2cuda_tracking = false;

struct options b2c_options = { false, false, false };

static void set_options(void) {
    /* TODO: use secure_getenv() ? */
    char *options = getenv("BLAS2CUDA_OPTIONS");
    char *saveptr = NULL;
    char *option = NULL;
    bool help = false;

    if (!options)
        return;

    option = strtok_r(options, ";", &saveptr);
    while (option != NULL) {
        if (strcmp(option, "help") == 0) {
            if (!help) {
                fprintf(stderr, 
                        "blas2cuda options:\n"
                        "You can chain these options with a semicolon (;)\n"
                        "help           -- print help\n"
                        "debug_execfail -- debug kernel failures\n"
                        "debug_exec     -- debug kernel invocations\n"
                        "trace_copy     -- trace copies between CPU and GPU\n"
                        "track=<file>   -- use an object tracking definition\n"
                       );
                help = true;
            }
        } 
        else if (strcmp(option, "debug_execfail") == 0)
            b2c_options.debug_execfail = true;
        else if (strcmp(option, "debug_exec") == 0)
            b2c_options.debug_exec = true;
        else if (strcmp(option, "trace_copy") == 0)
            b2c_options.trace_copy = true;
        else if (strncmp(option, "track=", 6) == 0) {
            char *fname = strchr(option, '=');
            if (fname) {
                fname++;
                obj_tracker_load(fname, &blas2cuda_manager);
                blas2cuda_tracking = true;
            } else
                fprintf(stderr, "blas2cuda: you must provide a filename. Set BLAS2CUDA_OPTIONS=help.\n");
        } else {
            fprintf(stderr, "blas2cuda: unknown option '%s'. Set BLAS2CUDA_OPTIONS=help.\n", option);
        }
        option = strtok_r(NULL, ";", &saveptr);
    }
}

void init_cublas(void) {
    if (!init) {
        switch (cublasCreate(&b2c_handle)) {
            case CUBLAS_STATUS_SUCCESS:
                /* do nothing */
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                fprintf(stderr, "blas2cuda: failed to allocate resources\n");
            case CUBLAS_STATUS_NOT_INITIALIZED:
            default:
                fprintf(stderr, "blas2cuda: failed to initialize cuBLAS\n");
                exit(EXIT_FAILURE);
                break;
        }
        init = true;
    }
}

void *b2c_copy_to_gpu(const void *devbuf, size_t size)
{
    void *gpubuf = NULL;

    init_cublas();

    cudaMalloc(&gpubuf, size);

    if (!gpubuf)
        return NULL;

    cudaMemcpy(gpubuf, devbuf, size, cudaMemcpyHostToDevice);

    if (b2c_options.trace_copy)
        printf("%s: %zu B : CPU ---> GPU\n", __func__, size);

    return gpubuf;
}

void *b2c_copy_to_cpu(const void *gpubuf, size_t size)
{
    void *devbuf = NULL;

    init_cublas();

    devbuf = malloc(size);

    if (devbuf == NULL)
        return devbuf;

    cudaMemcpy(devbuf, gpubuf, size, cudaMemcpyDeviceToHost);

    if (b2c_options.trace_copy)
        printf("%s: %zu B : GPU ---> CPU\n", __func__, size);

    return devbuf;
}

void b2c_copy_from_gpu(void *cpubuf, const void *gpubuf, size_t size)
{
    cudaMemcpy(cpubuf, gpubuf, size, cudaMemcpyDeviceToHost);

    if (b2c_options.trace_copy)
        printf("%s: %zu B : GPU ---> CPU\n", __func__, size);
}

void *b2c_place_on_gpu(void *cpubuf, 
        size_t size,
        const struct objinfo **info_in,
        void *gpubuf2,
        ...)
{
    void *gpubuf;
    const struct objinfo *gpubuf2_info;

    if (!cpubuf) {
        cudaMalloc(&gpubuf, size);
    } else if ((*info_in = obj_tracker_objinfo(cpubuf))) {
        gpubuf = cpubuf;
    } else
        gpubuf = b2c_copy_to_gpu(cpubuf, size);

    if (!gpubuf || cudaPeekAtLastError() != cudaSuccess) {
        va_list ap;
        cudaError_t err = cudaGetLastError();

        va_start(ap, gpubuf2);

        if (gpubuf2) {
            do {
                if (!(gpubuf2_info = va_arg(ap, const struct objinfo *)))
                    cudaFree(gpubuf2);
            } while ((gpubuf2 = va_arg(ap, void *)));
        }

        va_end(ap);

        fprintf(stderr, "%s: failed to copy to GPU\n", __func__);
        b2c_fatal_error(err, __func__);
        return NULL;
    }

    return gpubuf;
}

void b2c_cleanup_gpu_ptr(void *gpubuf, const struct objinfo *info)
{
    if (!info)
        cudaFree(gpubuf);
}


/* memory management */
static void *alloc_managed(size_t request)
{
    void *ptr;

    cudaMallocManaged(&ptr, sizeof(size_t) + request, cudaMemAttachGlobal);
    if (cudaPeekAtLastError() != cudaSuccess) {
        cudaError_t err = cudaGetLastError();
        fprintf(stderr, "%s @ %s, line %d: failed to allocate memory: %s - %s\n", 
                __func__, __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
        abort();
    }

    *((size_t *)ptr) = request;
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
    cudaFree(managed_ptr - sizeof(size_t));
}

static size_t get_size_managed(void *managed_ptr) {
    return *(size_t *)(managed_ptr - sizeof(size_t));
}
/* memory management */

__attribute__((constructor))
void blas2cuda_init(void)
{
    pid_t tid;
    obj_tracker_init(false);
    set_options();
    tid = syscall(SYS_gettid);
    printf("initialized blas2cuda on thread %d\n", tid);
    obj_tracker_set_tracking(blas2cuda_tracking);
}

__attribute__((destructor))
void blas2cuda_fini(void)
{
    pid_t tid;
    if (init && cublasDestroy(b2c_handle) == CUBLAS_STATUS_NOT_INITIALIZED)
        fprintf(stderr, "blas2cuda: failed to destroy. Not initialized\n");

    tid = syscall(SYS_gettid);
    printf("decommissioned blas2cuda on thread %d\n", tid);
    obj_tracker_fini();
}
