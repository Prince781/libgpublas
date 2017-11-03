/*****
 * obj_tracker.cu
 * Tracks objects.
 */
// #include "obj_tracker.h"
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <malloc.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include "rbtree.h"

/*
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void *reallocarray(void *ptr, size_t nmemb, size_t size);
*/

/**
 * Information about objects.
 */
struct objinfo {
    unsigned long ip;   /* instruction pointer at request. */
    size_t reqsize;     /* size of the memory object requested */
    size_t size;        /* size of the actual memory object */
    void *ptr;          /* location of object (= freeable block + sizeof(struct objinfo)) */
};

enum alloc_sym {
    ALLOC_MALLOC,
/*
    ALLOC_CALLOC,
    ALLOC_REALLOC,
    ALLOC_REALLOCARRAY,
*/
    N_ALLOC_SYMS
};

/**
 * Information for knowing when to record objects
 */
struct syminfo {
    const char *symbol;     /* symbol of the allocator */
    unsigned long *IPs;     /* instruction pointers (sorted) */
    size_t n_IPs;           /* number of instruction pointers */
};

/**
 * Sorted by symbol name.
 */
static struct syminfo symbols[N_ALLOC_SYMS] = {
    {
        .symbol = "malloc",
        .IPs = NULL,
        .n_IPs = 0
    },
/*
    {
        .symbol = "calloc",
        .IPs = NULL,
        .n_IPs = 0
    },
    {
        .symbol = "realloc",
        .IPs = NULL,
        .n_IPs = 0
    },
    {
        .symbol = "reallocarray",
        .IPs = NULL,
        .n_IPs = 0
    }
*/
};


static bool initialized = false;
static struct rbtree *objects = NULL;
static unsigned long num_objects = 0;

/**
 * Handle to libc.
 */
static void *handle = NULL;

void *(*real_malloc)(size_t) = NULL;
void *(*real_free)(void *) = NULL;

void *__real_malloc(size_t bytes);
void __real_free(void *ptr);

static void obj_tracker_get_fptrs(void) {
    extern void exit(int);
    static bool success = false;

    if (!success) {
        if (real_malloc == NULL) {
            dlerror();
            real_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
            if (real_malloc == NULL) {
                fprintf(stderr, "dlsym: %s\n", dlerror());
                exit(1);
            } else 
                printf("real_malloc = %p\n", real_malloc);
        }

        if (real_free == NULL) {
            dlerror();
            real_free = (void *(*)(void *)) dlsym(RTLD_NEXT, "free");
            if (real_free == NULL) {
                fprintf(stderr, "dlsym: %s\n", dlerror());
                exit(1);
            } else
                printf("real_free = %p\n", real_free);
        }

        success = true;
    }
}

void __attribute__((constructor)) obj_tracker_init(void)
{
    extern void exit(int status);

    if (!initialized) {
        initialized = true;
        /*
        if (!(handle = dlopen("libc.so.6", RTLD_LAZY | RTLD_LOCAL))) {
            fprintf(stderr, "%s: Failed to get handle to libc!\n", __func__);
            fprintf(stderr, "dlopen: %s\n", dlerror());
            exit(1);
        }
        */

        obj_tracker_get_fptrs();

        printf("initialized object tracker\n");
    }
}

int obj_tracker_load(const char *filename)
{
    extern void perror(const char *);
    extern void exit(int);

    FILE *fp;
    char *buf = NULL;
    enum alloc_sym sym;
    unsigned long ip;
    int res;

    if ((fp = fopen(filename, "r")) == NULL) {
        perror("fopen");
        exit(1);
    }

    do {
        if (fscanf(fp, "%ms", &buf) < 1) {
            perror("fscanf");
            break;
        }

        if (strcmp(symbols[ALLOC_MALLOC].symbol, buf) == 0)
            sym = ALLOC_MALLOC;
        /*
        else if (strcmp(symbols[ALLOC_CALLOC].symbol, buf) == 0)
            sym = ALLOC_CALLOC;
        else if (strcmp(symbols[ALLOC_REALLOC].symbol, buf) == 0)
            sym = ALLOC_REALLOC;
        else if (strcmp(symbols[ALLOC_REALLOCARRAY].symbol, buf) == 0)
            sym = ALLOC_REALLOCARRAY;
        */
        else {
            fprintf(stderr, "%s: unsupported symbol '%s'\n", __func__, buf);
            continue;
        }

        while ((res = fscanf(fp, "%lx", &ip)) == 1) {
            symbols[sym].n_IPs++;
            symbols[sym].IPs = (unsigned long *) reallocarray(symbols[sym].IPs, 
                    symbols[sym].n_IPs, sizeof(symbols[sym].IPs[0]));

            symbols[sym].IPs[symbols[sym].n_IPs - 1] = ip;
        }
        
        free(buf);
        buf = NULL;
    } while (res != EOF);

    free(buf);
    return fclose(fp);
}

void __attribute__((destructor)) obj_tracker_fini(void) {
    if (initialized) {
        initialized = false;
        if (handle)
            if (dlclose(handle) != 0)
                fprintf(stderr, "dlclose: %s\n", dlerror());
        printf("decommissioned object tracker\n");
    }
}

static int objects_compare(const void *o1, const void *o2) {
    return ((const struct objinfo *)o1)->ip - ((const struct objinfo *)o2)->ip;
}

static void track_object(void *ptr, size_t request, size_t size, unsigned long ip)
{
    struct objinfo *oinfo;

    oinfo = (struct objinfo *) real_malloc(sizeof(*oinfo));
    oinfo->ip = ip;
    oinfo->reqsize = request;
    oinfo->size = size;
    oinfo->ptr = ptr;
    rbtree_insert(&objects, oinfo, objects_compare);
    ++num_objects;
    printf("Tracking object @ %p (size=%zu B)\n", ptr, size);
}

static void untrack_object(void *ptr)
{
    struct objinfo o = { .ptr = ptr };
    struct rbtree *node;

    node = rbtree_find(&objects, &o, objects_compare);
    if (!rbtree_delete(&objects, node))
        fprintf(stderr, "could not delete objinfo for %p (not found)\n", ptr);
    else {
        --num_objects;
        fprintf(stderr, "Untracking object @ %p\n", ptr);
    }
}

void *__wrap_malloc(size_t request) {
    extern void exit(int);
    void *ptr;

    /*
    if (!initialized)
        obj_tracker_init();
    
    if (real_malloc == NULL) {
        fprintf(stderr, "error: real_malloc is NULL! Was constructor called?\n");
        exit(1);
    }
    */

    /* We have a valid function pointer to malloc().
     * Now we first do some bookkeeping.
     */
    ptr = __real_malloc(sizeof(struct objinfo) + request);

    /* quit early if there was an error */
    if (ptr)
        track_object(ptr, request, 
                malloc_usable_size(ptr) - sizeof(struct objinfo), 
                0 /* TODO (use libunwind) */);

    return ptr;
}

void __wrap_free(void *ptr) {
    extern void exit(int);

    /*
    if (!initialized)
        obj_tracker_init();
    if (real_free == NULL) {
        fprintf(stderr, "error: real_free is NULL! Was constructor called?\n");
        exit(1);
    }
    */

    /* We have a valid function pointer to free(). */
    if (ptr)
        untrack_object(ptr);

    __real_free(ptr);
}
