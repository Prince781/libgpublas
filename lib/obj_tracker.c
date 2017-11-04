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
#include "obj_tracker.h"

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

/* allows us to temporarily disable tracking */
static bool tracking = true;

/**
 * Handle to libc.
 */
static void *handle = NULL;

void *(*real_malloc)(size_t) = NULL;
void *(*real_free)(void *) = NULL;

static void obj_tracker_get_fptrs(void) {
    static bool success = false;

    if (!success) {
        if (real_malloc == NULL) {
            dlerror();
            real_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
            if (real_malloc == NULL) {
                fprintf(stderr, "dlsym: %s\n", dlerror());
                abort();
            }
        }

        if (real_free == NULL) {
            dlerror();
            real_free = (void *(*)(void *)) dlsym(RTLD_NEXT, "free");
            if (real_free == NULL) {
                fprintf(stderr, "dlsym: %s\n", dlerror());
                abort();
            }
        }

        printf("real_malloc = %p, fake malloc() = %p\n", real_malloc, &malloc);
        printf("real_free = %p, fake free() = %p\n", real_free, &free);

        success = true;
    }
}

void __attribute__((constructor)) obj_tracker_init(void)
{
    if (!initialized) {
        tracking = false;
        initialized = true;

        obj_tracker_get_fptrs();

        printf("initialized object tracker\n");
        tracking = true;
    }
}

int obj_tracker_load(const char *filename)
{
    extern void perror(const char *);

    FILE *fp;
    char *buf = NULL;
    enum alloc_sym sym;
    unsigned long ip;
    int res;

    if ((fp = fopen(filename, "r")) == NULL) {
        perror("fopen");
        abort();
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
    return ((const struct objinfo *)o1)->ptr - ((const struct objinfo *)o2)->ptr;
}

static void object_print(const void *o, int n, char buf[n]) {
    const struct objinfo *oinfo = o;

    if (oinfo)
        snprintf(buf, n, "label=\"ptr=%p,size=%zu\"", oinfo->ptr, oinfo->size);
    else
        snprintf(buf, n, "label=\"(nil)\"");
}

static void track_object(void *ptr, size_t request, size_t size, unsigned long ip)
{
    struct objinfo *oinfo;
    struct rbtree *node;
    char buf[64];

    tracking = false;
    oinfo = real_malloc(sizeof(*oinfo));
    oinfo->ip = ip;
    oinfo->reqsize = request;
    oinfo->size = size;
    oinfo->ptr = ptr;
    node = rbtree_insert(&objects, oinfo, objects_compare);
    if (node) {
        ++num_objects;
        printf("Tracking object @ %p (size=%zu B) {record @ %p}\n", ptr, size, node);
    } else {
        fprintf(stderr, "Failed to track object @ %p (size=%zu B)\n", ptr, size);
    }
    snprintf(buf, sizeof(buf), "tree%ld.dot", num_objects);
    obj_tracker_print_rbtree(buf);
    tracking = true;
}

static void untrack_object(void *ptr)
{
    struct objinfo o = { .ptr = ptr, .ip = 0 };
    struct rbtree *node;
    void *objinfo;
    char buf[64];

    tracking = false;
    node = rbtree_find(&objects, &o, objects_compare);
    if (!node) {
        fprintf(stderr, "could not delete objinfo for %p (not found)\n", ptr);
    } else {
        objinfo = (void *) node->item;
        rbtree_delete(&objects, node);
        real_free(objinfo);
        --num_objects;
        fprintf(stderr, "Untracking object @ %p\n", ptr);
    }
    snprintf(buf, sizeof(buf), "tree%ld.dot", num_objects);
    obj_tracker_print_rbtree(buf);
    tracking = true;
}

void obj_tracker_print_rbtree(const char *filename) {
    FILE *stream;

    stream = filename ? fopen(filename, "a") : tmpfile();

    if (!stream) {
        perror("fopen");
        return;
    }

    rbtree_print(objects, object_print, stream);

    fclose(stream);
}

void *malloc(size_t request) {
    void *ptr;
    static bool inside = false;

    if (inside)
        return real_malloc(request);

    inside = true;

    if (!initialized)
        obj_tracker_init();
    
    if (real_malloc == NULL) {
        fprintf(stderr, "error: real_malloc is NULL! Was constructor called?\n");
        abort();
    }

    /* We have a valid function pointer to malloc().
     * Now we first do some bookkeeping.
     */
    ptr = real_malloc(sizeof(struct objinfo) + request);

    /* quit early if there was an error */
    if (ptr && tracking)
        track_object(ptr, request, 
                malloc_usable_size(ptr) - sizeof(struct objinfo), 
                0 /* TODO (use libunwind) */);

    inside = false;
    return ptr;
}

void free(void *ptr) {
    static bool inside = false;

    if (inside) {
        real_free(ptr);
        return;
    }

    inside = true;

    if (!initialized)
        obj_tracker_init();
    if (real_free == NULL) {
        fprintf(stderr, "error: real_free is NULL! Was constructor called?\n");
        abort();
    }

    /* We have a valid function pointer to free(). */
    if (ptr && tracking)
        untrack_object(ptr);

    real_free(ptr);

    inside = false;
}
