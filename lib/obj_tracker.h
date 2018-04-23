#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>

extern void *(*real_malloc)(size_t);
extern void *(*real_calloc)(size_t, size_t);
extern void *(*real_realloc)(void *, size_t);
extern void (*real_free)(void *);

void *fake_malloc(size_t req);
void fake_free(void *ptr);

/**
 * These options are only available when in STANDALONE mode.
 */
#if STANDALONE
struct obj_options {
    /**
     * Whether we should only print calls. This can reduce 
     * the output of the object tracker where there are many
     * allocations.
     */
    bool only_print_calls;

    /**
     * The libraries to load, in order. 
     */
    char **blas_libs;

    /**
     * The number of BLAS libraries (max(0, size of list - 1)).
     */
    int num_blas;
};

extern struct obj_options objtracker_options;
#endif

enum alloc_sym {
    ALLOC_MALLOC,
    ALLOC_CALLOC,
    N_ALLOC_SYMS,
    ALLOC_UNKNWN
};

static inline const char *alloc_sym_tostr(enum alloc_sym sym) {
    switch (sym) {
        case ALLOC_MALLOC:
            return "malloc";
        case ALLOC_CALLOC:
            return "calloc";
        default:
            return "??";
    }
}

struct objmngr {
    void *(*ctor)(size_t);
    void *(*cctor)(size_t, size_t);
    void *(*realloc)(void *, size_t);
    void (*dtor)(void *);
    size_t (*get_size)(void *);
};

struct alloc_callinfo {
    struct objmngr mngr;
    enum alloc_sym alloc;
    size_t reqsize;
};

/**
 * Information about objects.
 */
struct objinfo {
    struct alloc_callinfo ci;
    size_t size;            /* size of the actual memory object */
    void *ptr;              /* location of object */
    struct timespec time;   /* when this object was created */
    uint64_t uid;           /* unique ID of this object */
};

enum objprint_type {
    OBJPRINT_TRACK,
    OBJPRINT_TRACK_FAIL,
    OBJPRINT_UNTRACK,
    OBJPRINT_CALL
};

void obj_tracker_print_info(enum objprint_type type, const struct objinfo *info);


/* Initialize the object tracker. */
void obj_tracker_init(bool tracking_enabled);

void obj_tracker_set_tracking(bool enabled);

/**
 * Loads a definition from a file.
 * See obj_tracker_print_obj() for file format.
 * @return 0 on success, < 0 on failure
 */
int obj_tracker_load(const char *filename, struct objmngr *mngr);

/**
 * If this pointer is managed by the object tracking system,
 * return the information about it. Otherwise, return NULL.
 */
const struct objinfo *
obj_tracker_objinfo(void *ptr);

/**
 * Decommission the object tracker.
 */
void obj_tracker_fini(void);

#if RBTREE
/**
 * Prints all objects in the tracker into a 
 */
void obj_tracker_print_rbtree(const char *filename);
#endif

#ifdef __cplusplus
};
#endif

