#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void *(*real_malloc)(size_t);
extern void *(*real_calloc)(size_t, size_t);
extern void *(*real_realloc)(void *, size_t);
extern void (*real_free)(void *);

void *fake_malloc(size_t req);
void fake_free(void *ptr);

/* include callinfo.h */
struct alloc_callinfo;

/**
 * Information about objects.
 */
struct objinfo {
    struct alloc_callinfo ci;
    size_t size;        /* size of the actual memory object */
    void *ptr;          /* location of object */
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

/* users of this library must include callinfo.h */
struct objmngr;

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

