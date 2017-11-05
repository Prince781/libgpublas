#pragma once

#include <stddef.h>
#include <stdbool.h>

#if __cplusplus
extern "C" {
#endif

extern void *(*real_malloc)(size_t);
extern void (*real_free)(void *);

void *fake_malloc(size_t req);
void fake_free(void *ptr);

/* Initialize the object tracker. */
void obj_tracker_init(bool tracking_enabled);

void obj_tracker_set_tracking(bool enabled);

/* users of this library must include callinfo.h */
struct objmngr;

/**
 * Loads a definition from a file.
 * The file format must be of the following:
 * <name of function 1>
 * [ PC 1 ]
 * [ PC 2 ]
 * ...
 * [ PC N ]
 * <name of function 2>
 * [ PC 1 (2) ]
 * [ PC 2 (2) ]
 * ...
 * [ PC N (2) ]
 * ...
 * <name of function M>
 * [ PC 1 (M) ]
 * [ PC 2 (M) ]
 * ...
 * [ PC N (M) ]
 *
 * @return 0 on success, < 0 on failure
 */
int obj_tracker_load(const char *filename, struct objmngr *mngr);

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

#if __cplusplus
};
#endif

