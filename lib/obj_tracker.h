#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

extern void *(*real_malloc)(size_t);
extern void (*real_free)(void *);

/* Initialize the object tracker. */
void __attribute__((constructor)) obj_tracker_init(void);

/**
 * Loads a definition from a file.
 * The file format must be of the following:
 * ------------------
 * <name of function 1>
 * [ PC 1 ]
 * [ PC 2 ]
 * ...
 * [ PC N ]
 * ------------------
 * <name of function 2>
 * [ PC 1 (2) ]
 * [ PC 2 (2) ]
 * ...
 * [ PC N (2) ]
 * ...
 * ------------------
 * <name of function M>
 * [ PC 1 (M) ]
 * [ PC 2 (M) ]
 * ...
 * [ PC N (M) ]
 *
 * @return 0 on success, < 0 on failure
 */
int obj_tracker_load(const char *filename);

/**
 * Decommission the object tracker.
 */
void __attribute__((destructor)) obj_tracker_fini(void);

#if RBTREE
/**
 * Prints all objects in the tracker into a 
 */
void obj_tracker_print_rbtree(const char *filename);
#endif

#ifdef __cplusplus
};
#endif

