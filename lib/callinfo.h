#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

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

void
init_callinfo(void);

/**
 * Adds a new description for a call.
 * {sym}        - What symbol to intercept.
 * {mngr}       - The manager to use for objects allocated
 *                at this call. If NULL, glibc's malloc()
 *                and free() will be used.
 * {reqsize}    - Requested size.
 * Returns: negative on error, 0 on success, and positive if 
 *          callinfo wasn't added because it already exists.
 */
int
add_callinfo(enum alloc_sym sym, 
             const struct objmngr *mngr,
             size_t reqsize);

/**
 * Queries by reqsize.
 * Returns NULL if not found.
 */
struct alloc_callinfo *
get_callinfo_reqsize(enum alloc_sym sym, size_t reqsize);


/**
 * Queries by either IP or reqsize.
 * returns NULL if not found
 */
struct alloc_callinfo *
get_callinfo_or(enum alloc_sym sym, 
                size_t reqsize);

/**
 * Queries by reqsize and unique ID.
 * returns NULL if not found
 */
struct alloc_callinfo *
get_callinfo_and(enum alloc_sym sym, 
                 size_t reqsize);


/**
 * returns ALLOC_UNKNWN if symbol doesn't match
 * any tracked symbols
 */
enum alloc_sym
get_alloc(const char *symbol);

#ifdef __cplusplus
};
#endif
