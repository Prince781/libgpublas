#pragma once

#include <stddef.h>
#include <stdbool.h>

enum alloc_sym {
    ALLOC_MALLOC,
/*
    ALLOC_CALLOC,
    ALLOC_REALLOC,
    ALLOC_REALLOCARRAY,
*/
    N_ALLOC_SYMS,
    ALLOC_UNKNWN
};

struct objmngr {
    void *(*ctor)(size_t);
    void (*dtor)(void *);
    size_t (*get_size)(void *);
};

struct alloc_callinfo {
    struct objmngr mngr;
    long ip;
    size_t reqsize;
};

void
init_callinfo(enum alloc_sym sym);

/**
 * Adds a new description for a call.
 * {sym}        - What symbol to intercept.
 * {mngr}       - The manager to use for objects allocated
 *                at this call. If NULL, glibc's malloc()
 *                and free() will be used.
 * {ip}         - Instruction pointer at time of call.
 * {reqsize}    - Requested size.
 */
bool
add_callinfo(enum alloc_sym sym, 
             const struct objmngr *mngr,
             long ip, 
             size_t reqsize);

/**
 * Queries by either IP or reqsize.
 * returns NULL if not found
 */
struct alloc_callinfo *
get_callinfo_or(enum alloc_sym sym, 
                long ip, 
                size_t reqsize);

/**
 * Queries by both IP and reqsize.
 * returns NULL if not found
 */
struct alloc_callinfo *
get_callinfo_and(enum alloc_sym sym, 
                 long ip,
                 size_t reqsize);


/**
 * returns ALLOC_UNKNWN if symbol doesn't match
 * any tracked symbols
 */
enum alloc_sym
get_alloc(const char *symbol);

