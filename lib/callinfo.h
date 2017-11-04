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

struct alloc_callinfo {
    long ip;
    size_t reqsize;
};

void
init_callinfo(enum alloc_sym sym);

bool
add_callinfo(enum alloc_sym sym, 
             long ip, 
             size_t reqsize);

/**
 * Queries by both IP and reqsize.
 * returns NULL if not found
 */
const struct alloc_callinfo *
get_callinfo(enum alloc_sym sym, 
             long ip, 
             size_t reqsize);

/**
 * returns ALLOC_UNKNWN if symbol doesn't match
 * any tracked symbols
 */
enum alloc_sym
get_alloc(const char *symbol);

