#include "callinfo.h"
#define _GNU_SOURCE
#include <search.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Information for knowing when to record objects
 */
struct syminfo {
    /* symbol of the allocator */
    const char *symbol;
    /* instruction pointers */
    struct hsearch_data *ip;
    /* requested size */
    struct hsearch_data *reqsize;
};

#define symdef(name) { .symbol = #name }

static struct syminfo symbols[N_ALLOC_SYMS] = {
    symdef(malloc)
};

#define def_tostr(type,fmt) \
static char *tostr_##type(type a) {\
    static char buf[64];\
    snprintf(buf, sizeof(buf), fmt, a);\
    return buf;\
}

def_tostr(long, "%lx")
def_tostr(size_t, "%zu")

#define tostr(val) _Generic((val), \
                    long:   tostr_long,\
                    size_t: tostr_size_t \
)(val)

/**
 * keyv will be evaluated twice
 */
#define insert(sym,keyv,val)\
    !sym->keyv ? 0 : hsearch_r((ENTRY){ .key = tostr(keyv), .data = (void *)(val) }, ENTER, NULL, sym->keyv )

/**
 * val will be evaluated three times
 */
#define lookup(sym,val,result)\
    !sym->val ? 0 : hsearch_r((ENTRY){ .key = tostr(val), .data = (void *)(val) }, FIND, &result, sym->val)

#define mktabl(sym,member,nel)\
    (!sym->member ? hcreate_r(nel,sym->member) : 0)

static inline struct syminfo *get_syminfo(enum alloc_sym sym) {
    return sym == ALLOC_MALLOC ? &symbols[sym] : NULL;
}

static inline struct alloc_callinfo *
make_callinfo(long ip, size_t reqsize)
{
    struct alloc_callinfo *ci = malloc(sizeof(*ci));

    ci->ip = ip;
    ci->reqsize = reqsize;
    return ci;
}

void
init_callinfo(enum alloc_sym sym)
{
    struct syminfo *si = get_syminfo(sym);

    for (int i=0; i<N_ALLOC_SYMS; ++i) {
        mktabl(si, ip, 1024);
        mktabl(si, reqsize, 1024);
    }
}

void 
add_callinfo(enum alloc_sym sym, long ip, size_t reqsize)
{
    struct syminfo *info;
    struct alloc_callinfo *ci;

    if ((info = get_syminfo(sym))) {
        ci = make_callinfo(ip, reqsize);
        insert(info, ip, ci);
    }
}

const struct alloc_callinfo *
get_callinfo(enum alloc_sym sym, long ip, size_t reqsize)
{
    struct syminfo *info;
    ENTRY *entp;

    if (!(info = get_syminfo(sym)))
        return NULL;

    if (lookup(info, ip, entp))
        return entp->data;
    if (lookup(info, reqsize, entp))
        return entp->data;

    return NULL;
}

enum alloc_sym
get_alloc(const char *symbol)
{
    if (strcmp(symbol, symbols[ALLOC_MALLOC].symbol) == 0)
        return ALLOC_MALLOC;
    return ALLOC_UNKNWN;
}

