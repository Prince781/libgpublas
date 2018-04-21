#define _GNU_SOURCE
#include "callinfo.h"
#include "obj_tracker.h"
#include <search.h>
#include <stdio.h>  /* asprintf */
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h> /* intptr_t */
#include "../common.h"

#define CAPACITY    (1 << 10)

/**
 * Information for knowing when to record objects
 */
struct syminfo {
    /* symbol of the allocator */
    const char *symbol;
    /* requested size */
    struct hsearch_data *reqsize;
    /* offs_str . reqsize */
    struct hsearch_data *ci_string;
};

#define symdef(name) { .symbol = #name }

static struct syminfo symbols[N_ALLOC_SYMS] = {
    symdef(malloc),
    symdef(calloc),
};

#define def_tostr_(type,typename,fmt) \
static char *tostr_##typename(type a) {\
    static char buf[64];\
    snprintf(buf, sizeof(buf), fmt, a);\
    return buf;\
}

#define def_tostr(type,fmt) def_tostr_(type,type,fmt)

def_tostr(long, "%lx")
def_tostr(size_t, "%zu")
def_tostr_(void *, ptr, "%p")
def_tostr_(const char *, str, "%s")

#define tostr(val) _Generic((val), \
                    long:   tostr_long,\
                    size_t: tostr_size_t, \
                    void *: tostr_ptr, \
                    const char *: tostr_str, \
                    char *: tostr_str \
)(val)

static ENTRY *insert_retval;

/**
 * keyv will be evaluated three times
 */
#define insert(sym,keyv,val,valstr)\
    (!sym->keyv ? (intptr_t)(valstr=0) : hsearch_r((ENTRY){ .key = (valstr = strdup(tostr(keyv))), .data = (void *)(val) }, ENTER, &insert_retval, sym->keyv ))

/**
 * val will be evaluated three times
 */
#define lookup(sym,val,result)\
    (!sym->val ? 0 : hsearch_r((ENTRY){ .key = tostr(val), .data = (void *)(val) }, FIND, &result, sym->val))

#define mktabl(sym,member,nel) \
    if (!sym->member) {\
        sym->member = real_calloc(1, sizeof(*sym->member));\
        if (hcreate_r(nel, sym->member) < 0) {\
            writef(STDERR_FILENO, "objtracker: Failed to create hash table of allocations: %s", strerror(errno));\
            abort(); \
        }\
    }

static inline int 
ci_tostr(char **bufp, size_t reqsize) {
    return asprintf(bufp, "req=%zu", reqsize);
}

static inline struct syminfo *get_syminfo(enum alloc_sym sym) {
    switch (sym) {
        case ALLOC_MALLOC:
        case ALLOC_CALLOC:
            return &symbols[sym];
        default:
            return NULL;
    }
}

static inline struct alloc_callinfo *
make_callinfo(const struct objmngr *mngr, 
              enum alloc_sym sym,
              size_t reqsize)
{
    struct alloc_callinfo *ci = malloc(sizeof(*ci));

    if (mngr)
        memcpy(&ci->mngr, mngr, sizeof(*mngr));
    else {
        ci->mngr.ctor = real_malloc;
        ci->mngr.cctor = real_calloc;
        ci->mngr.dtor = real_free;
        ci->mngr.get_size = malloc_usable_size;
    }
    ci->alloc = sym;
    ci->reqsize = reqsize;
    return ci;
}

void
init_callinfo(void)
{
    for (int i=0; i<N_ALLOC_SYMS; ++i) {
        struct syminfo *si = get_syminfo(i);
        mktabl(si, reqsize, CAPACITY);
        mktabl(si, ci_string, CAPACITY);
    }
}

int
add_callinfo(enum alloc_sym sym, 
             const struct objmngr *mngr,
             size_t reqsize)
{
    struct syminfo *info;
    struct alloc_callinfo *ci;
    ENTRY *entp;
    char *valstr = NULL;
    int retval = -1;
    char *ci_string = NULL;

    if ((info = get_syminfo(sym))) {
        ci = make_callinfo(mngr, sym, reqsize);
        if (ci_tostr(&ci_string, reqsize) < 0) {
            real_free(ci);
            return -1;
        }
        retval = 1;
        if (!lookup(info, ci_string, entp)) {
            if (!insert(info, ci_string, ci, valstr)) {
                real_free(ci);
                real_free(ci_string);
                real_free(valstr);
                return -1;
            }
        } else {
            writef(STDERR_FILENO, 
                    "objtracker: Warning: callinfo (%s) was already present in table!\n",
                    ci_string);
            real_free(ci);
            real_free(ci_string);
            return -1;
        }
        if (!lookup(info, reqsize, entp)) {
            if (!insert(info, reqsize, ci, valstr)) {
                real_free(valstr);
                return -1;
            } else
                retval = 0;
        }
    }

    return retval;
}

struct alloc_callinfo *
get_callinfo_reqsize(enum alloc_sym sym, size_t reqsize)
{
    struct syminfo *info;
    ENTRY *entp;

    if (!(info = get_syminfo(sym)))
        return NULL;

    if (lookup(info, reqsize, entp))
        return entp->data;

    return NULL;
}

struct alloc_callinfo *
get_callinfo_or(enum alloc_sym sym, 
        size_t reqsize)
{
    struct syminfo *info;
    ENTRY *entp;

    if (!(info = get_syminfo(sym)))
        return NULL;

    if (lookup(info, reqsize, entp))
        return entp->data;

    return NULL;
}

struct alloc_callinfo *
get_callinfo_and(enum alloc_sym sym,
        size_t reqsize)
{
    struct syminfo *info;
    ENTRY *entp = NULL;
    char *ci_string = NULL;

    if (!(info = get_syminfo(sym)))
        return NULL;

    ci_tostr(&ci_string, reqsize);

    if (!lookup(info, ci_string, entp)) {
        real_free(ci_string);
        return NULL;
    }

    real_free(ci_string);
    return entp ? entp->data : NULL;
}


enum alloc_sym
get_alloc(const char *symbol)
{
    for (int i=0; i<N_ALLOC_SYMS; ++i)
        if (strcmp(symbol, symbols[i].symbol) == 0)
            return i;

    return ALLOC_UNKNWN;
}

