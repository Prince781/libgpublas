#include "callinfo.h"
#include "obj_tracker.h"
#define _GNU_SOURCE
#include <search.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h> /* intptr_t */

#define CAPACITY    (1 << 10)

/**
 * Information for knowing when to record objects
 */
struct syminfo {
    /* symbol of the allocator */
    const char *symbol;
    /* instruction pointers */
    struct hsearch_data *offs_str;
    /* requested size */
    struct hsearch_data *reqsize;
    /* memory locations */
    struct hsearch_data *ptr;
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
            perror("Failed to create hash table of allocations");\
            abort(); \
        }\
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
              const struct ip_offs *offs, 
              size_t reqsize,
              void *ptr)
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
    ci->offs = *offs;
    ci->reqsize = reqsize;
    ci->ptr = ptr;
    return ci;
}

char *
ip_offs_tostr(const struct ip_offs *offs, char buf[1 + N_IP_OFFS * 8])
{
    char tmpbuf[8];

    buf[0] = '\0';
    for (int i=0; i<N_IP_OFFS; ++i) {
        snprintf(tmpbuf, sizeof(tmpbuf), "%hx", offs->off[i]);
        if (i > 0)
            strcat(buf, ".");
        strcat(buf, tmpbuf);
    }

    return buf;
}

int
ip_offs_parse(const char *str, struct ip_offs *offs)
{
    int n;
    short o;
    char *dupstr = strdup(str);
    char *saveptr = NULL;
    char *off_s;

    off_s = strtok_r(dupstr, ".", &saveptr);
    for (n = 0; n < N_IP_OFFS; ++n) {
        if (!off_s || sscanf(off_s, "%hx", &o) != 1)
            break;
        else {
            offs->off[n] = o;
            off_s = strtok_r(NULL, ".", &saveptr);
        }
    }

    free(dupstr);
    return n;
}

void
init_callinfo(void)
{
    for (int i=0; i<N_ALLOC_SYMS; ++i) {
        struct syminfo *si = get_syminfo(i);
        mktabl(si, offs_str, CAPACITY);
        mktabl(si, reqsize, CAPACITY);
        /* mktabl(si, ptr, CAPACITY); */
    }
}

int
add_callinfo(enum alloc_sym sym, 
             const struct objmngr *mngr,
             const struct ip_offs *offs,
             size_t reqsize,
             void *ptr)
{
    struct syminfo *info;
    struct alloc_callinfo *ci;
    ENTRY *entp;
    char *valstr = NULL;
    int retval = -1;
    char offs_str[1 + N_IP_OFFS * 8];

    ip_offs_tostr(offs, offs_str);
    if ((info = get_syminfo(sym))) {
        ci = make_callinfo(mngr, sym, offs, reqsize, ptr);
        retval = 1;
        if (!lookup(info, offs_str, entp)) {
            if (!insert(info, offs_str, ci, valstr)) {
                real_free(valstr);
                return -1;
            } else
                retval = 0;
        }
        if (!lookup(info, reqsize, entp)) {
            if (!insert(info, reqsize, ci, valstr)) {
                real_free(valstr);
                return -1;
            } else
                retval = 0;
        }
        /*
        if (!lookup(info, ptr, entp)) {
            if (!insert(info, ptr, ci, valstr)) {
                real_free(valstr);
                return -1;
            } else
                retval = 0;
        }
        */
    }

    return retval;
}

struct alloc_callinfo *
get_callinfo_or(enum alloc_sym sym, 
        const struct ip_offs *offs, 
        size_t reqsize, 
        void *ptr)
{
    struct syminfo *info;
    ENTRY *entp;
    char offs_str[1 + N_IP_OFFS * 8];

    if (!(info = get_syminfo(sym)))
        return NULL;

    ip_offs_tostr(offs, offs_str);
    if (lookup(info, offs_str, entp))
        return entp->data;
    if (lookup(info, reqsize, entp))
        return entp->data;
    /*
    if (lookup(info, ptr, entp))
        return entp->data;
        */

    return NULL;
}

struct alloc_callinfo *
get_callinfo_and(enum alloc_sym sym,
        const struct ip_offs *offs,
        size_t reqsize, 
        void *ptr)
{
    struct syminfo *info;
    ENTRY *entp = NULL;
    char offs_str[1 + N_IP_OFFS * 8];

    if (!(info = get_syminfo(sym)))
        return NULL;

    ip_offs_tostr(offs, offs_str);
    if (!lookup(info, offs_str, entp))
        return NULL;

    if (!lookup(info, reqsize, entp))
        return NULL;

    /* TODO: is tracking ptr reliable?
    if (!lookup(info, ptr, entp))
        return NULL;
        */

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

