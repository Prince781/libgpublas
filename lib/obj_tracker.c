/*****
 * obj_tracker.cu
 * Tracks objects.
 */
// #include "obj_tracker.h"
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <malloc.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#if RBTREE
#include "rbtree.h"
#else
#include <search.h>
#endif
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <libunwind.h>
#include "obj_tracker.h"

#if DEBUG_TRACKING
#define TRACE_OUTPUT
#endif

/*
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void *reallocarray(void *ptr, size_t nmemb, size_t size);
*/

/**
 * Information about objects.
 */
struct objinfo {
    unsigned long ip;   /* instruction pointer at request. */
    size_t reqsize;     /* size of the memory object requested */
    size_t size;        /* size of the actual memory object */
    void *ptr;          /* location of object (= freeable block + sizeof(struct objinfo)) */
};

enum alloc_sym {
    ALLOC_MALLOC,
/*
    ALLOC_CALLOC,
    ALLOC_REALLOC,
    ALLOC_REALLOCARRAY,
*/
    N_ALLOC_SYMS
};

/**
 * Information for knowing when to record objects
 */
struct syminfo {
    const char *symbol;     /* symbol of the allocator */
    unsigned long *IPs;     /* instruction pointers (sorted) */
    size_t n_IPs;           /* number of instruction pointers */
};

#if DEBUG_TRACKING
static void **pointers;
static unsigned long num_pointers;
static unsigned long pnt_len;

/* use from gdb */
static bool __attribute__((unused)) is_tracked(void *ptr)
{
    for (unsigned long i=0; i < num_pointers; ++i)
        if (pointers[i] == ptr)
            return true;
    return false;
}

static void track(void *ptr)
{
    unsigned long newN = num_pointers + 1;

    if (newN > pnt_len) {
        pnt_len = (pnt_len ? pnt_len : 1) << 1;
        pointers = reallocarray(pointers, pnt_len, sizeof(*pointers));
    }

    pointers[num_pointers] = ptr;
    num_pointers = newN;
}

static void untrack(void *ptr)
{
    unsigned long newN = num_pointers - 1;
    unsigned long i;

    for (i=0; i<num_pointers && pointers[i] != ptr; ++i)
        ;

    if (i < num_pointers && pointers[i] == ptr) {
        pointers[i] = pointers[newN];
        num_pointers = newN;
    }
}
#else
static inline void track(void *ptr) { }
static inline void untrack(void *ptr) { }
#endif

/**
 * Sorted by symbol name.
 */
static struct syminfo symbols[N_ALLOC_SYMS] = {
    {
        .symbol = "malloc",
        .IPs = NULL,
        .n_IPs = 0
    },
/*
    {
        .symbol = "calloc",
        .IPs = NULL,
        .n_IPs = 0
    },
    {
        .symbol = "realloc",
        .IPs = NULL,
        .n_IPs = 0
    },
    {
        .symbol = "reallocarray",
        .IPs = NULL,
        .n_IPs = 0
    }
*/
};


static bool initialized = false;
static bool destroying = false;
static pid_t creation_thread = 0;
#if RBTREE
static struct rbtree __thread *objects = NULL;
#else
static void __thread *objects = NULL;
#endif
static unsigned long __thread num_objects = 0;

/* allows us to temporarily disable tracking */
static bool tracking = true;

void *(*real_malloc)(size_t) = NULL;
void (*real_free)(void *) = NULL;

static void obj_tracker_get_fptrs(void) {
    static bool success = false;

    if (!success) {
        if (real_malloc == NULL) {
            dlerror();
            real_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
            if (real_malloc == NULL) {
                fprintf(stderr, "dlsym: %s\n", dlerror());
                abort();
            }
        }

        if (real_free == NULL) {
            dlerror();
            real_free = (void (*)(void *)) dlsym(RTLD_NEXT, "free");
            if (real_free == NULL) {
                fprintf(stderr, "dlsym: %s\n", dlerror());
                abort();
            }
        }

        printf("real_malloc = %p, fake malloc() = %p\n", real_malloc, &malloc);
        printf("real_free = %p, fake free() = %p\n", real_free, &free);

        success = true;
    }
}

void __attribute__((constructor)) obj_tracker_init(void)
{
    extern char etext, edata, end;

    if (!initialized) {
        tracking = false;
        initialized = true;

        creation_thread = syscall(SYS_gettid);

        obj_tracker_get_fptrs();

        printf("initialized object tracker on thread %d\n", creation_thread);
        printf("segments: {\n"
               "    .text = %10p\n"
               "    .data = %10p\n"
               "    .bss  = %10p\n"
               "    .heap = %10p\n"
               "}\n", &etext, &edata, &end, sbrk(0));
        tracking = true;
    }
}

int obj_tracker_load(const char *filename)
{
    extern void perror(const char *);

    FILE *fp;
    char *buf = NULL;
    enum alloc_sym sym;
    unsigned long ip;
    int res;

    if ((fp = fopen(filename, "r")) == NULL) {
        perror("fopen");
        abort();
    }

    do {
        if (fscanf(fp, "%ms", &buf) < 1) {
            perror("fscanf");
            break;
        }

        if (strcmp(symbols[ALLOC_MALLOC].symbol, buf) == 0)
            sym = ALLOC_MALLOC;
        /*
        else if (strcmp(symbols[ALLOC_CALLOC].symbol, buf) == 0)
            sym = ALLOC_CALLOC;
        else if (strcmp(symbols[ALLOC_REALLOC].symbol, buf) == 0)
            sym = ALLOC_REALLOC;
        else if (strcmp(symbols[ALLOC_REALLOCARRAY].symbol, buf) == 0)
            sym = ALLOC_REALLOCARRAY;
        */
        else {
            fprintf(stderr, "%s: unsupported symbol '%s'\n", __func__, buf);
            continue;
        }

        while ((res = fscanf(fp, "%lx", &ip)) == 1) {
            symbols[sym].n_IPs++;
            symbols[sym].IPs = (unsigned long *) reallocarray(symbols[sym].IPs, 
                    symbols[sym].n_IPs, sizeof(symbols[sym].IPs[0]));

            symbols[sym].IPs[symbols[sym].n_IPs - 1] = ip;
        }
        
        free(buf);
        buf = NULL;
    } while (res != EOF);

    free(buf);
    return fclose(fp);
}

#if RBTREE
static void object_destroy(void *o, void *udata) {
    real_free(o);
}
#endif

void __attribute__((destructor)) obj_tracker_fini(void) {
    pid_t tid;
    if (initialized && !destroying) {
        destroying = true;
#if RBTREE
        rbtree_destroy(&objects, object_destroy, NULL);
#else
        tdestroy(objects, real_free);
#endif
        objects = NULL;
        tid = syscall(SYS_gettid);
        printf("decommissioned object tracker on thread %d\n", tid);
        initialized = false;
        destroying = false;
    }
}

static int objects_compare(const void *o1, const void *o2) {
    return ((const struct objinfo *)o1)->ptr - ((const struct objinfo *)o2)->ptr;
}

#if RBTREE
static void object_print(const void *o, int n, char buf[n]) {
    const struct objinfo *oinfo = o;

    if (oinfo)
        snprintf(buf, n, "label=\"ptr=%p,size=%zu\"", oinfo->ptr, oinfo->size);
    else
        snprintf(buf, n, "label=\"(nil)\"");
}
#endif

static void track_object(void *ptr, size_t request, size_t size, unsigned long ip)
{
    struct objinfo *oinfo;
#if RBTREE
    struct rbtree *node;
#else
    void *node;
#endif

    tracking = false;

    oinfo = real_malloc(sizeof(*oinfo));
    oinfo->ip = ip;
    oinfo->reqsize = request;
    oinfo->size = size;
    oinfo->ptr = ptr;
#if RBTREE
    node = rbtree_insert(&objects, oinfo, objects_compare);
#else
    node = tsearch(oinfo, &objects, objects_compare);
#endif

#if TRACE_OUTPUT
    pid_t tid;
    tid = syscall(SYS_gettid);

    if (node) {
        ++num_objects;
        track(ptr);
        printf("[thread %d] Tracking object @ %p (size=%zu B) {record @ %p}\n", 
                tid, ptr, size, node);
    } else {
        fprintf(stderr, "[thread %d] Failed to track object @ %p (size=%zu B)\n", 
                tid, ptr, size);
    }
#endif
    if (node)
        ++num_objects;
    tracking = true;
}

static void untrack_object(void *ptr)
{
    struct objinfo o = { .ptr = ptr, .ip = 0 };
#if RBTREE
    struct rbtree *node;
#else
    void *node;
#endif
    struct objinfo *objinfo;
#if TRACE_OUTPUT
    pid_t tid;

    tid = syscall(SYS_gettid);
#endif

    tracking = false;
#if RBTREE
    node = rbtree_find(&objects, &o, objects_compare);
#else
    node = tfind(&o, &objects, objects_compare);
#endif

    if (node) {
#if RBTREE
        objinfo = (void *) node->item;
        rbtree_delete(&objects, node);
#else
        objinfo = *(struct objinfo **) node;
        tdelete(&o, &objects, objects_compare);
#endif
        real_free(objinfo);
        --num_objects;
#if TRACE_OUTPUT
        fprintf(stderr, "[thread %d] Untracking object @ %p\n", tid, ptr);
        untrack(ptr);
#endif
    }
    tracking = true;
}

#if RBTREE
void obj_tracker_print_rbtree(const char *filename) {
    FILE *stream;

    stream = filename ? fopen(filename, "a") : tmpfile();

    if (!stream) {
        perror("fopen");
        return;
    }

    rbtree_print(objects, object_print, stream);

    fclose(stream);
}
#endif

/* TODO: use noinline ? */
static long get_ip(void) {
    unw_cursor_t cursor; unw_context_t uc;
    unw_word_t ip = 0;
    /*
    unw_word_t offp;
    char name[256];
    int retval;
    */

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);

    /**
     * [function that called malloc()   ]
     * [malloc()                        ]
     */
    for (int i=0; i<2 && unw_step(&cursor) > 0; ++i) {
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        /*
        retval = unw_get_proc_name(&cursor, name, sizeof(name), &offp);
        if (retval == -UNW_EUNSPEC || retval == -UNW_ENOINFO)
            snprintf(name, sizeof(name), "%s", "??");
        printf("[%10s:0x%0lx] IP = 0x%0lx\n", name, offp, (long) ip);
        */
    }

    return ip;
}

static jmp_buf jump_memcheck;

static void segv_handler(int sig) {
    longjmp(jump_memcheck, 1);
}

/* conditionally access a pointer
 * If we get a segfault, return false
 * Otherwise, return true
 */
static bool memcheck(const void *ptr) {
    bool valid = true;
    __attribute__((unused)) char c;
    struct sigaction action, old_action;

    action.sa_handler = segv_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;

    sigaction(SIGSEGV, &action, &old_action);

    /* setjmp will initially return 0
     * when we dereference ptr, if
     * we get a segfault, our segfault handler
     * will jump back to setjmp and return 1,
     * and control will go to the other statement
     * in this conditional */
    if (!setjmp(jump_memcheck))
        c = *(const char *)ptr;
    else
        valid = false;

    sigaction(SIGSEGV, &old_action, NULL);

    return valid;
}

void *malloc(size_t request) {
    void *ptr;
    static bool inside = false;
    long ip;

    if (inside || destroying)
        return real_malloc(request);

    inside = true;

    if (!initialized)
        obj_tracker_init();
    
    if (real_malloc == NULL) {
        fprintf(stderr, "error: real_malloc is NULL! Was constructor called?\n");
        abort();
    }

    /* We have a valid function pointer to malloc().
     * Now we first do some bookkeeping.
     */
    ptr = real_malloc(sizeof(struct objinfo) + request);

    /* quit early if there was an error */
    if (ptr && tracking) {
        ip = get_ip();
        track_object(ptr, request, 
                malloc_usable_size(ptr) - sizeof(struct objinfo), 
                ip);
    }

    if (ptr && !memcheck(ptr)) {
        fprintf(stderr, "invalid pointer %p\n", ptr);
        abort();
    }


    inside = false;
    return ptr;
}

void free(void *ptr) {
    static bool inside = false;
    static bool reporting = false;

    if (ptr && !memcheck(ptr) && !reporting) {
        reporting = true;
        fprintf(stderr, "invalid pointer %p\n", ptr);
        abort();
    }

    if (inside || destroying) {
        real_free(ptr);
        return;
    }

    inside = true;

    if (!initialized)
        obj_tracker_init();
    if (real_free == NULL) {
        fprintf(stderr, "error: real_free is NULL! Was constructor called?\n");
        abort();
    }

    /* We have a valid function pointer to free(). */
    if (ptr && tracking)
        untrack_object(ptr);

    real_free(ptr);

    inside = false;
}
