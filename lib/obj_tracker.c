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
#include <errno.h>
#include "callinfo.h"
#include "obj_tracker.h"

#if DEBUG_TRACKING
#define TRACE_OUTPUT    1
#endif

static void *find_objinfo(struct objinfo *o);


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
#if STANDALONE
static bool tracking = true;    /* initial value */
#else
static bool tracking = false;
#endif
/* whether the user has specified any watchpoints */
static bool watchpoints = false;

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

void obj_tracker_init(bool tracking_enabled)
{
    extern char etext, edata, end;
#if STANDALONE
    char *opt;
    char *option, *saveptr;
#endif

    if (!initialized) {
        tracking = false;
        initialized = true;

        creation_thread = syscall(SYS_gettid);

        obj_tracker_get_fptrs();

#if STANDALONE

        if ((opt = getenv("OBJTRACKER_HELP"))) {
            printf("Object Tracker Options\n"
                   "HELP                ->  shows this message\n"
                   "WATCHPOINTS         ->  a comma-separated list of files\n");
        }

        if ((opt = getenv("OBJTRACKER_WATCHPOINTS"))) {
            option = strtok_r(opt, ",", &saveptr);
            while (option) {
                obj_tracker_load(option, NULL);
                option = strtok_r(NULL, ",", &saveptr);
            }
        }
#endif

        printf("initialized object tracker on thread %d\n", creation_thread);
        printf("segments: {\n"
               "    .text = %10p\n"
               "    .data = %10p\n"
               "    .bss  = %10p\n"
               "    .heap = %10p\n"
               "}\n", &etext, &edata, &end, sbrk(0));
        tracking = tracking_enabled;
    }
}

void obj_tracker_set_tracking(bool enabled)
{
    tracking = enabled;
}

int obj_tracker_load(const char *filename, struct objmngr *mngr)
{
    FILE *fp;
    enum alloc_sym sym;
    long ip;
    size_t reqsize;
    int res;
    void *ptr;
    char *buf = NULL;
    size_t bufsize = 0;
    char *nl = NULL;

    if ((fp = fopen(filename, "r")) == NULL) {
        perror("fopen");
        abort();
    }

    do {
        if ((res = getline(&buf, &bufsize, fp)) != EOF) {
            if ((nl = strchr(buf, '\n')))
                *nl = '\0';
            if ((sym = get_alloc(buf)) == ALLOC_UNKNWN)
                fprintf(stderr, "%s: unsupported symbol '%s'\n", __func__, buf);
            else {
                init_callinfo(sym);
                while ((res = fscanf(fp, "ptr=[%p] reqsize=[%zu] ip=[0x%lx]\n", &ptr, &reqsize, &ip)) == 3) {
                    if (add_callinfo(sym, mngr, ip, reqsize, ptr)) {
                        watchpoints = true;
                        printf("W [%s] reqsize=[%zu] ip=[0x%lx] ptr=[%10p]\n", 
                                buf, reqsize, ip, ptr);
                    } else
                        fprintf(stderr, "failed to add watch: %s\n", strerror(errno));
                }
            }
        }
    } while (res != EOF);

    free(buf);
    return fclose(fp);
}

const struct objinfo *
obj_tracker_objinfo(void *ptr)
{
    struct objinfo fake_objinfo = { .ptr = ptr };
    void *node = find_objinfo(&fake_objinfo);

    if (!node)
        return NULL;

#if RBTREE
    return ((struct rbtree *)node)->item;
#else
    return *(struct objinfo **)node;
#endif
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

static void *insert_objinfo(struct objinfo *oinfo)
{
#if RBTREE
    struct rbtree *node;
    node = rbtree_insert(&objects, oinfo, objects_compare);
#else
    void *node;
    node = tsearch(oinfo, &objects, objects_compare);
#endif

    return node;
}

static void track_object(void *ptr, 
        struct objmngr *mngr, size_t request, size_t size, unsigned long ip)
{
    struct objinfo *oinfo;
    void *node;

    tracking = false;

    oinfo = real_malloc(sizeof(*oinfo));
    memcpy(&oinfo->ci.mngr, mngr, sizeof(*mngr));
    oinfo->ci.ip = ip;
    oinfo->ci.reqsize = request;
    oinfo->size = size;
    oinfo->ptr = ptr;

    node = insert_objinfo(oinfo);

#if TRACE_OUTPUT
    pid_t tid;
    tid = syscall(SYS_gettid);

    if (node) {
        ++num_objects;
        track(ptr);
        printf("T [%p] reqsize=[%zu] ip=[0x%lx] ptr=[%10p] tid=[%d]\n", ptr, request, ip, ptr, tid);
    } else {
        fprintf(stderr, "F [%p] reqsize=[%zu] ip=[0x%lx] ptr=[%10p] tid=[%d]\n", ptr, request, ip, ptr, tid);
    }
#else
    if (node)
        ++num_objects;
#endif
    tracking = true;
}

static void *find_objinfo(struct objinfo *o)
{
#if RBTREE
    struct rbtree *node;
    node = rbtree_find(&objects, &o, objects_compare);
#else
    void *node;
    node = tfind(o, &objects, objects_compare);
#endif
    return node;
}

static void *delete_objinfo(void *node, struct objmngr *mngr)
{
    struct objinfo *objinfo;
    void *result;
#if RBTREE
    objinfo = ((struct rbtree *)node)->item;
    result = rbtree_delete(&objects, node);
#else
    objinfo = *(struct objinfo **) node;
    result = tdelete(objinfo, &objects, objects_compare);
#endif
#if TRACE_OUTPUT
    pid_t tid;

    tid = syscall(SYS_gettid);

    printf("U [%p] reqsize=[%zu] ip=[0x%lx] ptr=[%10p] tid=[%d]\n", 
            objinfo->ptr, objinfo->ci.reqsize, objinfo->ci.ip, objinfo->ptr, tid);
    untrack(objinfo->ptr);
#endif
    memcpy(mngr, &objinfo->ci.mngr, sizeof(*mngr));
    real_free(objinfo);
    return result;
}

static void untrack_object(void *ptr, struct objmngr *mngr)
{
    /* change this if objects_compare changes */
    struct objinfo searchobj = { .ptr = ptr };
    void *node;
    
    tracking = false;

    node = find_objinfo(&searchobj);

    if (node) {
        delete_objinfo(node, mngr);
        --num_objects;
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

static __attribute__((noinline)) long get_ip(int num_steps) {
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
    for (int i=0; i<num_steps && unw_step(&cursor) > 0; ++i) {
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
    static bool inside = false;
    void *ptr;
    void *fake_ptr;
    size_t actual_size;
    long ip;
    struct objmngr mngr;
    struct alloc_callinfo *ci;

    if (inside || destroying)
        return real_malloc(request);

    inside = true;

    if (!initialized)
        obj_tracker_init(tracking);
    
    if (real_malloc == NULL) {
        fprintf(stderr, "error: real_malloc is NULL! Was constructor called?\n");
        abort();
    }

    /*
     * We call malloc(request) to see what malloc
     * would have returned.
     */
    fake_ptr = real_malloc(request);

    /* Only track the object if we are supposed
     * to be tracking it.
     */
    ip = get_ip(2);
    if (tracking && (!watchpoints || (ci = get_callinfo_and(ALLOC_MALLOC, ip, request, fake_ptr)))) {
        if (!watchpoints) {
            mngr.ctor = real_malloc;
            mngr.dtor = real_free;
            mngr.get_size = malloc_usable_size;
        } else /* ci is non-NULL */ {
            memcpy(&mngr, &ci->mngr, sizeof(mngr));
        }

        /*
         * If our memory manager constructor is
         * not libc's malloc(), then we free this
         * memory and use the manager's constructor.
         */
        if (mngr.ctor != real_malloc) {
            real_free(fake_ptr);
            fake_ptr = NULL;
            ptr = mngr.ctor(request);
        } else
            ptr = fake_ptr;
        actual_size = mngr.get_size(ptr);
        track_object(ptr, &mngr, request, actual_size, ip);
    } else
        ptr = fake_ptr;

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
    struct objmngr mngr;

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
        obj_tracker_init(tracking);
    if (real_free == NULL) {
        fprintf(stderr, "error: real_free is NULL! Was constructor called?\n");
        abort();
    }

    /* We have a valid function pointer to free(). */
    if (ptr && tracking) {
        /*
         * Set our defaults in case we
         * fail to set mngr.
         */
        mngr.ctor = real_malloc;
        mngr.dtor = real_free;
        untrack_object(ptr, &mngr);
        /*
         * mngr may be something else now
         */
        mngr.dtor(ptr);
    } else
        real_free(ptr);

    inside = false;
}
