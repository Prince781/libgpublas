/*****
 * obj_tracker.c
 * Tracks objects.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <malloc.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <search.h>         /* for red-black tree */
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#include <pthread.h>        /* for read-write locks */
#include "obj_tracker.h"
#include "../common.h"
#if STANDALONE
#include "blas_tracker.h"
#endif
#include <assert.h>
#include <err.h>

#if DEBUG_TRACKING
#define TRACE_OUTPUT    1
#endif

static void *find_objinfo(struct objinfo *o);

#if STANDALONE
struct obj_options objtracker_options = {
    .only_print_calls = false,
    .blas_libs = NULL,
    .num_blas = 0
};
#endif

#ifndef STANDALONE
extern struct objmngr blas2cuda_manager;
#endif

static bool initialized = false;
static bool initializing = false;
static bool destroying = false;
static pid_t creation_thread = 0;
static void *objects = NULL;
static unsigned long num_objects = 0;
static uint64_t next_uid = 0;

static pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

static __thread bool grabbed_reader_lock = false;
static __thread bool grabbed_writer_lock = false;
static __thread bool inside_internal = false;

static inline void read_lock(void) {
    int ret;

    assert (!grabbed_writer_lock);
    assert (!grabbed_reader_lock);

    while ((ret = pthread_rwlock_rdlock(&rwlock)) < 0
            && errno == EAGAIN)
        writef(STDERR_FILENO, "objtracker: failed to acquire read lock\n");
    if (ret < 0) {
        writef(STDERR_FILENO, "objtracker: failed to acquire read lock: %s", strerror(errno));
        abort();
    }
}

static inline void write_lock(void) {
    assert (!grabbed_reader_lock);
    assert (!grabbed_writer_lock);

    if (pthread_rwlock_wrlock(&rwlock) < 0) {
        writef(STDERR_FILENO, "objtracker: failed to acquire write lock: %s", strerror(errno));
        abort();
    }
}

static inline void unlock(void) {
    if (pthread_rwlock_unlock(&rwlock) < 0) {
        writef(STDERR_FILENO, "objtracker: failed to unlock write lock: %s", strerror(errno));
        abort();
    }
}

/* allows us to temporarily disable tracking */
#if STANDALONE
static bool tracking = true;    /* initial value */
#else
static bool tracking = false;
#endif

void *__libc_malloc(size_t);
void *__libc_calloc(size_t, size_t);
void *__libc_realloc(void *, size_t);
void __libc_free(void *);

void *(*real_malloc)(size_t);
void *(*real_calloc)(size_t, size_t);
void *(*real_realloc)(void *, size_t);
void (*real_free)(void *);

struct objmngr glibc_manager;

static void get_real_free(void);

static void get_real_malloc(void) {
    static bool inside = false;

    if (!inside) {
        inside = true;
        if (real_malloc == NULL) {
            dlerror();
            writef(STDOUT_FILENO, "Reset dlerror\n");
            real_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
            if (real_malloc == NULL) {
                writef(STDERR_FILENO, "dlsym: %s\n", dlerror());
                abort();
            } else
                writef(STDOUT_FILENO, "Got malloc\n");
        } else
            writef(STDOUT_FILENO, "malloc already found\n");
        get_real_free();
        inside = false;
    }
}

static void get_real_calloc(void) {
    static bool inside = false;

    if (!inside) {
        inside = true;
        if (real_calloc == NULL) {
            dlerror();
            writef(STDOUT_FILENO, "Reset dlerror\n");
            real_calloc = (void *(*)(size_t, size_t)) dlsym(RTLD_NEXT, "calloc");
            if (real_calloc == NULL) {
                writef(STDERR_FILENO, "dlsym: %s\n", dlerror());
                abort();
            } else
                writef(STDOUT_FILENO, "Got calloc\n");
        } else
            writef(STDOUT_FILENO, "calloc already found\n");
        get_real_free();
        inside = false;
    }
}

static void get_real_realloc(void) {
    static bool inside = false;
    
    if (!inside) {
        inside = true;
        if (real_realloc == NULL) {
            dlerror();
            writef(STDOUT_FILENO, "Reset dlerror\n");
            real_realloc = (void *(*)(void *, size_t)) dlsym(RTLD_NEXT, "realloc");
            if (real_realloc == NULL) {
                writef(STDERR_FILENO, "dlsym: %s\n", dlerror());
                abort();
            } else
                writef(STDOUT_FILENO, "Got realloc\n");
        } else
            writef(STDOUT_FILENO, "realloc already found\n");
        get_real_free();
        inside = false;
    }
}

static void get_real_free(void) {
    static bool inside = false;

    if (!inside) {
        inside = true;
        if (real_free == NULL) {
            dlerror();
            writef(STDOUT_FILENO, "Reset dlerror\n");
            real_free = (void (*)(void *)) dlsym(RTLD_NEXT, "free");
            if (real_free == NULL) {
                writef(STDERR_FILENO, "dlsym: %s\n", dlerror());
                abort();
            } else
                writef(STDOUT_FILENO, "Got free\n");
        } else
            writef(STDOUT_FILENO, "free already found\n");
        inside = false;
    }
}

bool obj_tracker_should_alloc_managed_ptr(void) {
    // TODO
    return false;
}

#if STANDALONE
void obj_tracker_print_help(void) {
    writef(STDERR_FILENO,
        "object tracker options: (OBJTRACKER_OPTIONS)\n"
	"You can chain these options with a semicolon (;)\n"
	"help               -- Print help.\n"
	"only_print_calls   -- Only print BLAS calls that use \n"
	"                      objects. This reduces the output of\n"
	"                      the tracker. By default, this option\n"
	"                      is set to 'false'.\n"
	"blas_libs          -- A comma(,)-separated list of libraries\n"
	"                      to load, in the order specified. Use \n"
	"                      this for libraries that are not linked\n"
	"                      to their dependencies, like Intel MKL.\n");
}

const char my_interp[] __attribute__((section(".interp"))) = "/lib64/ld-linux-x86-64.so.2";

int entry(void) {
    obj_tracker_print_help();
    exit(0);
}

static void obj_tracker_get_options(void) {
    char *options = getenv("OBJTRACKER_OPTIONS");
    char *saveptr = NULL;
    char *option = NULL;
    bool help = false;

    if (!options)
        return;

    option = strtok_r(options, ";", &saveptr);
    while (option != NULL) {
        if (strcmp(option, "help") == 0) {
            if (!help) {
		obj_tracker_print_help();
                help = true;
            }
        } else if (strcmp(option, "only_print_calls") == 0) {
            objtracker_options.only_print_calls = true;
            writef(STDERR_FILENO, "object tracker: will only print calls\n");
        } else if (strncmp(option, "blas_libs=", 10) == 0) {
            char *blaslibs = strchr(option, '=');
            char *blas_lib = NULL;
            char *saveptr_bs = NULL;
            int size = 1;
            char **array = real_calloc(size, sizeof(*array));
            int n = 0;
            if (blaslibs) {
                blaslibs++;
                blas_lib = strtok_r(blaslibs, ",", &saveptr_bs);
                while (blas_lib != NULL) {
                    if (n >= size) {
                        size *= 2;
                        array = real_realloc(array, sizeof(*array) * size);
                    }
                    array[n] = strdup(blas_lib);
                    ++n;
                    blas_lib = strtok_r(NULL, ",", &saveptr_bs);
                }
                array = real_realloc(array, sizeof(*array) * n);
                objtracker_options.blas_libs = array;
                objtracker_options.num_blas = n;
            } else
                writef(STDERR_FILENO, "objtracker: you must provide a filename. Set OBJTRACKER_OPTIONS=help.\n");
        } else {
            writef(STDERR_FILENO, "objtracker: unknown option '%s'. Set OBJTRACKER_OPTIONS=help.\n", option);
        }
        option = strtok_r(NULL, ";", &saveptr);
    }
}
#endif

static void obj_tracker_get_fptrs(void) {
    static bool success = false;

    if (!success) {
        get_real_malloc();
        get_real_calloc();
        get_real_realloc();
        get_real_free();
        glibc_manager = (struct objmngr) {
            .ctor = real_malloc,
            .cctor = real_calloc,
            .realloc = real_realloc,
            .dtor = real_free,
            .get_size = malloc_usable_size
        };
        success = true;
    }
}

void obj_tracker_print_info(enum objprint_type type, const struct objinfo *info)
{
    char c;
    const char *fun_name;
    pid_t tid;

    switch (type) {
        case OBJPRINT_TRACK:
            c = 'T';
            break;
        case OBJPRINT_TRACK_FAIL:
            c = 'F';
            break;
        case OBJPRINT_UNTRACK:
            c = 'U';
            break;
        case OBJPRINT_CALL:
            c = 'C';
            break;
        default:
            c = '?';
            break;
    }

    switch (info->ci.alloc) {
        case ALLOC_MALLOC:
            fun_name = (type != OBJPRINT_CALL) ? "malloc" : "0";
            break;
        case ALLOC_CALLOC:
            fun_name = (type != OBJPRINT_CALL) ? "calloc" : "1";
            break;
        default:
            fun_name = "?";
            break;
    }

    tid = syscall(SYS_gettid);

    struct timespec tm = info->time;

    if (type != OBJPRINT_TRACK)
        clock_gettime(CLOCK_MONOTONIC_RAW, &tm);

#if STANDALONE
    if (!objtracker_options.only_print_calls || type == OBJPRINT_CALL)
#endif
        writef(STDOUT_FILENO, "%c [%p] fun=[%s] reqsize=[%zu] tid=[%d] time=[%lds+%ldns] uid=[%lu]\n",
                c, info->ptr, fun_name, info->ci.reqsize, tid, 
                tm.tv_sec, tm.tv_nsec, info->uid);
}

#if STANDALONE
__attribute__((constructor))
#endif
void obj_tracker_init(bool tracking_enabled)
{
/*    extern char etext, edata, end; */
    if (!initialized && !initializing) {
        tracking = false;
        initializing = true;

        creation_thread = syscall(SYS_gettid);

        writef(STDOUT_FILENO, "objtracker: Initializing...\n");
        if (pthread_rwlock_init(&rwlock, NULL) < 0) {
            writef(STDERR_FILENO, "Failed to initialize rwlock");
            abort();
        }

        obj_tracker_get_fptrs();
        initialized = true;

#if STANDALONE
        obj_tracker_get_options();
        blas_tracker_init();
#endif
        tracking = tracking_enabled;
        initializing = false;
    }
}

void obj_tracker_set_tracking(bool enabled)
{
    tracking = enabled;
}

int obj_tracker_load(const char *filename, struct objmngr *mngr)
{
    warn("unimplemented");
    return 0;
}

const struct objinfo *
obj_tracker_objinfo(void *ptr)
{
    struct objinfo fake_objinfo = { .ptr = ptr };
    void *node = find_objinfo(&fake_objinfo);

    if (!node)
        return NULL;

    return *(struct objinfo **)node;
}

#if STANDALONE
__attribute__((destructor))
#endif
void obj_tracker_fini(void) {
    pid_t tid;
    if (initialized && !destroying) {
        destroying = true;

        if (pthread_rwlock_destroy(&rwlock) < 0) {
            writef(STDERR_FILENO, "objtracker: Failed to destroy rwlock: %s", strerror(errno));
            abort();
        }

        tdestroy(objects, real_free);
        objects = NULL;

#if STANDALONE
        blas_tracker_fini();
#endif
        tid = syscall(SYS_gettid);
        writef(STDERR_FILENO, "objtracker: decommissioned on thread %d\n", tid);
        initialized = false;
        destroying = false;
    }
}

static int objects_compare(const void *o1, const void *o2) {
    return ((const struct objinfo *)o1)->ptr - ((const struct objinfo *)o2)->ptr;
}

static void *insert_objinfo(struct objinfo *oinfo)
{
    void *node;
    bool saved_grabbed_reader_lock = grabbed_reader_lock;
    
    inside_internal = true;

    if (grabbed_reader_lock) {
        unlock();
        grabbed_reader_lock = false;
    }

    write_lock();
    grabbed_writer_lock = true;

    node = tsearch(oinfo, &objects, objects_compare);

    if (node) {
        ++num_objects;
    }

    unlock();
    grabbed_writer_lock = false;

    if (saved_grabbed_reader_lock) {
        read_lock();
        grabbed_reader_lock = true;
    }

    inside_internal = false;

    return node;
}

static void track_object(void *ptr, 
        enum alloc_sym sym,
        struct objmngr *mngr, size_t request, size_t size)
{
    struct objinfo *oinfo;
#if TRACE_OUTPUT
    void *node;
#else
    __attribute__((unused)) void *node;
#endif

    tracking = false;

    oinfo = real_malloc(sizeof(*oinfo));
    memcpy(&oinfo->ci.mngr, mngr, sizeof(*mngr));
    oinfo->ci.alloc = sym;
    oinfo->ci.reqsize = request;
    oinfo->size = size;
    oinfo->ptr = ptr;
    clock_gettime(CLOCK_MONOTONIC_RAW, &oinfo->time);
    oinfo->uid = __sync_fetch_and_add(&next_uid, 1);

    node = insert_objinfo(oinfo);

#if TRACE_OUTPUT
    obj_tracker_print_info(node ? OBJPRINT_TRACK : OBJPRINT_TRACK_FAIL, oinfo);
#endif

    tracking = true;
}

static void *find_objinfo(struct objinfo *o)
{
    void *node;

    inside_internal = true;

    if (!grabbed_writer_lock) {
        read_lock();
        grabbed_reader_lock = true;
    }

    node = tfind(o, &objects, objects_compare);

    if (!grabbed_writer_lock) {
        unlock();
        grabbed_reader_lock = false;
    }

    inside_internal = false;

    return node;
}

static void *delete_objinfo(void *node, struct objmngr *mngr)
{
    struct objinfo *objinfo;
    void *result;

    bool saved_grabbed_reader_lock = grabbed_reader_lock;

    inside_internal = true;

    if (grabbed_reader_lock) {
        unlock();
        grabbed_reader_lock = false;
    }

    write_lock();
    grabbed_writer_lock = true;

    objinfo = *(struct objinfo **) node;
    result = tdelete(objinfo, &objects, objects_compare);

    unlock();
    grabbed_writer_lock = false;

    if (saved_grabbed_reader_lock) {
        read_lock();
        grabbed_reader_lock = true;
    }

#if TRACE_OUTPUT
    obj_tracker_print_info(OBJPRINT_UNTRACK, objinfo);
#endif
    memcpy(mngr, &objinfo->ci.mngr, sizeof(*mngr));
    real_free(objinfo);

    inside_internal = false;

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

void *malloc(size_t request) {
    static __thread bool inside = false;
    void *ptr;
    size_t actual_size;
    struct objmngr mngr;

    if (inside || inside_internal || destroying || initializing || !tracking || !initialized) {
        if (!real_malloc)
            get_real_malloc();
        if (!real_malloc) {
            writef(STDOUT_FILENO, "malloc: returning NULL because real_malloc is undefined\n");
            return NULL;
        }
        return real_malloc(request);
    }

    if (!request)
        return NULL;

    inside = true;

    if (real_malloc == NULL) {
        writef(STDERR_FILENO, "error: real_malloc is NULL! Was constructor called?\n");
        abort();
    }

    if (tracking) {
#if STANDALONE
        mngr = glibc_manager;
#else
        if (obj_tracker_should_alloc_managed_ptr()) {
            mngr = blas2cuda_manager;
        } else {
            mngr = glibc_manager;
        }
#endif

        ptr = mngr.ctor(request);
        actual_size = mngr.get_size(ptr);
        track_object(ptr, ALLOC_MALLOC, &mngr, request, actual_size);
    } else
        ptr = real_malloc(request);

    inside = false;
    return ptr;
}

void *calloc(size_t nmemb, size_t size) {
    static __thread bool inside = false;
    size_t request;
    void *ptr;
    size_t actual_size;
    struct objmngr mngr;

    request = nmemb * size;
    if (nmemb != 0 && request / nmemb != size) {
        errno = ERANGE;
        return NULL;
    }

    if (inside || inside_internal || destroying || initializing || !tracking || !initialized) {
        if (!real_calloc)
            get_real_calloc();
        if (!real_calloc) {
            writef(STDOUT_FILENO, "calloc: returning NULL because real_calloc is undefined\n");
            return NULL;
        }
        return real_calloc(nmemb, size);
    }

    if (!nmemb || !size)
        return real_calloc(nmemb, size);

    inside = true;

    if (real_calloc == NULL) {
        writef(STDERR_FILENO, "error: real_calloc is NULL! Was constructor called?\n");
        abort();
    }

    if (tracking) {
#if STANDALONE
        mngr = glibc_manager;
#else
        if (obj_tracker_should_alloc_managed_ptr()) {
            mngr = blas2cuda_manager;
        } else {
            mngr = glibc_manager;
        }
#endif

        ptr = mngr.cctor(nmemb, size);
        actual_size = mngr.get_size(ptr);
        track_object(ptr, ALLOC_CALLOC, &mngr, request, actual_size);
    } else
        ptr = real_calloc(nmemb, size);

    inside = false;
    return ptr;
}

void *realloc(void *ptr, size_t size) {
    static __thread bool inside = false;
    void *new_ptr;
    const struct objinfo *ptr_info;
    if (!size)
        return NULL;

    if (inside || inside_internal || destroying || initializing || !tracking || !initialized) {
        if (!real_realloc)
            get_real_realloc();
        if (!real_realloc) {
            writef(STDOUT_FILENO,"realloc: returning NULL because real_realloc is undefined\n");
            return NULL;
        }
        return real_realloc(ptr, size);
    }

    if ((ptr_info = obj_tracker_objinfo(ptr))) {
        new_ptr = ptr_info->ci.mngr.realloc(ptr, size);
    } else
        new_ptr = real_realloc(ptr, size);

    inside = false;
    return new_ptr;
}

void free(void *ptr) {
    static __thread bool inside = false;
    struct objmngr mngr;

    if (inside || inside_internal || destroying || initializing || !initialized) {
        if (real_free)
            real_free(ptr);
        return;
    }

    inside = true;

    if (real_free == NULL) {
        writef(STDERR_FILENO, "error: real_free is NULL! Was constructor called?\n");
        abort();
    }

    /* We have a valid function pointer to free(). */
    if (ptr && tracking) {
        /*
         * Set our defaults in case we
         * fail to set mngr.
         */
        mngr = glibc_manager;
        untrack_object(ptr, &mngr);
        /*
         * mngr may be something else now
         */
        assert(mngr.dtor != NULL);
        mngr.dtor(ptr);
    } else
        real_free(ptr);

    inside = false;
}
