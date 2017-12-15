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
#include <pthread.h>        /* for read-write locks */
#include "callinfo.h"
#include "obj_tracker.h"
#if STANDALONE
#include "blas_tracker.h"
#endif

#if DEBUG_TRACKING
#define TRACE_OUTPUT    1
#endif

#define write_conststr(fd, str) \
{                                               \
    int fild = fd;                              \
    syscall(SYS_write, fild, str, sizeof(str)); \
}

#define write_str(fd, str)              \
{                                       \
    int fild = fd;                      \
    const char *_str = (str);           \
    size_t sz = strlen(_str);           \
    syscall(SYS_write, fild, _str, sz); \
}

static void *find_objinfo(struct objinfo *o);

#if STANDALONE
struct obj_options objtracker_options = {
    .only_print_calls = false,
    .blas_libs = NULL,
    .num_blas = 0
};
#endif


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
static bool initializing = false;
static bool destroying = false;
static pid_t creation_thread = 0;
#if RBTREE
static struct rbtree *objects = NULL;
#else
static void *objects = NULL;
#endif
static unsigned long num_objects = 0;

static pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

static inline void read_lock(void) {
    /*
    int ret;
    while ((ret = pthread_rwlock_rdlock(&rwlock)) < 0
            && errno == EAGAIN)
        write_str("failed to acquire read lock\n");
    if (ret < 0) {
        perror("Failed to acquire read lock");
        abort();
    }
    */
}

static inline void write_lock(void) {
    /*
    if (pthread_rwlock_wrlock(&rwlock) < 0) {
        perror("Failed to acquire write lock");
        abort();
    }
    */
}

static inline void unlock(void) {
    /*
    if (pthread_rwlock_unlock(&rwlock) < 0) {
        perror("Failed to unlock write lock");
        abort();
    }
    */
}

/* allows us to temporarily disable tracking */
#if STANDALONE
static bool tracking = true;    /* initial value */
#else
static bool tracking = false;
#endif
/* whether the user has specified any watchpoints */
static bool watchpoints = false;

void *__libc_malloc(size_t);
void *__libc_calloc(size_t, size_t);
void *__libc_realloc(void *, size_t);
void __libc_free(void *);

void *(*real_malloc)(size_t);
void *(*real_calloc)(size_t, size_t);
void *(*real_realloc)(void *, size_t);
void (*real_free)(void *);

static void get_real_free(void);

static void get_real_malloc(void) {
    static bool inside = false;

    if (!inside) {
        inside = true;
        if (real_malloc == NULL) {
            dlerror();
            write_str(STDOUT_FILENO, "Reset dlerror\n");
            real_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
            if (real_malloc == NULL) {
                write_conststr(STDERR_FILENO, "dlsym: ");
                write_str(STDERR_FILENO, dlerror());
                write_conststr(STDERR_FILENO, "\n");
                abort();
            } else
                write_str(STDOUT_FILENO, "Got malloc\n");
        } else
            write_str(STDOUT_FILENO, "malloc already found\n");
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
            write_str(STDOUT_FILENO, "Reset dlerror\n");
            real_calloc = (void *(*)(size_t, size_t)) dlsym(RTLD_NEXT, "calloc");
            if (real_calloc == NULL) {
                write_conststr(STDERR_FILENO, "dlsym: ");
                write_str(STDERR_FILENO, dlerror());
                write_conststr(STDERR_FILENO, "\n");
                abort();
            } else
                write_str(STDOUT_FILENO, "Got calloc\n");
        } else
            write_str(STDOUT_FILENO, "calloc already found\n");
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
            write_str(STDOUT_FILENO, "Reset dlerror\n");
            real_realloc = (void *(*)(void *, size_t)) dlsym(RTLD_NEXT, "realloc");
            if (real_realloc == NULL) {
                write_conststr(STDERR_FILENO, "dlsym: ");
                write_str(STDERR_FILENO, dlerror());
                write_conststr(STDERR_FILENO, "\n");
                abort();
            } else
                write_str(STDOUT_FILENO, "Got realloc\n");
        } else
            write_str(STDOUT_FILENO, "realloc already found\n");
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
            write_str(STDOUT_FILENO, "Reset dlerror\n");
            real_free = (void (*)(void *)) dlsym(RTLD_NEXT, "free");
            if (real_free == NULL) {
                write_conststr(STDERR_FILENO, "dlsym: ");
                write_str(STDERR_FILENO, dlerror());
                write_conststr(STDERR_FILENO, "\n");
                abort();
            } else
                write_str(STDOUT_FILENO, "Got free\n");
        } else
            write_str(STDOUT_FILENO, "free already found\n");
        inside = false;
    }
}

#if STANDALONE
void obj_tracker_print_help(void) {
    fprintf(stderr,
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
            printf("object tracker: will only print calls\n");
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
                fprintf(stderr, "objtracker: you must provide a filename. Set OBJTRACKER_OPTIONS=help.\n");
        } else {
            fprintf(stderr, "objtracker: unknown option '%s'. Set OBJTRACKER_OPTIONS=help.\n", option);
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
        success = true;
    }
}

void obj_tracker_print_info(enum objprint_type type, const struct objinfo *info)
{
    char c;
    const char *fun_name;
    char ip_offs_str[1 + N_IP_OFFS * 8];
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

    ip_offs_tostr(&info->ci.offs, ip_offs_str);

    tid = syscall(SYS_gettid);

#if STANDALONE
    if (!objtracker_options.only_print_calls || type == OBJPRINT_CALL)
#endif
        printf("%c [%p] fun=[%s] reqsize=[%zu] ip_offs=[%s] tid=[%d]\n",
                c, info->ptr, fun_name, info->ci.reqsize, ip_offs_str, tid);
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

        write_str(STDOUT_FILENO, "Initializing object tracker...\n");
        write_str(STDOUT_FILENO, "Initializing lock...\n");
        if (pthread_rwlock_init(&rwlock, NULL) < 0) {
            write_str(STDERR_FILENO, "Failed to initialize rwlock");
            abort();
        }
        write_str(STDOUT_FILENO, "Lock initialized ...\n");

        write_str(STDOUT_FILENO, "Getting function pointers...\n");
        obj_tracker_get_fptrs();
        write_str(STDOUT_FILENO, "Got function pointers...\n");
        initialized = true;

        write_str(STDOUT_FILENO, "Initializing call info...\n");
        init_callinfo();
        write_str(STDOUT_FILENO, "Initialized call info...\n");

#if STANDALONE
        obj_tracker_get_options();
        blas_tracker_init();
#endif

        /*
        printf("initialized object tracker on thread %d\n", creation_thread);
        printf("segments: {\n"
               "    .text = %10p\n"
               "    .data = %10p\n"
               "    .bss  = %10p\n"
               "    .heap = %10p\n"
               "}\n", &etext, &edata, &end, sbrk(0));
               */
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
    FILE *fp = NULL;
    enum alloc_sym sym;
    size_t reqsize;
    int res = 0;
    int ci_res = 0;
    char *line = NULL;
    size_t line_len = 0;
    const char *symname = NULL;
    /* char *off_str = NULL; */
    struct ip_offs offs;
    int lineno = 0;

    if ((fp = fopen(filename, "r")) == NULL) {
        perror("fopen");
        abort();
    }

    while ((res = getline(&line, &line_len, fp)) != EOF) {
        char *nl;
        if ((nl = strchr(line, '\n')))
            *nl = '\0';
        ++lineno;
        if ((res = sscanf(line, "fun=[%d] reqsize=[%zu] ip_offs=[%hx.%hx.%hx.%hx]\n", 
                        (int *) &sym, &reqsize, 
                        &offs.off[0], &offs.off[1],
                        &offs.off[2], &offs.off[3])) == 6) {
            if (sym < 0 || sym > N_ALLOC_SYMS) {
                fprintf(stderr, "%s: unsupported symbol '%d'\n",
                        filename, sym);
                continue;
            }
            symname = alloc_sym_tostr(sym);
            /* if (ip_offs_parse(off_str, &offs) != N_IP_OFFS) {
                fprintf(stderr, "%s: Failed to add watch: could not parse IP offset"
                            " string '%s' on line %d\n", filename, off_str, lineno);
            } else */
            if ((ci_res = add_callinfo(sym, mngr, &offs, reqsize)) == 0) {
                watchpoints = true;
                printf("W fun=[%s] reqsize=[%zu] ip_offs=[%hx.%hx.%hx.%hx]\n",
                        symname, reqsize, 
                        offs.off[0], offs.off[1], offs.off[2], offs.off[3]);
            } else if (ci_res < 0)
                fprintf(stderr, "Failed to add watch: %s\n", strerror(errno));
        } else {
            fprintf(stderr, "%s: Could not parse line %d (matched %d items)\n", 
                    filename, lineno, res);
            if (res > 0)
                fprintf(stderr, "\tgot fun=%s\n", symname);
            if (res > 1)
                fprintf(stderr, "\tgot reqsize=%zu\n", reqsize);
        }
        /*
        free(symname);
        symname = NULL;
        */
        /*
        free(off_str);
        off_str = NULL;
        */
    }

    free(line);
    line = NULL;
    line_len = 0;


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

#if STANDALONE
__attribute__((destructor))
#endif
void obj_tracker_fini(void) {
    pid_t tid;
    if (initialized && !destroying) {
        destroying = true;

        if (pthread_rwlock_destroy(&rwlock) < 0) {
            perror("Failed to destroy rwlock");
            abort();
        }
#if RBTREE
        rbtree_destroy(&objects, object_destroy, NULL);
#else
        tdestroy(objects, real_free);
#endif
        objects = NULL;

#if STANDALONE
        blas_tracker_fini();
#endif
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
    void *node;

    write_lock();
#if RBTREE
    node = rbtree_insert(&objects, oinfo, objects_compare);
#else
    node = tsearch(oinfo, &objects, objects_compare);
#endif

    if (node) {
        ++num_objects;
    }

    unlock();

    return node;
}

static void track_object(void *ptr, 
        enum alloc_sym sym,
        struct objmngr *mngr, size_t request, size_t size,
        const struct ip_offs *offs)
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
    oinfo->ci.offs = *offs;
    oinfo->ci.reqsize = request;
    oinfo->size = size;
    oinfo->ptr = ptr;

    node = insert_objinfo(oinfo);

#if TRACE_OUTPUT
    if (node) {
        track(ptr);
        obj_tracker_print_info(OBJPRINT_TRACK, oinfo);
    } else {
        obj_tracker_print_info(OBJPRINT_TRACK_FAIL, oinfo);
    }
#endif

    tracking = true;
}

static void *find_objinfo(struct objinfo *o)
{
    void *node;

    read_lock();
#if RBTREE
    node = rbtree_find(&objects, &o, objects_compare);
#else
    node = tfind(o, &objects, objects_compare);
#endif
    unlock();
    return node;
}

static void *delete_objinfo(void *node, struct objmngr *mngr)
{
    struct objinfo *objinfo;
    void *result;

    write_lock();

#if RBTREE
    objinfo = ((struct rbtree *)node)->item;
    result = rbtree_delete(&objects, node);
#else
    objinfo = *(struct objinfo **) node;
    result = tdelete(objinfo, &objects, objects_compare);
#endif
    unlock();
#if TRACE_OUTPUT
    obj_tracker_print_info(OBJPRINT_UNTRACK, objinfo);
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

#define WORDS_BEFORE_RBP    4

static __attribute__((noinline)) 
struct ip_offs *get_ip_offs(struct ip_offs *offs) 
{
    int n_found = 0;
    unw_cursor_t cursor; unw_context_t uc;
    unw_word_t ip = 0;
    /* unw_word_t offp; */
    int retval;
    /* char buf[2]; */

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);

    /**
     * [function that called ?alloc()   ]
     * [?alloc()                        ]
     * [this function                   ]
     */
    for (int i=0; i<1+N_IP_OFFS && unw_step(&cursor) > 0; ++i) {
        retval = unw_get_reg(&cursor, UNW_REG_IP, &ip);
        /* retval = unw_get_proc_name(&cursor, buf, 0, &offp); */
        if (i == 0)
            /* skip this offset */
            continue;
        if (retval != -UNW_EUNSPEC && retval != -UNW_ENOINFO) {
            offs->off[n_found] = ip; /* offp; */
            if (n_found > 0)
                offs->off[n_found-1] -= ip;
            ++n_found;
        }
        /*
        if (retval == -UNW_EUNSPEC || retval == -UNW_ENOINFO)
            snprintf(name, sizeof(name), "%s", "??");
        printf("[%10s:0x%0lx] IP = 0x%0lx\n", name, offp, (long) ip);
        */
    }

    return offs;
}


void *malloc(size_t request) {
    static bool inside = false;
    void *ptr;
    /*
    void *fake_ptr;
    */
    size_t actual_size;
    struct ip_offs offs;
    struct objmngr mngr;
    struct alloc_callinfo *ci;

    if (inside || destroying || initializing || !tracking || !initialized) {
        if (!real_malloc)
            get_real_malloc();
        if (!real_malloc) {
            write_str(STDOUT_FILENO, "malloc: returning NULL because real_malloc is undefined\n");
            return NULL;
        }
        return real_malloc(request);
    }

    if (!request)
        return NULL;

    inside = true;

    if (real_malloc == NULL) {
        fprintf(stderr, "error: real_malloc is NULL! Was constructor called?\n");
        abort();
    }

    /* Only track the object if we are supposed
     * to be tracking it.
     */
    if (tracking 
            && (!watchpoints 
                || 
                (get_callinfo_reqsize(ALLOC_MALLOC, request) 
                 && (ci = get_callinfo_and(ALLOC_MALLOC, get_ip_offs(&offs), request))))
        ) {
        if (!watchpoints) {
            mngr.ctor = real_malloc;
            mngr.cctor = real_calloc;
            mngr.realloc = real_realloc;
            mngr.dtor = real_free;
            mngr.get_size = malloc_usable_size;
            get_ip_offs(&offs);
        } else /* ci is non-NULL */ {
            memcpy(&mngr, &ci->mngr, sizeof(mngr));
        }

        /*
         * If our memory manager constructor is
         * not libc's malloc(), then we free this
         * memory and use the manager's constructor.
         */
        if (mngr.ctor != real_malloc) {
            ptr = mngr.ctor(request);
        } else
            ptr = real_malloc(request);
        actual_size = mngr.get_size(ptr);
        track_object(ptr, ALLOC_MALLOC, &mngr, request, actual_size, &offs);
    } else
        ptr = real_malloc(request);

    if (ptr && !memcheck(ptr)) {
        fprintf(stderr, "invalid pointer %p\n", ptr);
        abort();
    }


    inside = false;
    return ptr;
}

void *calloc(size_t nmemb, size_t size) {
    static bool inside = false;
    size_t request;
    void *ptr;
    size_t actual_size;
    struct ip_offs offs;
    struct objmngr mngr;
    struct alloc_callinfo *ci;

    request = nmemb * size;
    if (request / nmemb != size) {
        errno = ERANGE;
        return NULL;
    }

    if (inside || destroying || initializing || !tracking || !initialized) {
        if (!real_calloc)
            get_real_calloc();
        if (!real_calloc) {
            write_str(STDOUT_FILENO, "calloc: returning NULL because real_calloc is undefined\n");
            return NULL;
        }
        return real_calloc(nmemb, size);
    }

    if (!nmemb || !size)
        return real_calloc(nmemb, size);

    inside = true;

    if (real_calloc == NULL) {
        fprintf(stderr, "error: real_calloc is NULL! Was constructor called?\n");
        abort();
    }

    /* Only track the object if we are supposed
     * to be tracking it.
     */
    if (tracking 
            && (!watchpoints 
                || 
                (get_callinfo_reqsize(ALLOC_CALLOC, request)
                 && (ci = get_callinfo_and(ALLOC_CALLOC, get_ip_offs(&offs), request))))
        ) {
        if (!watchpoints) {
            mngr.ctor = real_malloc;
            mngr.cctor = real_calloc;
            mngr.realloc = real_realloc;
            mngr.dtor = real_free;
            mngr.get_size = malloc_usable_size;
            get_ip_offs(&offs);
        } else /* ci is non-NULL */ {
            memcpy(&mngr, &ci->mngr, sizeof(mngr));
        }

        /*
         * If our memory manager constructor is
         * not libc's calloc(), then we free this
         * memory and use the manager's constructor.
         */
        if (mngr.cctor != real_calloc) {
            ptr = mngr.cctor(nmemb, size);
        } else
            ptr = real_calloc(nmemb, size);
        actual_size = mngr.get_size(ptr);
        track_object(ptr, ALLOC_CALLOC, &mngr, request, actual_size, &offs);
    } else
        ptr = real_calloc(nmemb, size);

    if (ptr && !memcheck(ptr)) {
        fprintf(stderr, "invalid pointer %p\n", ptr);
        abort();
    }


    inside = false;
    return ptr;
}

void *realloc(void *ptr, size_t size) {
    static bool inside = false;
    void *new_ptr;
    const struct objinfo *ptr_info;
    if (!size)
        return NULL;

    if (inside || destroying || initializing || !tracking || !initialized) {
        if (!real_realloc)
            get_real_realloc();
        if (!real_realloc) {
            write_str(STDOUT_FILENO, 
                    "realloc: returning NULL because real_realloc is undefined\n");
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
    static bool inside = false;
    static bool reporting = false;
    struct objmngr mngr;

    if (ptr && !memcheck(ptr) && !reporting) {
        reporting = true;
        fprintf(stderr, "invalid pointer %p\n", ptr);
        abort();
    }

    if (inside || destroying || initializing) {
        if (real_free)
            real_free(ptr);
        return;
    }

    inside = true;

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
        mngr.cctor = real_calloc;
        mngr.realloc = real_realloc;
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
