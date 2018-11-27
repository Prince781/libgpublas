#include "obj_tracker.h"
#include "oracle.h"
#include "../common.h"
#include <assert.h>
#include <stdlib.h>
#include <inttypes.h>

static bool oracle_trace_loaded;

/* maps [number] -> 8[true (GPU), false] */
static uint8_t *decisions;
static size_t num_decisions;

bool oracle_should_alloc_managed_ptr(bool is_malloc,
                                     uint64_t nth_alloc,
                                     size_t obj_size) {
    if (!oracle_trace_loaded) {
        writef(STDERR_FILENO, "blas2cuda:oracle: please load a file\n");
        abort();
    }

    return nth_alloc < num_decisions
        && decisions[nth_alloc/8] & (1 << (nth_alloc%8));
}

bool oracle_load_file(const char *filename) {
    FILE *f;
    unsigned lineno = 0;;
    char c[2];
    uint64_t nth;
    size_t bufsize = 0;
    size_t ndecisions = 0;
    char *cur_line = NULL;
    size_t line_sz = 0;

    if (!(f = fopen(filename, "r")))
        return false;

    while (getline(&cur_line, &line_sz, f) > 0) {
        lineno++;
        if (sscanf(cur_line, "%1[HD] #%" SCNu64 " %*s\n", c, &nth) != 2) {
            writef(STDERR_FILENO, "oracle_load_file:%s:%u: could not parse line: %m\n", filename, lineno);
            writef(STDERR_FILENO, "note: line %u = %s\n", lineno, cur_line);
            continue;
        }

        ndecisions = MAX(nth, ndecisions);

        if (ndecisions/8 >= bufsize) {
            uint64_t old_bufsize = bufsize;
            bufsize = bufsize ? bufsize * 2 : 1024;
            decisions = internal_realloc(decisions, bufsize);
            memset(&decisions[old_bufsize], 0, bufsize - old_bufsize);
        }

        decisions[nth/8] |= (c[0] == 'D') << (nth%8);
    }

    free(cur_line);

    decisions = internal_realloc(decisions, ndecisions);
    num_decisions = ndecisions;

    if (ferror(f)) {
        fclose(f);
        return false;
    }

    oracle_trace_loaded = true;
    fclose(f);

    for (size_t i = 0; i < ndecisions; ++i) {
    }

    return true;
}
