#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t load_file(const char *filename, char ***lines) {
    size_t nlines = 0;
    size_t bufsize = 1024;
    size_t line_sz = 0;
    FILE *textfile;

    if ((textfile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "failed to open `%s': %m\n", filename);
        return 0;
    }

    *lines = realloc(*lines, bufsize * sizeof(**lines));
    memset(*lines, 0, bufsize * sizeof(**lines));

    while (getline(&(*lines)[nlines], &line_sz, textfile) >= 0) {
        nlines++;
        if (nlines >= bufsize) {
            size_t old_bufsize = bufsize;
            bufsize *= 2;
            *lines = realloc(*lines, bufsize * sizeof(**lines));
            memset(*lines + nlines, 0, old_bufsize * sizeof(**lines));
        }
    }

    *lines = realloc(*lines, nlines * sizeof(**lines));

    fclose(textfile);
    return nlines;
}
