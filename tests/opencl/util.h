#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

size_t load_file(const char *filename, char ***lines);

int read_int(const char *prompt, int min, int max);

#endif
