#ifndef ORACLE_H
#define ORACLE_H

#include "obj_tracker.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

bool oracle_should_alloc_managed_ptr(bool is_malloc, 
                                     uint64_t nth_alloc, 
                                     size_t obj_size);

bool oracle_load_file(const char *filename);

#endif
