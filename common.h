#ifndef COMMON_H
#define COMMON_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define write_str(fd, str)              \
{                                       \
    int fild = fd;                      \
    const char *_str = (str);           \
    size_t sz = strlen(_str);           \
    syscall(SYS_write, fild, _str, sz); \
}

#define writef(fd, ...)                             \
{                                                   \
    char buf[2048];                                 \
    snprintf(buf, sizeof buf,  __VA_ARGS__);        \
    write_str(fd, buf);                             \
}

#endif
