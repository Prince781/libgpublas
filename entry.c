#include <stdio.h>
#include <stdlib.h>

const char my_interp[] __attribute__((section(".interp"))) = "/lib64/ld-linux-x86-64.so.2";

int entry(void) {
    extern void b2c_print_help(void);

    b2c_print_help();
    exit(0);
}
