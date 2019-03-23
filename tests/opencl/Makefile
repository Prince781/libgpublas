CFLAGS=-Wall -Werror -Wextra -Wno-unused-parameter -std=gnu11 -ggdb3
LDFLAGS=-lOpenCL

.PHONY: clean

all: svm svm-kernel svm-mem-kernel

svm: svm.c util.h

svm-kernel: svm-kernel.c util.c

svm-mem-kernel: svm-mem-kernel.c util.c

clean:
	$(RM) svm svm-kernel svm-mem-kernel
