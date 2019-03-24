CFLAGS=-Wall -Werror -Wextra -Wno-unused-parameter -std=gnu11 -ggdb3
LDFLAGS=-lOpenCL

.PHONY: clean

all: svm svm-kernel svm-mem-kernel

svm: svm.c util.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

svm-kernel: svm-kernel.c util.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

svm-mem-kernel: svm-mem-kernel.c util.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	$(RM) svm svm-kernel svm-mem-kernel
