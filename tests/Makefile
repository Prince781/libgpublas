TEST_SOURCES=$(filter-out test.c,$(wildcard *.c))
BINARIES=$(TEST_SOURCES:%.c=%)
CFLAGS=-std=gnu99 -Wall -Werror
LDFLAGS=-lm

ifeq ($(USE_MKL), y)
    CFLAGS += -I$(MKLROOT)/include 
    LDFLAGS += -L$(MKLROOT)/lib/intel64 -L$(TBBROOT)/lib/intel64/gcc4.7
    LDFLAGS += -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread
    LDFLAGS += -Wl,-rpath=$(MKLROOT)/lib/intel64,-rpath=$(TBBROOT)/lib/intel64/gcc4.7
else
    LDFLAGS += -lcblas
endif

.PHONY: all-tests clean

%.o: %.c test.h Makefile
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

copy: copy.o test.o

dsdot: dsdot.o test.o

all-tests: $(BINARIES)

clean:
	rm -f $(BINARIES)
	rm -f *.o
