TEST_SOURCES=$(filter-out test.c,$(wildcard *.c))
BINARIES=$(TEST_SOURCES:%.c=%)
CFLAGS=-std=gnu11 -Wall -Werror
LDFLAGS=-lm

ifeq ($(USE_MKL), y)
    CFLAGS += -I$(MKLROOT)/include -DUSE_MKL
    LDFLAGS += -L$(MKLROOT)/lib/intel64 -L$(TBBROOT)/lib/intel64/gcc4.7
    LDFLAGS += -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread
    LDFLAGS += -Wl,-rpath=$(MKLROOT)/lib/intel64,-rpath=$(TBBROOT)/lib/intel64/gcc4.7
else
    LDFLAGS += -lcblas
endif

.PHONY: all-tests clean 

all-tests: $(BINARIES)

%.o: %.c test.h Makefile
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

copy: copy.o test.o

dsdot: dsdot.o test.o

rot: rot.o test.o

gbmv: gbmv.o test.o

clean:
	rm -f $(BINARIES)
	rm -f *.o
	rm -f *.out
