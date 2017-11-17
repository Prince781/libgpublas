# put /opt/cuda/bin/ into $PATH
NVCC=nvcc
comma:= ,
empty:=
space:=$(empty) $(empty)
CUDA ?= /opt/cuda
SOURCES=$(wildcard *.cu) $(wildcard blas_level1/*.cu)
CSOURCES=$(wildcard *.c) $(wildcard */*.c)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
LIBDIR=lib
CFLAGS += -Wall -Werror -fPIC -ggdb3 -I$(CUDA)/include
NVCFLAGS=$(subst $(space),$(comma),$(CFLAGS))
LDFLAGS += -L$(CUDA)/lib64 -lcublas -L$(LIBDIR) -ldl -lunwind -lunwind-x86_64
NVLDFLAGS=$(subst $(space),$(comma),$(LDFLAGS))

libblas2cuda.so: $(SOURCES:%.cu=$(OBJDIR)/%.o) $(CSOURCES:%.c=$(OBJDIR)/%.o) #$(LIBDIR)/libobjtracker.so
	$(NVCC) -shared -Xlinker $(NVLDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.c
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -c $^ -o $@

$(OBJDIR)/%.o: %.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(NVCC) -Xcompiler $(NVCFLAGS) -shared -c $^ -o $@

#$(LIBDIR)/libobjtracker.so:
#	$(MAKE) -C $(LIBDIR) libobjtracker.so

.PHONY: clean

clean:
	rm -rf $(OBJDIR)
	rm -f $(OBJECTS)
	rm -f libblas2cuda.so
	@#@$(MAKE) -C $(LIBDIR) clean
