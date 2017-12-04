# put /opt/cuda/bin/ into $PATH
NVCC=nvcc
comma:= ,
empty:=
space:=$(empty) $(empty)
CUDA ?= /opt/cuda
NVSOURCES=$(wildcard *.cu) $(wildcard blas_level1/*.cu) $(wildcard blas_level2/*.cu)
CSOURCES=$(wildcard *.c) $(filter-out $(wildcard tests/*.c),$(wildcard */*.c))
CXXSOURCES=$(wildcard *.cc) $(wildcard blas_level1/*.cc) $(wildcard blas_level2/*.cc)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
LIBDIR=lib
CFLAGS += -Wall -Werror -fPIC -ggdb3 -I$(CUDA)/include
NVCFLAGS=$(subst $(space),$(comma),$(CFLAGS))
LDFLAGS += -L$(CUDA)/lib64 -lcublas -L$(LIBDIR) -ldl -lunwind -lunwind-x86_64
NVLDFLAGS=$(subst $(space),$(comma),$(LDFLAGS))

libblas2cuda.so: $(NVSOURCES:%.cu=$(OBJDIR)/%.o) $(CSOURCES:%.c=$(OBJDIR)/%.o) $(CXXSOURCES:%.cc=$(OBJDIR)/%.o)
	$(NVCC) -shared -Xlinker $(NVLDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.c
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -c $^ -o $@

$(OBJDIR)/%.o: %.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(NVCC) -Xcompiler $(NVCFLAGS) -shared -c $^ -o $@

$(OBJDIR)/%.o: %.cc
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CXX) $(CFLAGS) -shared $(LDFLAGS) -c $^ -o $@

#$(LIBDIR)/libobjtracker.so:
#	$(MAKE) -C $(LIBDIR) libobjtracker.so

.PHONY: clean

clean:
	rm -rf $(OBJDIR)
	rm -f libblas2cuda.so
	@#@$(MAKE) -C $(LIBDIR) clean
