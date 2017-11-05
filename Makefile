# put /opt/cuda/bin/ into $PATH
NVCC=nvcc
CUDA=/opt/lib64/cuda
SOURCES=$(wildcard *.cu) $(wildcard blas_level1/*.cu)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
LIBDIR=lib
CFLAGS=-Wall,-Werror,-fPIC,-ggdb3
LDFLAGS=-init,blas2cuda_init,-fini,blas2cuda_fini,-L,$(CUDA),-l,cublas,-L$(LIBDIR)

libblas2cuda.so: $(SOURCES:%.cu=$(OBJDIR)/%.o) $(LIBDIR)/libobjtracker.so
	$(NVCC) -shared -Xlinker $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(NVCC) -Xcompiler $(CFLAGS) -shared -c $^ -o $@

$(LIBDIR)/libobjtracker.so:
	$(MAKE) -C $(LIBDIR) libobjtracker.so

.PHONY: clean

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(OBJECTS)
	@rm -f libblas2cuda.so
	@$(MAKE) -C $(LIBDIR) clean
