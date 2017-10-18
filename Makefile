# put /opt/cuda/bin/ into $PATH
NVCC=nvcc
CUDA=/opt/lib64/cuda
SOURCES=$(wildcard *.cu) $(wildcard */*.cu)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
CFLAGS=-Wall,-Werror,-fPIC
LDFLAGS=-init,blas2cuda_init,-fini,blas2cuda_fini,-L,$(CUDA),-l,cublas

libblas2cuda.so: $(SOURCES:%.cu=$(OBJDIR)/%.o)
	$(NVCC) -shared -Xlinker $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(NVCC) -Xcompiler $(CFLAGS) -shared -c $^ -o $@

.PHONY: clean

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(OBJECTS)
	@rm -f libblas2cuda.so
