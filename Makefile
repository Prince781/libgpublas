# put /opt/cuda/bin/ into $PATH
NVCC=nvcc
SOURCES=$(wildcard *.cu) $(wildcard */*.cu)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
CFLAGS=-Wall,-Werror,-fPIC
LDFLAGS=-fPIC,-init,blas2cuda_init,-fini,blas2cuda_fini

libmkl2cuda.so: $(SOURCES:%.cu=$(OBJDIR)/%.o)
	$(NVCC) -shared -Xlinker $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(NVCC) -Xcompiler $(CFLAGS) -c $^ -o $@

.PHONY: clean

clean: $(OBJECTS)
	@rm -rf $(OBJDIR)
	@rm -f $(OBJECTS)
	@rm -f libmkl2cuda.so
