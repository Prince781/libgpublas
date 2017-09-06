# put /opt/cuda/bin/ into $PATH
NVCC=nvcc
SOURCES=$(wildcard *.c)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
CFLAGS=-Wall,-Werror
LDFLAGS=-fPIC

$(OBJDIR):
	mkdir $@

$(OBJDIR)/%.o: %.c
	$(NVCC) -Xcompiler $(CFLAGS) -c $^ -o $@

.PHONY: clean

libmkl2cuda.so: $(SOURCES:%.c=$(OBJDIR)/%.o)
	$(NVCC) -shared -Xlinker $(LDFLAGS) $^ -o $@

clean: $(OBJECTS)
	rmdir $(OBJDIR)
	rm $(OBJECTS)
