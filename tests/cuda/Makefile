NVCC=nvcc
CFLAGS=-g3,-ggdb3,-Wall,-Werror,-DDEBUG
%: %.cu Makefile
	$(NVCC) -Xcompiler $(CFLAGS) $< -o $@ -lcuda -lcudart
