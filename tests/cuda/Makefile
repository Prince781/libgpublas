NVCC=nvcc
CFLAGS=-g3,-ggdb3,-Wall,-Werror
%: %.cu 
	$(NVCC) -Xcompiler $(CFLAGS) $^ -o $@ -lcuda -lcudart
