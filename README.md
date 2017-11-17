# blas2cuda

This is a library to intercept calls to CPU-bound BLAS kernels and run their
equivalent on the GPU in a CUDA environment.

## Compiling
```
$ make -j`nproc`
```

## How it works
TODO: expand this section

### Allocation tracking
- done once
- track ALL object allocations
    - remove object-specific information, giving us only calls to malloc()
- saved to file

### Object tracking
- in code (blas2cuda):
    - define custom object manager (`struct objmngr`)
    - any time an allocation file is loaded, we use this memory manager
- any time `malloc()` is called:
    1. object tracker compares call info (from requested size and using
       libunwind to get instruction pointer) with allocation list
    2. if call info matches, allocate the object using the custom memory
       manager defined for the call, and track the object
    3. if call info doesn't match, act normally

### Motivation for object tracking
- Each time a kernel is called, we would have to copy data to GPU, invoke the
  kernel, and copy it back to the CPU. For a series of calls to kernels that
  aren't computation-intensive (Level 1 and Level 2 BLAS calls are vector-vector
  and matrix-vector operations), throughput is significantly degraded as the
  time to transfer data dominates computation.
    - This is why [NVBLAS](https://docs.nvidia.com/cuda/nvblas/index.html), a
      similar project, only intercepts computation-intensive Level 3
      matrix-matrix operations, where the computation dominates data transfer.
    - However, there's still this issue of copying back and forth.
- blas2cuda uses object tracking to distinguish memory objects that are used in
  BLAS kernels from other memory objects we don't care about.
- When a call is made to `malloc()` that we should care about, we use
  `cudaMallocManaged()` instead and return a memory address that is shared
  between the CPU and GPU. This memory is a managed object, and a later call to
  `free()` will use `cudaFree()` instead.
- By intercepting the right calls, we can tell when these memory objects are
  later used in kernels, and avoid copying.
- Instead of explicit copying, a [page faulting mechanism is used to move data
  between the CPU and GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-data-migration).

### Creating an object allocation-tracking definition
`./objtrack.sh <program>`

### Running a program with `libblas2cuda`
`./blas2cuda.sh <objtrackfile> <program>`

## Caveats/TODO
- Currently only works with ASLR disabled, since it relies on IP, SP, etc not
  being randomized.
- Needs more testing.
- Needs implementation of all BLAS kernels.
- Needs to work with `calloc`, `realloc`, `reallocarray`
