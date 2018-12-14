gpublas
-------

This is a library to intercept calls to CPU BLAS kernels and run their
equivalent on the GPU in a CUDA/OpenCL environment.

## Compiling
```
$ export CUDA=...
$ meson -DCUDA=$CUDA build && ninja -C build
```
