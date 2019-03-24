TODO
----

- [ ] profile GPU transfers (nvprof) to see whether we're actually avoiding copies
- [x] track statistics of sizes of objects
  - [x] modify dataflow and objtrace scripts
- [ ] collect examples
  - [x] nwchem samples from repo
  - [x] mpb samples from repo
  - [x] meep samples from repo (v1.6)
- [ ] collect performance info 
  - [ ] nwchem
  - [ ] Octave
  - [ ] cp2k
  - [ ] mpb
  - [ ] meep
- [ ] implement OpenCL backend
  - [x] abstract away CUDA/OpenCL runtimes
  - [ ] abstract away cublas/clblast BLAS runtimes
