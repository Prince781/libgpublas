project('blas2cuda', 'c', 'cpp',
    default_options: ['c_std=gnu11', 'cpp_std=gnu++11'],
    version: '0.1',
    license: 'GPL3+',
    meson_version: '>= 0.42.0')

cc = meson.get_compiler('c')
root_inc = include_directories('.')

c_args = [
  '-Wall', 
  '-Wextra',
  '-Wformat=2',
  '-Werror', 
  '-Wno-unused-parameter',
  '-Winit-self',
  '-O3'
]

sources = files(
    'lib/fortran-compat.c',
    'blas2cuda.c',
    'entry.c',
)

blas_level1_sources = files(
  'blas_level1/amax.cc',
  'blas_level1/amin.cc',
  'blas_level1/asum.cc',
  'blas_level1/axpy.cc',
  'blas_level1/copy.cc',
  'blas_level1/dot.cc',
  'blas_level1/dotc.cc',
  'blas_level1/dotu.cc',
  'blas_level1/nrm2.cc',
  'blas_level1/rot.cc',
  'blas_level1/rotg.cc',
  'blas_level1/rotm.cc',
  'blas_level1/rotmg.cc',
  'blas_level1/scal.cc',
  'blas_level1/swap.cc',
)

blas_level2_sources = files(
  'blas_level2/bmv.cc',
  'blas_level2/gbmv.cc',
  'blas_level2/gemv.cc',
  'blas_level2/ger.cc',
  'blas_level2/hemv.cc',
  'blas_level2/her.cc',
  'blas_level2/her2.cc',
  'blas_level2/hpr.cc',
  'blas_level2/hpr2.cc',
  'blas_level2/pmv.cc',
  'blas_level2/spr.cc',
  'blas_level2/spr2.cc',
  'blas_level2/symv.cc',
  'blas_level2/syr.cc',
  'blas_level2/syr2.cc',
  'blas_level2/tbmv.cc',
  'blas_level2/tbsv.cc',
  'blas_level2/tpmv.cc',
  'blas_level2/tpsv.cc',
  'blas_level2/trmv.cc',
  'blas_level2/trsv.cc',
)

blas_level3_sources = files(
    'blas_level3/gemm.cc',
    'blas_level3/hemm.cc',
    'blas_level3/her2k.cc',
    'blas_level3/herk.cc',
    'blas_level3/level3.cc',
    'blas_level3/symm.cc',
    'blas_level3/syr2k.cc',
    'blas_level3/syrk.cc',
    'blas_level3/trmm.cc',
    'blas_level3/trsm.cc',
)

cudadir = get_option('CUDA')
cuda_lib_dirs = [cudadir + '/lib64', cudadir + '/lib64/stubs']

## dependencies
libpthread_dep = cc.find_library('pthread')
libdl_dep = cc.find_library('dl')
libcublas_dep = cc.find_library('cublas', dirs: cuda_lib_dirs)
libcuda_dep = cc.find_library('cuda', dirs: cuda_lib_dirs)
libcudart_dep = cc.find_library('cudart', dirs: cuda_lib_dirs)

cuda_inc = include_directories(cudadir + '/include')

subdir('lib')

library('blas2cuda', sources + blas_level1_sources +
    blas_level2_sources + blas_level3_sources, 
  c_args: c_args,
  cpp_args: c_args,
  link_args: '-Wl,-init,blas2cuda_init,-fini,blas2cuda_fini,-eentry',
  dependencies: [
    libobjtracker_dep,
    libcuda_dep,
    libcudart_dep,
    libcublas_dep,
  ],
  include_directories: [root_inc, cuda_inc, lib_inc],
  install: true,
)
