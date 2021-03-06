project('libgpublas', 'c', 'cpp', 'fortran',
    default_options: ['c_std=gnu11', 'cpp_std=gnu++17'],
    version: '0.1',
    license: 'GPL3+',
    meson_version: '>= 0.43.0')

cc = meson.get_compiler('c')
root_inc = include_directories('.')

optflags = get_option('optflags')

c_args = cc.get_supported_arguments([
  '-Wall', 
  '-Wextra',
  '-Wformat=2',
  '-Werror', 
  '-Wno-unused-parameter',
  '-Wnull-dereference',
  '-D_GLIBCXX_ASSERTIONS',
  optflags,
  '-g',
  '-ggdb3',
])

link_args = ['-Wl,-init,blas2cuda_init,-fini,blas2cuda_fini,-eentry']

sources = files(
    'blas2cuda.c',
    'entry.c',
    'runtime.c',
    'runtime-blas.c',
)

# TODO: fix BLAS level 1 and 2 before compiling

blas_level1_sources = files(
  #  'blas_level1/amax.cc',
  #  'blas_level1/amin.cc',
  #  'blas_level1/asum.cc',
  #  'blas_level1/axpy.cc',
  #  'blas_level1/copy.cc',
  #  'blas_level1/dot.cc',
  #  'blas_level1/dotc.cc',
  #  'blas_level1/dotu.cc',
  #  'blas_level1/nrm2.cc',
  #  'blas_level1/rot.cc',
  #  'blas_level1/rotg.cc',
  #  'blas_level1/rotm.cc',
  #  'blas_level1/rotmg.cc',
  #  'blas_level1/scal.cc',
  #  'blas_level1/swap.cc',
)

blas_level2_sources = files(
  #  'blas_level2/bmv.cc',
  #  'blas_level2/gbmv.cc',
  #  'blas_level2/gemv.cc',
  #  'blas_level2/ger.cc',
  #  'blas_level2/hemv.cc',
  #  'blas_level2/her.cc',
  #  'blas_level2/her2.cc',
  #  'blas_level2/hpr.cc',
  #  'blas_level2/hpr2.cc',
  #  'blas_level2/pmv.cc',
  #  'blas_level2/spr.cc',
  #  'blas_level2/spr2.cc',
  #  'blas_level2/symv.cc',
  #  'blas_level2/syr.cc',
  #  'blas_level2/syr2.cc',
  #  'blas_level2/tbmv.cc',
  #  'blas_level2/tbsv.cc',
  #  'blas_level2/tpmv.cc',
  #  'blas_level2/tpsv.cc',
  #  'blas_level2/trmv.cc',
  #  'blas_level2/trsv.cc',
)

blas_level3_sources = files(
    'blas_level3/gemm.cc',
    'blas_level3/hemm.cc',
    'blas_level3/her2k.cc',
    'blas_level3/herk.cc',
    'blas_level3/symm.cc',
    'blas_level3/syr2k.cc',
    'blas_level3/syrk.cc',
    # TODO: cannot enable until https://github.com/clMathLibraries/clBLAS/issues/341 is resolved
    # 'blas_level3/trmm.cc',
    'blas_level3/trsm.cc',
)

runtime = ''
prefix = get_option('prefix')

cudadir = get_option('CUDA')
cuda_lib_dirs = [cudadir + '/lib64', cudadir + '/lib64/stubs']

## dependencies
libpthread_dep = cc.find_library('pthread')
libdl_dep = cc.find_library('dl')
libblas_dep = dependency('blas', required: false)

if not libblas_dep.found()
    libblas_dep = cc.find_library('blas')
endif

gpu_libs = []
gpu_inc = []
gpu_srcs = []

# select the supported runtime
if get_option('runtime') == 'auto' or get_option('backend') == 'cuda'
  libcudart_dep = cc.find_library('cudart', dirs: cuda_lib_dirs, required: get_option('backend') == 'cuda')
  libcuda_dep = cc.find_library('cuda', dirs: cuda_lib_dirs, required: get_option('backend') == 'cuda')
  libcublas_dep = cc.find_library('cublas', dirs: cuda_lib_dirs, required: get_option('backend') == 'cuda')
endif

if get_option('runtime') != 'opencl' and (libcublas_dep.found() and libcuda_dep.found() and libcudart_dep.found())
  # use CUDA
  cuda_inc = include_directories(cudadir + '/include')
  gpu_libs = [libcublas_dep, libcuda_dep, libcudart_dep]
  gpu_inc = [cuda_inc]
  c_args += '-DUSE_CUDA'
  link_args += ',-rpath='+cudadir+'/lib64'
  runtime = 'CUDA'
else
  #use OpenCL
  libcl_dep = dependency('OpenCL')
  libclblas_dep = dependency('clBLAS')
  gpu_libs = [libclblas_dep, libcl_dep]
  clext_h = custom_target('clext.h',
                          output: 'clext.h',
                          command: [find_program('./clext.py'), libcl_dep.get_pkgconfig_variable('includedir') + '/CL/'])

  clblas_ext_h = custom_target('clblas_ext.h',
                                input: files(libclblas_dep.get_pkgconfig_variable('includedir') + '/clBLAS.h'),
                                output: 'clblas_ext.h',
                                command: [find_program('./clblas_ext.py'), '@INPUT@'])
  gpu_srcs += [clext_h, clblas_ext_h]
  c_args += '-DUSE_OPENCL'
  runtime = 'OpenCL'
endif

if not get_option('blas_opt')
  c_args += '-DUSE_GPU_ALWAYS'
endif

subdir('lib')

libgpublas = library('blas2cuda', gpu_srcs + sources + blas_level1_sources +
    blas_level2_sources + blas_level3_sources,
  c_args: c_args,
  cpp_args: c_args,
  link_args: link_args,
  dependencies: [
    libobjtracker_dep,
    libblas_dep,
  ] + gpu_libs,
  include_directories: [root_inc, lib_inc] + gpu_inc,
  install: true,
)

subdir('tests/netlib')

output = [
  '',
  '',
  '   libgpublas ' + meson.project_version(),
  '  ===============',
  '',
  '    Prefix....................... ' + prefix,
  '    Runtime...................... ' + runtime,
  '    C/C++ flags.................. ' + ' '.join(c_args),
  '',
  '  Now type \'ninja -C ' + meson.build_root() + '\' to build ' + meson.project_name(),
  '',
  '',
]
message('\n'.join(output))
