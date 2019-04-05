project('blas2cuda-tests', 'c',
    default_options: ['c_std=gnu11'],
    license: 'GPL3+',
    meson_version: '>= 0.45.0')

exe_prefixes = [
    'copy',
    'dsdot',
    'gbmv',
    'gemm',
    'hemm',
    'trmv',
    'trsm',
]

cc = meson.get_compiler('c')

libblas_dep = cc.find_library('mkl_rt')

incdir = include_directories('.')

foreach prefix : exe_prefixes
    executable(prefix, [prefix + '.c', 'test.c'], include_directories: incdir, dependencies: [libblas_dep])
endforeach
