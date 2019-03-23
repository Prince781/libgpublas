project('blas2cuda-tests', 'c',
    default_options: ['c_std=gnu11'],
    license: 'GPL3+',
    meson_version: '>= 0.49.0')

exe_prefixes = [
    'copy',
    'dsdot',
    'gbmv',
    'gemm',
    'hemm',
]

cc = meson.get_compiler('c')

blas_libs = {
    'mkl': cc.find_library('mkl_rt'),
    'tatlas': dependency('tatlas', required: false),
    'cblas': dependency('cblas', required: false),
}

incdir = include_directories('.')

foreach prefix : exe_prefixes
  foreach suffix, dep : blas_libs
    if dep.found()
      executable(prefix + '-' + suffix, [prefix + '.c', 'test.c'], include_directories: incdir, dependencies: [dep])
    endif
  endforeach
endforeach
