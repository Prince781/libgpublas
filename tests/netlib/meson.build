project('netlib-tests', 'fortran',
    meson_version: '>= 0.49.0')

exe_prefixes = [
  'sblat3',
  'dblat3',
  'cblat3',
  'zblat3',
]

cc = meson.get_compiler('fortran')

blas_libs = {
    'mkl': cc.find_library('mkl_rt'),
    'tatlas': dependency('tatlas', required: false),
    'blas': cc.find_library('blas', required: false),
}

incdir = include_directories('.')

sh = find_program('./test.py')
b2c = files('../../build/libblas2cuda.so')

foreach prefix : exe_prefixes
  foreach suffix, dep : blas_libs
    if dep.found()
      exe = executable(prefix + '-' + suffix, [prefix + '.f'], include_directories: incdir, 
        dependencies: [dep], c_args: ['-O0', '-g', '-ggdb3'])
      if suffix == 'blas'
        test(prefix + '-' + 'correctness', sh, args: [prefix, exe, files('input.'+prefix), b2c, '--nointeractive'])
      endif
    endif
  endforeach
endforeach
