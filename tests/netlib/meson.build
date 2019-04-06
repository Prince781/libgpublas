exe_prefixes = [
  'sblat3',
  'dblat3',
  'cblat3',
  'zblat3',
]

incdir = include_directories('.')

sh = find_program('./test.py')

libm_dep = cc.find_library('m')
libft_dep = cc.find_library('gfortran')

foreach prefix : exe_prefixes
  exe = executable(prefix + '-blas', [prefix + '.f'], include_directories: incdir, 
    dependencies: [libblas_dep, libm_dep, libft_dep], c_args: ['-O0', '-g', '-ggdb3'])
  test(prefix + '-' + 'correctness', sh, 
    args: [prefix, exe, files('input.'+prefix), libgpublas, '--nointeractive'],
    timeout: 300)
endforeach
