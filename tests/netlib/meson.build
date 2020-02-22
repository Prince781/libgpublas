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

foreach exe_prefix : exe_prefixes
  exe = executable(exe_prefix + '-blas', [exe_prefix + '.f'], include_directories: incdir, 
    dependencies: [libblas_dep, libm_dep, libft_dep], c_args: ['-O0', '-g', '-ggdb3'], link_with: [libgpublas])
  test(exe_prefix + '-' + 'correctness', sh, 
    args: [exe_prefix, exe, files('input.'+exe_prefix), libgpublas, '--nointeractive'],
    timeout: 300)
endforeach
