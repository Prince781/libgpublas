lib_inc = include_directories('.')

libobjtracker = static_library('objtracker',
    sources: [
      'fortran-compat.c',
      'obj_tracker.c',
    ],
    include_directories: [root_inc],
    dependencies: [
      libpthread_dep, 
      libdl_dep,
    ],
    c_args: ['-Wall', '-Werror', '-fdump-rtl-expand'],
    install: false,
)

libobjtracker_dep = declare_dependency(
    link_with: libobjtracker,
    include_directories: [root_inc, lib_inc],
)

shared_library('objtracker',
    sources: [
      'blas_tracker.c',
      'obj_tracker.c',
    ],
    include_directories: [root_inc],
    dependencies: [
      libpthread_dep, 
      libdl_dep,
    ],
    c_args: ['-Wall', '-Werror', '-DSTANDALONE', '-DTRACE_OUTPUT'],
    link_args: '-Wl,-init,obj_tracker_init,-fini,obj_tracker_fini,-eentry',
    install: true,
)
