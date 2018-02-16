lib_inc = include_directories('.')

libobjtracker = static_library('objtracker',
    sources: [
      'callinfo.c',
      'fortran-compat.c',
      'obj_tracker.c',
    ],
    include_directories: [root_inc],
    dependencies: [
      libpthread_dep, 
      libunwind_dep, 
      libunwind_x86_64_dep,
      libdl_dep,
    ],
    install: false,
)

libobjtracker_dep = declare_dependency(
    link_with: libobjtracker,
    include_directories: [root_inc, lib_inc],
)

shared_library('objtracker',
    sources: [
      'blas_tracker.c',
      'callinfo.c',
      'fortran-compat.c',
      'obj_tracker.c',
    ],
    include_directories: [root_inc],
    dependencies: [
      libpthread_dep, 
      libunwind_dep, 
      libunwind_x86_64_dep,
      libdl_dep,
    ],
    c_args: ['-Wall', '-Werror', '-DSTANDALONE', '-DTRACE_OUTPUT'],
    link_args: '-Wl,-init,obj_tracker_init,-fini,obj_tracker_fini,-eentry',
    install: true,
)
