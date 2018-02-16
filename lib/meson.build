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
