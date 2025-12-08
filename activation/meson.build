project('activation_tests', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
    'buildtype=release',
  ]
)

catch2_dep = dependency('catch2', required: true)

march = get_option('march')

riscv_vector_flags = [
  '-march=' + march,
]

if get_option('buildtype') == 'release'
  riscv_vector_flags += ['-ffast-math']
endif

inc_dirs = include_directories('.')

test_exe = executable('activation_tests',
  'tests.cpp',
  include_directories: inc_dirs,
  dependencies: [catch2_dep],
  cpp_args: riscv_vector_flags,
  link_args: ['-static'],
)

test('activation tests', test_exe)

bench_exe = executable('activation_bench',
  'bench.cpp',
  include_directories: inc_dirs,
  cpp_args: riscv_vector_flags,
  link_args: ['-static'],
)
