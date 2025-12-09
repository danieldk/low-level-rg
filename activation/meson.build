catch2_dep = dependency('catch2', required: true)

# Get march option from parent project
march = get_option('march')

riscv_vector_flags = [
  '-march=' + march,
  '-mno-fence-tso',
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

# benchmark_dep is provided by parent meson.build
bench_exe = executable('activation_bench',
  'bench.cpp',
  include_directories: inc_dirs,
  dependencies: [benchmark_dep],
  cpp_args: riscv_vector_flags,
  link_args: ['-static'],
)
