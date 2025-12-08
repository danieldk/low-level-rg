project('activation_tests', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
  ]
)

# Find Catch2 dependency
catch2_dep = dependency('catch2', required: true)

# Compiler flags for RISC-V vector extension
riscv_vector_flags = [
  '-march=rv64gcv',
]

# Include directories
inc_dirs = include_directories('.')

# Build the test executable
test_exe = executable('activation_tests',
  'tests.cpp',
  include_directories: inc_dirs,
  dependencies: [catch2_dep],
  cpp_args: riscv_vector_flags,
)

# Register the test with Meson
test('activation tests', test_exe)
