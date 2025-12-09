project('low-level-rg', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
  ]
)

march = get_option('march')

# Build Google Benchmark as a CMake subproject
cmake = import('cmake')

cmake_opts = cmake.subproject_options()
cmake_opts.add_cmake_defines({
  'BENCHMARK_ENABLE_TESTING': 'OFF',
  'BENCHMARK_ENABLE_GTEST_TESTS': 'OFF',
  'BENCHMARK_ENABLE_INSTALL': 'OFF',
  'CMAKE_BUILD_TYPE': 'Release',
})

# Pass march flag to benchmark build
if march != ''
  cmake_opts.append_compile_args('cpp', ['-march=' + march, '-mno-fence-tso'])
  cmake_opts.append_link_args('cpp', ['-march=' + march])
endif

benchmark_proj = cmake.subproject('google-benchmark', options: cmake_opts)
benchmark_dep = benchmark_proj.dependency('benchmark')

# Add subdirectories
subdir('activation')
