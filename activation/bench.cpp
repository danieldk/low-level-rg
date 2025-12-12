#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "activation_rvv.hh"
#include "activation_scalar.hh"

__attribute__((noinline)) void relu_vectorized(float const *__restrict__ x,
                                               size_t n,
                                               float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfrelu, x, n, out);
}

__attribute__((noinline)) void swish_vectorized(float const *__restrict__ x,
                                                size_t n,
                                                float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfswish, x, n, out);
}

__attribute__((noinline)) void swish_scalar(float const *__restrict__ x,
                                            size_t n, float *__restrict__ out) {
  elementwise_loop_scalar(swishf, x, n, out);
}

__attribute__((noinline)) void gelu_cook_vectorized(float const *__restrict__ x,
                                                    size_t n,
                                                    float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfgelu_cook, x, n, out);
}

__attribute__((noinline)) void gelu_cook_scalar(float const *__restrict__ x,
                                                size_t n,
                                                float *__restrict__ out) {
  elementwise_loop_scalar(geluf_tanh_cook, x, n, out);
}

__attribute__((noinline)) void
gelu_logistic_vectorized(float const *__restrict__ x, size_t n,
                         float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfgelu_logistic, x, n, out);
}

__attribute__((noinline)) void gelu_logistic_scalar(float const *__restrict__ x,
                                                    size_t n,
                                                    float *__restrict__ out) {
  elementwise_loop_scalar(geluf_logistic, x, n, out);
}

__attribute__((noinline)) void dish_vectorized(float const *__restrict__ x,
                                               size_t n,
                                               float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfdish, x, n, out);
}

__attribute__((noinline)) void dish_scalar(float const *__restrict__ x,
                                           size_t n, float *__restrict__ out) {
  elementwise_loop_scalar(dish, x, n, out);
}

__attribute__((noinline)) void
masked_sqrt_vectorized(float const *__restrict__ x, size_t n,
                       float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfmasked_sqrt, x, n, out);
}

static std::vector<float> generate_test_data(size_t size) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::normal_distribution dist{0.0, 2.0};
  auto gen = [&]() { return dist(mersenne_engine); };

  std::vector<float> data(size);
  std::generate(data.begin(), data.end(), gen);
  return data;
}

static std::vector<float> generate_positive_data(size_t size) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution dist{0.1f, 10.0f};
  auto gen = [&]() { return dist(mersenne_engine); };

  std::vector<float> data(size);
  std::generate(data.begin(), data.end(), gen);
  return data;
}

static std::vector<float> generate_mixed_data(size_t size,
                                              float positive_percent) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution pos_dist{0.1f, 10.0f};
  std::uniform_real_distribution neg_dist{-10.0f, -0.1f};

  std::vector<float> data(size);

  size_t num_positive = static_cast<size_t>(size * positive_percent);

  // Fill with positive values
  for (size_t i = 0; i < num_positive; ++i) {
    data[i] = pos_dist(mersenne_engine);
  }
  // Fill rest with negative values
  for (size_t i = num_positive; i < size; ++i) {
    data[i] = neg_dist(mersenne_engine);
  }

  // Shuffle to avoid patterns
  std::shuffle(data.begin(), data.end(), mersenne_engine);

  return data;
}

static void BM_swish_scalar(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    swish_scalar(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_swish_vectorized(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    swish_vectorized(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_gelu_cook_scalar(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    gelu_cook_scalar(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_gelu_cook_vectorized(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    gelu_cook_vectorized(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_gelu_logistic_scalar(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    gelu_logistic_scalar(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_gelu_logistic_vectorized(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    gelu_logistic_vectorized(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_dish_scalar(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    dish_scalar(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_dish_vectorized(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    dish_vectorized(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

static void BM_relu_vectorized(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    relu_vectorized(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

template <int Percent> static void BM_masked_sqrt(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_mixed_data(size, Percent / 100.0f);
  std::vector<float> out(size);
  for (auto _ : state) {
    masked_sqrt_vectorized(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

const size_t BENCH_SIZE = 1024;

// Register benchmarks with different sizes
BENCHMARK(BM_swish_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_swish_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_gelu_cook_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_gelu_cook_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_gelu_logistic_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_gelu_logistic_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_dish_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_dish_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_relu_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_masked_sqrt<0>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<10>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<20>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<30>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<40>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<50>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<60>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<70>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<80>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<90>)->Arg(BENCH_SIZE);
BENCHMARK(BM_masked_sqrt<100>)->Arg(BENCH_SIZE);

// Alternative: Register with multiple sizes for comparison
// BENCHMARK(BM_swish_scalar)->Range(1024, 1<<20);
// BENCHMARK(BM_swish_vectorized)->Range(1024, 1<<20);

BENCHMARK_MAIN();
