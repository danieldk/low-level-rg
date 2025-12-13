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

__attribute__((noinline)) void leaky_relu_scalar(float const *__restrict__ x,
                                                 size_t n,
                                                 float *__restrict__ out) {
  elementwise_loop_scalar(leaky_reluf, x, n, out);
}

__attribute__((noinline)) void
leaky_relu_max_vectorized(float const *__restrict__ x, size_t n,
                          float *__restrict__ out) {
  elementwise_loop_rvv(riscv_leaky_relu_max, x, n, out);
}

__attribute__((noinline)) void
leaky_relu_mask_vectorized(float const *__restrict__ x, size_t n,
                           float *__restrict__ out) {
  elementwise_loop_rvv(riscv_leaky_relu_masked, x, n, out);
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

template <auto Func> static void BM_activation(benchmark::State &state) {
  const size_t size = state.range(0);
  auto x = generate_test_data(size);
  std::vector<float> out(size);

  for (auto _ : state) {
    Func(x.data(), x.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(size));
}

const size_t BENCH_SIZE = 1024;

// Register benchmarks with different sizes
BENCHMARK(BM_activation<swish_scalar>)->Arg(BENCH_SIZE);
BENCHMARK(BM_activation<swish_vectorized>)->Arg(BENCH_SIZE);

BENCHMARK(BM_activation<gelu_cook_scalar>)->Arg(BENCH_SIZE);
BENCHMARK(BM_activation<gelu_cook_vectorized>)->Arg(BENCH_SIZE);

BENCHMARK(BM_activation<gelu_logistic_scalar>)->Arg(BENCH_SIZE);
BENCHMARK(BM_activation<gelu_logistic_vectorized>)->Arg(BENCH_SIZE);

BENCHMARK(BM_activation<dish_scalar>)->Arg(BENCH_SIZE);
BENCHMARK(BM_activation<dish_vectorized>)->Arg(BENCH_SIZE);

BENCHMARK(BM_activation<relu_vectorized>)->Arg(BENCH_SIZE);

BENCHMARK(BM_activation<leaky_relu_scalar>)->Arg(BENCH_SIZE);
BENCHMARK(BM_activation<leaky_relu_max_vectorized>)->Arg(BENCH_SIZE);
BENCHMARK(BM_activation<leaky_relu_mask_vectorized>)->Arg(BENCH_SIZE);

// Alternative: Register with multiple sizes for comparison
// BENCHMARK(BM_activation<swish_scalar>)->Range(1024, 1<<20);
// BENCHMARK(BM_activation<swish_vectorized>)->Range(1024, 1<<20);

BENCHMARK_MAIN();
