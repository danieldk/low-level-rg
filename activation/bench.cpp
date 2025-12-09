#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <algorithm>

#include "activation_rvv.hh"
#include "activation_scalar.hh"

__attribute__((noinline))
void relu_vectorized(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_rvv(riscv_vfrelu, x, n, out);
}


__attribute__((noinline))
void swish_vectorized(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_rvv(riscv_vfswish, x, n, out);
}

__attribute__((noinline))
void swish_scalar(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_scalar(swishf, x, n, out);
}

__attribute__((noinline))
void gelu_cook_vectorized(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_rvv(riscv_vfgelu_cook, x, n, out);
}

__attribute__((noinline))
void gelu_cook_scalar(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_scalar(geluf_tanh_cook, x, n, out);
}

__attribute__((noinline))
void dish_vectorized(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_rvv(riscv_vfdish, x, n, out);
}

__attribute__((noinline))
void dish_scalar(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_scalar(dish, x, n, out);
}

static std::vector<float> generate_test_data(size_t size) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };

    std::vector<float> data(size);
    std::generate(data.begin(), data.end(), gen);
    return data;
}

static void BM_swish_scalar(benchmark::State& state) {
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

static void BM_swish_vectorized(benchmark::State& state) {
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

static void BM_gelu_cook_scalar(benchmark::State& state) {
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

static void BM_gelu_cook_vectorized(benchmark::State& state) {
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

static void BM_dish_scalar(benchmark::State& state) {
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

static void BM_dish_vectorized(benchmark::State& state) {
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

static void BM_relu_vectorized(benchmark::State& state) {
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

const size_t BENCH_SIZE = 1024;

// Register benchmarks with different sizes
BENCHMARK(BM_swish_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_swish_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_gelu_cook_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_gelu_cook_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_dish_scalar)->Arg(BENCH_SIZE);
BENCHMARK(BM_dish_vectorized)->Arg(BENCH_SIZE);

BENCHMARK(BM_relu_vectorized)->Arg(BENCH_SIZE);

// Alternative: Register with multiple sizes for comparison
// BENCHMARK(BM_swish_scalar)->Range(1024, 1<<20);
// BENCHMARK(BM_swish_vectorized)->Range(1024, 1<<20);

BENCHMARK_MAIN();
