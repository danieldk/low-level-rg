#include <ctime>
#include <random>
#include <vector>
#include <algorithm>

#include "../ubench.h"
#include "activation_rvv.hh"
#include "activation_scalar.hh"

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

// Swish benchmarks
UBENCH_EX(swish, scalar) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };
    std::vector<float> x(65536);
    std::generate(x.begin(), x.end(), gen);

    std::vector<float> out(x.size());

    UBENCH_DO_BENCHMARK() {
        swish_scalar(x.data(), x.size(), out.data());
    }
}

UBENCH_EX(swish, vectorized) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };
    std::vector<float> x(65536);
    std::generate(x.begin(), x.end(), gen);

    std::vector<float> out(x.size());

    UBENCH_DO_BENCHMARK() {
        swish_vectorized(x.data(), x.size(), out.data());
    }
}

// GELU Cook approximation benchmarks
UBENCH_EX(gelu_cook, scalar) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };
    std::vector<float> x(65536);
    std::generate(x.begin(), x.end(), gen);

    std::vector<float> out(x.size());

    UBENCH_DO_BENCHMARK() {
        gelu_cook_scalar(x.data(), x.size(), out.data());
    }
}

UBENCH_EX(gelu_cook, vectorized) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };
    std::vector<float> x(65536);
    std::generate(x.begin(), x.end(), gen);

    std::vector<float> out(x.size());

    UBENCH_DO_BENCHMARK() {
        gelu_cook_vectorized(x.data(), x.size(), out.data());
    }
}

// Dish activation benchmarks
UBENCH_EX(dish, scalar) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };
    std::vector<float> x(65536);
    std::generate(x.begin(), x.end(), gen);

    std::vector<float> out(x.size());

    UBENCH_DO_BENCHMARK() {
        dish_scalar(x.data(), x.size(), out.data());
    }
}

UBENCH_EX(dish, vectorized) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::normal_distribution dist{0.0, 2.0};
    auto gen = [&](){ return dist(mersenne_engine); };
    std::vector<float> x(65536);
    std::generate(x.begin(), x.end(), gen);

    std::vector<float> out(x.size());

    UBENCH_DO_BENCHMARK() {
        dish_vectorized(x.data(), x.size(), out.data());
    }
}

//UBENCH_STATE();
UBENCH_MAIN();
