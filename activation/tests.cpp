#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <vector>
#include <cmath>

#include "activation_rvv.hh"
#include "activation_scalar.hh"

using namespace Catch::Detail;

void swish_vectorized(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_rvv(riscv_vfswish, x, n, out);
}

void swish_scalar(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_scalar(swishf, x, n, out);
}

void gelu_cook_vectorized(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_rvv(riscv_vfgelu_cook, x, n, out);
}

void gelu_cook_scalar(float const * __restrict__ x, size_t n, float * __restrict__ out) {
  elementwise_loop_scalar(geluf_tanh_cook, x, n, out);
}

TEST_CASE("Vectorized loop handles various buffer sizes", "[vectorized]") {
  for (size_t n : {1, 7, 16, 33, 64, 100, 257}) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
      x[i] = -3.0f + (6.0f * i / n); // Range from -3 to 3
    }

    std::vector<float> out_vectorized(n);
    std::vector<float> out_scalar(n);

    swish_vectorized(x.data(), n, out_vectorized.data());
    swish_scalar(x.data(), n, out_scalar.data());

    for (size_t i = 0; i < n; ++i) {
      INFO("n = " << n << ", x[" << i << "] = " << x[i]);
      REQUIRE(out_vectorized[i] == Approx(out_scalar[i]).epsilon(0.01));
    }
  }
}

TEST_CASE("Swish activation function", "[swish]") {
  std::vector<float> x;
  for (float e = -10.0f; e <= 10.0f; e += 0.25f) {
    x.push_back(e);
  }

  std::vector<float> out_vectorized(x.size());
  std::vector<float> out_scalar(x.size());

  swish_vectorized(x.data(), x.size(), out_vectorized.data());
  swish_scalar(x.data(), x.size(), out_scalar.data());

  for (size_t i = 0; i < x.size(); ++i) {
    INFO("x[" << i << "] = " << x[i]);
    REQUIRE(out_vectorized[i] == Approx(out_scalar[i]).epsilon(0.01));
  }
}

TEST_CASE("GELU Cook approximation", "[gelu][cook]") {
  std::vector<float> x;
  for (float e = -10.0f; e <= 10.0f; e += 0.25f) {
    x.push_back(e);
  }

  std::vector<float> out_vectorized(x.size());
  std::vector<float> out_scalar(x.size());

  gelu_cook_vectorized(x.data(), x.size(), out_vectorized.data());
  gelu_cook_scalar(x.data(), x.size(), out_scalar.data());

  for (size_t i = 0; i < x.size(); ++i) {
    INFO("x[" << i << "] = " << x[i]);
    REQUIRE(out_vectorized[i] == Approx(out_scalar[i]).margin(0.01));
  }
}
