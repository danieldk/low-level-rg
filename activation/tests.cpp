#include <cmath>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "activation_rvv.hh"
#include "activation_scalar.hh"

using namespace Catch::Detail;

void swish_vectorized(float const *__restrict__ x, size_t n,
                      float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfswish, x, n, out);
}

void swish_scalar(float const *__restrict__ x, size_t n,
                  float *__restrict__ out) {
  elementwise_loop_scalar(swishf, x, n, out);
}

void gelu_cook_vectorized(float const *__restrict__ x, size_t n,
                          float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfgelu_cook, x, n, out);
}

void gelu_cook_scalar(float const *__restrict__ x, size_t n,
                      float *__restrict__ out) {
  elementwise_loop_scalar(geluf_tanh_cook, x, n, out);
}

void gelu_logistic_vectorized(float const *__restrict__ x, size_t n,
                              float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfgelu_logistic, x, n, out);
}

void gelu_logistic_scalar(float const *__restrict__ x, size_t n,
                          float *__restrict__ out) {
  elementwise_loop_scalar(geluf_logistic, x, n, out);
}

void dish_vectorized(float const *__restrict__ x, size_t n,
                     float *__restrict__ out) {
  elementwise_loop_rvv(riscv_vfdish, x, n, out);
}

void dish_scalar(float const *__restrict__ x, size_t n,
                 float *__restrict__ out) {
  elementwise_loop_scalar(dish, x, n, out);
}

void leaky_relu_scalar(float const *__restrict__ x, size_t n,
                       float *__restrict__ out) {
  elementwise_loop_scalar(leaky_reluf, x, n, out);
}

void leaky_relu_max_vectorized(float const *__restrict__ x, size_t n,
                               float *__restrict__ out) {
  elementwise_loop_rvv(riscv_leaky_relu_max, x, n, out);
}

void leaky_relu_mask_vectorized(float const *__restrict__ x, size_t n,
                                float *__restrict__ out) {
  elementwise_loop_rvv(riscv_leaky_relu_masked, x, n, out);
}

void elish_vectorized(float const *__restrict__ x, size_t n,
                      float *__restrict__ out) {
  elementwise_loop_rvv(riscv_elish, x, n, out);
}

void elish_scalar(float const *__restrict__ x, size_t n,
                  float *__restrict__ out) {
  elementwise_loop_scalar(elishf, x, n, out);
}

// Test fixtures for activation functions
struct SwishTest {
  static constexpr auto vectorized = swish_vectorized;
  static constexpr auto scalar = swish_scalar;
  static constexpr const char *name = "Swish";
};

struct GeluCookTest {
  static constexpr auto vectorized = gelu_cook_vectorized;
  static constexpr auto scalar = gelu_cook_scalar;
  static constexpr const char *name = "GELU Cook";
};

struct GeluLogisticTest {
  static constexpr auto vectorized = gelu_logistic_vectorized;
  static constexpr auto scalar = gelu_logistic_scalar;
  static constexpr const char *name = "GELU Logistic";
};

struct DishTest {
  static constexpr auto vectorized = dish_vectorized;
  static constexpr auto scalar = dish_scalar;
  static constexpr const char *name = "Dish";
};

struct LeakyReluMaxTest {
  static constexpr auto vectorized = leaky_relu_max_vectorized;
  static constexpr auto scalar = leaky_relu_scalar;
  static constexpr const char *name = "Leaky ReLU Max";
};

struct LeakyReluMaskTest {
  static constexpr auto vectorized = leaky_relu_mask_vectorized;
  static constexpr auto scalar = leaky_relu_scalar;
  static constexpr const char *name = "Leaky ReLU Mask";
};

struct ElishTest {
  static constexpr auto vectorized = elish_vectorized;
  static constexpr auto scalar = elish_scalar;
  static constexpr const char *name = "ELiSH";
};

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

TEMPLATE_TEST_CASE("Activation function correctness", "[activation]", SwishTest,
                   GeluCookTest, GeluLogisticTest, DishTest, LeakyReluMaxTest,
                   LeakyReluMaskTest, ElishTest) {
  std::vector<float> x;
  for (float e = -10.0f; e <= 10.0f; e += 0.25f) {
    x.push_back(e);
  }

  std::vector<float> out_vectorized(x.size());
  std::vector<float> out_scalar(x.size());

  TestType::vectorized(x.data(), x.size(), out_vectorized.data());
  TestType::scalar(x.data(), x.size(), out_scalar.data());

  for (size_t i = 0; i < x.size(); ++i) {
    INFO(TestType::name << ": x[" << i << "] = " << x[i]);
    REQUIRE(out_vectorized[i] == Approx(out_scalar[i]).margin(0.01));
  }
}
