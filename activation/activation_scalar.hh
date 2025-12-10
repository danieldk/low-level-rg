#include <cmath>
#include <limits>

constexpr float INV_SQRT2 = 1.0 / M_SQRT2;

inline float normal_cdf(float x) { return 0.5 * (1 + erff(x * INV_SQRT2)); }

inline float geluf(float x) { return x * normal_cdf(x); }

inline float logistic_cdf(float x) { return 1.0f / (1 + std::expf(-x)); }

inline float dish(float x) {
  return 0.5f * x * (1.0 + x / std::sqrt(1 + x * x));
}

inline float swishf(float x) { return x * logistic_cdf(x); }

constexpr float SQRT_2_INV_PI = 0.7978845608028654;

inline float geluf_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_INV_PI * (x + 0.044715f * x * x * x)));
}

inline float geluf_tanh_cook(float x) {
  return 0.5f * x * (1.0f + tanhf(0.8f * x));
}

inline float geluf_logistic(float x) { return x * logistic_cdf(1.702 * x); }

template <typename F>
void elementwise_loop_scalar(F f, float const *__restrict__ x, size_t n,
                             float *__restrict__ out) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = f(x[i]);
  }
}
