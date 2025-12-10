#include <riscv_vector.h>

inline vint32m8_t floor_f32_i32(vfloat32m8_t x, size_t vl) {
    return __riscv_vfcvt_x_f_v_i32m8_rm(x, __RISCV_FRM_RDN, vl);
}

// Remove implicit bits.
constexpr size_t FLOAT_MANTISSA_BITS = std::numeric_limits<float>::digits - 1;

// https://stackoverflow.com/a/47025627
inline vfloat32m8_t riscv_vfexp(vfloat32m8_t x, size_t vl) {
  // x * log2(e)
  auto t = __riscv_vfmul_vf_f32m8(x, 1.442695041f, vl);

  // Floor to get the integral part of x * log2(e).
  auto i = floor_f32_i32(t, vl);

  // The fractional part f is t - floor(t).
  auto e = __riscv_vfcvt_f_x_v_f32m8(i, vl);
  auto f = __riscv_vfsub_vv_f32m8(t, e, vl);

  // Compute 2^f with a polynomial approximation.
  auto p = __riscv_vfmul_vf_f32m8(f, 0.3371894346f, vl);
  p = __riscv_vfadd_vf_f32m8(p, 0.657636276f, vl);
  p = __riscv_vfmul_vv_f32m8(p, f, vl);
  p = __riscv_vfadd_vf_f32m8(p, 1.00172476f, vl);

  // Add the integral part to the to the exponent part of the float.
  auto j = __riscv_vsll_vx_i32m8(i, FLOAT_MANTISSA_BITS, vl);
  auto r_int = __riscv_vadd_vv_i32m8(j, __riscv_vreinterpret_v_f32m8_i32m8(p), vl);

  return __riscv_vreinterpret_v_i32m8_f32m8(r_int);
}

// Logistic CDF: 1 / (1+e^-x)
inline vfloat32m8_t riscv_vflogcdf(vfloat32m8_t x, size_t vl) {
  auto r = __riscv_vfneg_v_f32m8(x, vl);
  r = riscv_vfexp(r, vl);
  r = __riscv_vfadd_vf_f32m8(r, 1.0f, vl);
  return __riscv_vfrdiv_vf_f32m8(r, 1.0f, vl);
}

inline vfloat32m8_t riscv_vfswish(vfloat32m8_t x, size_t vl) {
  auto cdf = riscv_vflogcdf(x, vl);
  return __riscv_vfmul_vv_f32m8(x, cdf, vl);
}

inline vfloat32m8_t riscv_vftanh(vfloat32m8_t x, size_t vl) {
  auto a = riscv_vfexp(x, vl);
  auto b = riscv_vfexp(__riscv_vfneg_v_f32m8(x, vl), vl);
  return __riscv_vfdiv_vv_f32m8(__riscv_vfsub_vv_f32m8(a, b, vl), __riscv_vfadd_vv_f32m8(a, b, vl), vl);
}


inline vfloat32m8_t riscv_vfgelu_cook(vfloat32m8_t x, size_t vl) {
    // Approximation: 0.5x (1 + tanh(1 + tanh(0.8x)))
    // 0.8x
    auto r = __riscv_vfmul_vf_f32m8(x, 0.8f, vl);
    // tanh(0.8x)
    r = riscv_vftanh(r, vl);
    // 1.0 + tanh(0.8x)
    r = __riscv_vfadd_vf_f32m8(r, 1.0f, vl);
    // x(1.0 + tanh(0.8x))
    r = __riscv_vfmul_vv_f32m8(r, x, vl);
    return __riscv_vfmul_vf_f32m8(r, 0.5f, vl);
}

inline vfloat32m8_t riscv_vfgelu_logistic(vfloat32m8_t x, size_t vl) {
  // xσ(1.702x), from Hendrycks & Gimpel, 2016
  auto scaled_x = __riscv_vfmul_vf_f32m8(x, 1.702, vl);
  auto r = riscv_vflogcdf(scaled_x, vl);
  return __riscv_vfmul_vv_f32m8(x, r, vl);
}


// Newton-Raphson iteration for inverse square root: y' = y * (1.5 - 0.5 * a * y * y)
inline vfloat32m8_t rsqrt_newton_raphson(vfloat32m8_t y, vfloat32m8_t a, size_t vl) {
  auto tmp = __riscv_vfmul_vv_f32m8(y, y, vl);
  tmp = __riscv_vfmul_vv_f32m8(a, tmp, vl);
  tmp = __riscv_vfmul_vf_f32m8(tmp, 0.5f, vl);
  tmp = __riscv_vfrsub_vf_f32m8(tmp, 1.5f, vl);
  return __riscv_vfmul_vv_f32m8(y, tmp, vl);
}

// https://en.wikipedia.org/wiki/Fast_inverse_square_root
inline vfloat32m8_t fast_rsqrt(vfloat32m8_t x, size_t vl) {
  auto i = __riscv_vreinterpret_v_f32m8_u32m8(x);
  i = __riscv_vsrl_vx_u32m8(i, 1, vl);
  i = __riscv_vrsub_vx_u32m8(i, 0x5f3759df, vl);
  auto rsqrt = __riscv_vreinterpret_v_u32m8_f32m8(i);

  // Newton-Raphson iterations for better accuracy, one iteration
  // diverges too much from scalar Dish.
  rsqrt = rsqrt_newton_raphson(rsqrt, x, vl);
  rsqrt = rsqrt_newton_raphson(rsqrt, x, vl);

  return rsqrt;
}

// Dish: https://danieldk.eu/Dish-Activation
inline vfloat32m8_t riscv_vfdish(vfloat32m8_t x, size_t vl) {
  // First make the sigmoidal 0.5 (1 + x / sqrt(1 + x^2))
  auto sigmoidal = __riscv_vfmul_vv_f32m8(x, x, vl);
  sigmoidal = __riscv_vfadd_vf_f32m8(sigmoidal, 1.0f, vl);

#if __riscv_xtheadvector && false
  // T-Head like the Milk-V doesn't have inverse square root :(.
  sigmoidal = __riscv_vfsqrt_v_f32m8(sigmoidal, vl);
  sigmoidal = __riscv_vfdiv_vv_f32m8(x, sigmoidal, vl);
#elif __riscv_xtheadvector && true
  auto rsqrt = fast_rsqrt(sigmoidal, vl);
  sigmoidal = __riscv_vfmul_vv_f32m8(x, rsqrt, vl);
#else
  // Inverse square root with 7 bit precision.
  auto rsqrt = __riscv_vfrsqrt7_v_f32m8(sigmoidal, vl);

  // Do an additional Newton-Raphson iteration.
  rsqrt = rsqrt_newton_raphson(rsqrt, sigmoidal, vl);
  sigmoidal = __riscv_vfmul_vv_f32m8(x, rsqrt, vl);
#endif


  sigmoidal = __riscv_vfadd_vf_f32m8(sigmoidal, 1.0f, vl);
  sigmoidal = __riscv_vfmul_vf_f32m8(sigmoidal, 0.5, vl);

  // x * sigmoidal(x)
  return __riscv_vfmul_vv_f32m8(x, sigmoidal, vl);
}

inline vfloat32m8_t riscv_vfrelu(vfloat32m8_t x, size_t vl) {
  return __riscv_vfmax_vf_f32m8(x, 0.0f, vl);
}

template <typename F>
void elementwise_loop_rvv(F f, float const * __restrict__ x, size_t n, float * __restrict__ out) {
    for (size_t vl; n > 0; n -= vl, x += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(x, vl);
        auto r = f(v, vl);
        __riscv_vse32_v_f32m8(out, r, vl);
    }
}
