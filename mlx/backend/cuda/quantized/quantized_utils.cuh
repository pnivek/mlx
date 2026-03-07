// Copyright © 2025 Apple Inc.

#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace mlx::core {

namespace cu {

inline __device__ float4 dequant_fp8(uint32_t bits) {
  auto out = *(__nv_fp8x4_e4m3*)(&bits);
  return out.operator float4();
}

inline __device__ float4 dequant_fp4(uint16_t bits) {
  auto out = *(__nv_fp4x4_e2m1*)(&bits);
  return out.operator float4();
}

// Hardware FP4 E2M1 → __half2 dequant using cvt.rn.f16x2.e2m1x2 (F2FP SASS).
// Converts 8 FP4 values (packed in uint32) directly to 4 __half2 registers
// without the float32 widening step. Saves 4 HADD2.F32 instructions per
// uint32 compared to operator float4() + float conversion.
inline __device__ void dequant_fp4_half2x4(
    uint32_t word,
    __half2& out01,
    __half2& out23,
    __half2& out45,
    __half2& out67) {
  uint32_t fp16[4];
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
      "}\n"
      : "=r"(fp16[0]), "=r"(fp16[1]), "=r"(fp16[2]), "=r"(fp16[3])
      : "r"(word));
  out01 = *reinterpret_cast<__half2*>(&fp16[0]);
  out23 = *reinterpret_cast<__half2*>(&fp16[1]);
  out45 = *reinterpret_cast<__half2*>(&fp16[2]);
  out67 = *reinterpret_cast<__half2*>(&fp16[3]);
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T>
__device__ __forceinline__ void absmax_x2(T& out, const T& x1, const T& x2) {
  if constexpr (
      (std::is_same<T, __nv_bfloat162>::value) ||
      (std::is_same<T, __half2>::value)) {
    T a = x1;
    T b = x2;
    out = __hmax2(__habs2(a), __habs2(b));
  } else if constexpr (std::is_same<T, float2>::value) {
    float2 a = x1;
    float2 b = x2;
    out.x = fmaxf(fabsf(a.x), fabsf(b.x));
    out.y = fmaxf(fabsf(a.y), fabsf(b.y));
  }
}

} // namespace cu

template <typename F>
void dispatch_groups(int group_size, F&& f) {
  switch (group_size) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

template <typename F>
void dispatch_bits(int bits, F&& f) {
  switch (bits) {
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 3:
      f(std::integral_constant<int, 3>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
    case 5:
      f(std::integral_constant<int, 5>{});
      break;
    case 6:
      f(std::integral_constant<int, 6>{});
      break;
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
  }
}

} // namespace mlx::core
