// Copyright © 2025 Apple Inc.
// FP4/FP8 device-side dequantization helpers for QMV kernels.
// Extracted from quantized_utils.cuh (deleted upstream).
#pragma once

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

namespace mlx::core::cu {

// Round a positive float up to the nearest power of two (the E8M0 grid).
inline __device__ float round_up_pow2(float v) {
  uint32_t bits = __float_as_uint(v);
  uint32_t exp = (bits >> 23) & 0xffu;
  bool is_pow2 = (bits & 0x7fffffu) == 0;
  return __uint_as_float((is_pow2 ? exp : exp + 1) << 23);
}

// Group scale for block-scaled activation quantization.
// inv_max_rep = 1/max_representable of the element type (1/6 for E2M1,
// 1/448 for E4M3). E8M0-scaled formats (MXFP4/MXFP8) can only store powers
// of two, so the scale used to quantize a group must be the SAME power of
// two that is stored — quantizing against the continuous amax-based scale
// and storing its E8M0 rounding inflates every group by up to 2x.
// Continuous scales are fine for UE4M3-scaled NVFP4.
template <bool kUE8M0>
inline __device__ float block_group_scale(float amax, float inv_max_rep) {
  if (amax <= 0.0f) {
    return 1.0f;
  }
  float s = amax * inv_max_rep;
  if constexpr (kUE8M0) {
    return round_up_pow2(s);
  }
  return s;
}

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

} // namespace mlx::core::cu
