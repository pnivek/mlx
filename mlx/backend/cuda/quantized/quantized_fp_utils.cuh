// Copyright © 2025 Apple Inc.
// FP4/FP8 device-side dequantization helpers for QMV kernels.
// Extracted from quantized_utils.cuh (deleted upstream).
#pragma once

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

namespace mlx::core::cu {

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
