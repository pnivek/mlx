# Exploration 04: FP16 Accumulation with Hardware PTX Dequant

**Verdict: IMPLEMENTED — 16-19% L2 improvement, 9% DRAM improvement**

## Problem

The FP4 QMV inner loop spends 10 SASS instructions per 4 FP4 values:
- 2× `F2FP.F16.E2M1.UNPACK_B` (hardware e2m1→f16x2 convert)
- 4× `HADD2.F32` (f16→f32 widening)
- 4× `FFMA` (FP32 fused multiply-add)

This is 2.5× more instructions than FP8 (4 per 4 values), explaining the L2 BW gap.

## Solution

Use `cvt.rn.f16x2.e2m1x2` inline PTX to get `__half2` weights directly, then
accumulate with `__hfma2` (packed FP16 FMA). Per 4 FP4 values:
- 2× `F2FP.F16.E2M1.UNPACK_B` (same as before)
- 2× `HFMA2` (packed FP16 FMA, 2 values per instruction)

**4 SASS instructions vs 10 — a 2.5× reduction.** Scale multiply stays in FP32.

## Key Insight from PTX Investigation

`__nv_fp4x4_e2m1::operator float4()` (our existing `dequant_fp4()`) already uses
`cvt.rn.f16x2.e2m1x2` internally — verified by SASS disassembly. There is NO
`cvt.rn.f32x2.e2m1x2` instruction; FP4 can only hardware-convert to f16x2.

The overhead in the FP32 path is entirely from the 4 extra widening instructions
(`cvt.f32.f16` in PTX → `HADD2.F32 Rd, -RZ, Rsrc.Hx_Hx` in SASS).

## Previous Failed Attempt

The first FP16 accumulation attempt (2026-03, now reverted) used a software `__half`
LUT for dequantization. This was slower than the hardware `F2FP` instruction because:
1. LUT needs shared memory or register file storage (16 entries × 2B)
2. Each lookup needs bit extraction + indexing, multiple instructions
3. The hardware `F2FP.F16.E2M1.UNPACK_B` does all of this in 1 instruction

The fix: use inline PTX `cvt.rn.f16x2.e2m1x2` (same instruction the hardware path
uses), but stay in FP16 instead of widening to FP32.

## Implementation

```cpp
// quantized_utils.cuh: Hardware FP4 → __half2 dequant
inline __device__ void dequant_fp4_half2x4(
    uint32_t word, __half2& out01, __half2& out23,
    __half2& out45, __half2& out67) {
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
  // ... (4 total __half2 outputs)
}
```

Inner loop (compile-time branch for FP4 + __half):
```cpp
if constexpr (bits < 8 && std::is_same_v<T, __half>) {
  __half2 h2_a = __float2half2_rn(0.0f);
  __half2 h2_b = __float2half2_rn(0.0f);
  for (int j = 0; j < n_per_step; ++j) {
    dequant_fp4_half2x4(local_mat[k], w01, w23, w45, w67);
    h2_a = __hfma2(w01, a01, h2_a);  // chain a
    h2_b = __hfma2(w23, a23, h2_b);  // chain b (ILP)
    h2_a = __hfma2(w45, a45, h2_a);
    h2_b = __hfma2(w67, a67, h2_b);
  }
  // Widen to FP32 only for scale multiply (once per group)
  sum += (half2float(h2_a + h2_b)) * float(scale);
}
```

## Precision Analysis

Max values accumulated in FP16 per scale group:
- gs=16 (NVFP4): 16 FP4 values, max product = 6.0 × activation ≈ 60
- gs=32 (MXFP4): 32 FP4 values, max sum ≈ 32 × 60 = 1920
- FP16 max: 65,504

Safe margin: ~34× for gs=32, ~1092× for gs=16.

## Measured Results (nsys kernel-level)

See README.md for full comparison tables.
