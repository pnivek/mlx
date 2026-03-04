# Exploration 03: Tensor Core GEMV (Block-Scaled MMA for M=1)

**Status:** NOT RECOMMENDED (Effort/Reward Unfavorable)
**Date:** 2026-03-04
**Target:** Close FP4 L2 BW gap (51% vs FP8's 83%)

## Summary

Using SM120 tensor core `mma.sync.aligned.block_scale` instructions for M=1 GEMV
is theoretically sound but practically unfavorable. The MMA minimum M=16 wastes 93.75%
of compute, but this is irrelevant since GEMV is 450x memory-bound. The real barriers
are extreme implementation complexity (fragment layout management) and the existence
of simpler alternatives that achieve most of the benefit.

## PTX Instruction Specifications

### MMA Shapes (SM120)

| Format | PTX Instruction | Shape | K/tile |
|--------|----------------|-------|--------|
| FP4 (E2M1) | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::{2X\|4X}.m16n8k64` | 16x8x64 | 64 |
| FP8 (E4M3) | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32` | 16x8x32 | 32 |

**Minimum M = 16. No M=1 or M=8 variant exists.**

### Register Requirements Per Thread (32 threads/warp)

| Operand | Registers | Description |
|---------|-----------|-------------|
| D (output) | 4 float | 16x8 output tile distributed across warp |
| A (activation) | 4 uint32_t | 16x64 A-matrix fragment |
| B (weight) | 2 uint32_t | 8x64 B-matrix fragment |
| C (accumulator) | 4 float | Same layout as D |
| SFA (scale A) | 1 uint8/16/32 | Scale factors for A |
| SFB (scale B) | 1 uint8/16/32 | Scale factors for B |
| **Total** | **16 regs** | Per-thread |

### Fragment Layouts (from CUTLASS `mma_traits_sm120.hpp`)

The fragment layouts are highly non-trivial:

**A Layout (16x64, row-major):** `(T32, V32) -> (M16, K64)`
```
Stride<Stride<_128,_1>,Stride<_16,_8,_512>>
```
Threads 0-7 cover 4 M-rows, threads 8-15 the next 4, etc. Each thread holds 32 FP4
values distributed non-contiguously across the 16x64 tile.

**B Layout (8x64, col-major):** `(T32, V16) -> (N8, K64)`
```
Stride<Stride<_64,_1>,Stride<_8,_256>>
```

**C Layout (16x8):** Standard SM80_16x8_Row.

## The M=1 Padding Problem

### Compute Waste (Irrelevant)

For M=1 padded to M=16: **93.75% compute waste**. But GEMV is 450x memory-bound:
- At K=N=8192: arithmetic intensity = 3.9 FLOPs/byte
- SM121 memory BW: 273 GB/s -> memory-limited to 1.1 TFLOPS
- SM121 FP4 tensor capacity: ~500 TFLOPS
- Headroom: 500 / 1.1 = **~450x**

Even with 16x compute waste, effective capacity is 500/16 = 31 TFLOPS, still 28x above
the memory-limited throughput. **Compute waste is completely irrelevant.**

### Memory Traffic (Key Question)

Does M=16 padding increase memory traffic?

| Data | M=1 | M=16 (padded) | Overhead |
|------|-----|---------------|----------|
| Weights (B matrix) | N*K/2 bytes | Same | 0% |
| Activation (A) | K*2 bytes | Load once, broadcast to 16 rows via smem | ~0% |
| SFA scales | K/gs bytes | Zeros for 15 padded rows | ~0% |
| SFB scales | N*K/gs bytes | Same | 0% |
| Output writes | N*2 bytes | Discard 15/16 rows | ~0% |

**Memory traffic is unchanged.** The weight reads (dominant cost) are identical.

## Four Approaches Evaluated

### Approach A: Direct Inline PTX MMA GEMV Kernel

Write a custom kernel using `mma.sync.aligned.block_scale` directly.

**Performance estimate:** 85-90% DRAM BW, 70-80% L2 BW

**Why it could help L2:** The MMA instruction replaces the software dequant+FMA pipeline.
Current FP4: 8 LUT lookups + 32 FP32 FMAs + 32 h2f conversions = ~82 instructions per 16B weights.
MMA: 1 instruction replaces all of this, eliminating instruction gaps between loads.

**Why it's not worth it:**
- Fragment layout management requires handling the non-trivial stride patterns above
- Weight format: current `(N, K/8)` uint32 row-major must be rearranged to MMA B-fragment format
- Scale factors must be reformatted to MMA-compatible SFA/SFB layouts
- Must support all group sizes (16, 32, 64, 128) for both FP4 and FP8
- Estimated development: 2-4 weeks of expert CUDA work
- Estimated improvement: ~30% L2 BW (from 51% to ~70-80%)

**Risk: HIGH.** Extremely error-prone to get right.

### Approach B: CUTLASS GEMM with M=128 Padding

Use existing `Sm120BlockScaledGemm` infrastructure (already working in `qmm_sm120.cu`).

**Problem:** CUTLASS minimum tile is **128x128** (not 16x16). TMA scale factor layout
requires BM >= 128 and BN >= 128. For M=1, this means padding to M=128.

**Overhead for M=1:**
- Activation quantization: 128 * K / 2 bytes (vs K/2) -- 128x overhead but small absolute cost
- CUTLASS kernel launch: ~5-10 us
- Total overhead: ~20-30 us
- For DRAM-bound shapes (117 us target): 20-30 us = 17-26% degradation
- For L2-cached shapes (12-24 us target): overhead **completely dominates**

**Risk: LOW but performance is WORSE than current QMV.**

### Approach C: Lower QMV-to-GEMM Dispatch Threshold

Change the `M <= 8` threshold in `quantized.cpp` to route M=1 through the SM120 GEMM path.

**Problem:** Same as Approach B -- CUTLASS tile overhead dominates at M=1.

**Risk: MEDIUM.** CUTLASS may not handle M=1 correctly (TMA layout requirements).

### Approach D: `cvt.rn.f16x2.e2m1x2` + `fma.rn.f16x2` in Existing Kernel

Replace the current software dequant path with PTX-level hardware conversion:

```
Current:  __nv_fp4x4_e2m1::operator float4()  -> ~6 instructions for 4 FP4 values
          4x FFMA (FP32)                       -> 4 instructions
          Total: ~10 instructions per 4 values

Proposed: cvt.rn.f16x2.e2m1x2 h_pair, byte   -> 1 instruction for 2 FP4->FP16
          fma.rn.f16x2 acc, h_pair, vec, acc   -> 1 instruction for 2 FP16 FMAs
          Total: ~4 instructions per 4 values (2x improvement)
```

**This is NOT the same as the previously-failed FP16 accumulation attempt.** The key
differences:
1. Uses hardware `cvt.rn.f16x2.e2m1x2` (single PTX instruction) instead of software LUT
2. Uses `fma.rn.f16x2` (packed CUDA core FMA) instead of manual `__halves2half2` packing
3. Stays within the existing kernel structure (no fragment layout complexity)

**Performance estimate:** FP4 L2 BW from 51% to 65-75%.

**Risk: LOW.** Pure software optimization to existing scalar path.

## CUTLASS's Own GEMV Approach

CUTLASS 4.3.5 includes `cutlass::gemm::kernel::GemvBlockScaled` which does **NOT** use
tensor core MMA. It uses:
1. `cvt.rn.f16x2.e2m1x2` for FP4 -> FP16 conversion
2. `fma.rn.f16x2` for FP16 multiply-accumulate
3. `cp.async` for pipelined global -> shared memory copies

This confirms that even NVIDIA's own library uses scalar FMA (not tensor cores) for GEMV.

## Recommendation

### Do NOT pursue tensor core GEMV (Approaches A-C)

The effort/reward is unfavorable:
- DRAM-bound: Current 81% BW, tensor core max ~90%. 10% gain for weeks of work.
- L2-cached: Theoretical 2x improvement, but absolute savings are ~12 us (L2 shapes are already fast)
- Framework overhead (380-680 us) dwarfs kernel time at M=1

### Consider Approach D as a follow-up

The `cvt.rn.f16x2.e2m1x2` + `fma.rn.f16x2` approach is interesting because:
1. CUTLASS uses exactly this approach in their own GEMV
2. It halves instruction count for dequant+multiply
3. It stays within the existing kernel (no layout complexity)
4. It failed previously as "FP16 accumulation" but the implementation was different (software LUT vs hardware CVT)

**Key insight from the failure:** The previous attempt used a `__half` LUT (4 bit-extract + 4
constant cache loads + 2 pack operations = slower than hardware `__nv_fp4x4_e2m1::operator float4()`).
The `cvt.rn.f16x2.e2m1x2` PTX instruction is fundamentally different -- it's a single hardware
instruction, like the existing `__nv_fp4x4_e2m1::operator float4()` but producing FP16 instead of FP32.

**However:** This needs careful investigation. The `__nv_fp4x4_e2m1::operator float4()` already
compiles to a hardware instruction on SM121. If `cvt.rn.f16x2.e2m1x2` is essentially the same
hardware unit, the benefit would come only from the FP16 FMA throughput (2x of FP32 FMA on SM121)
and reduced register width. Worth a careful PTX-level analysis before implementing.

## References

- CUTLASS `cute/arch/mma_sm120.hpp` -- PTX assembly for block-scaled MMA
- CUTLASS `cute/atom/mma_traits_sm120.hpp` -- Fragment layouts
- CUTLASS `cutlass/gemm/kernel/gemv_blockscaled.h` -- Scalar GEMV (no tensor cores)
- [NVIDIA PTX ISA 9.1 Section 9.7.14.3](https://docs.nvidia.com/cuda/parallel-thread-execution/) -- Block Scaling for mma.sync
- [NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/) -- GEMV is always memory-limited
- [CUDA HGEMV Optimization](https://bruce-lee-ly.medium.com/nvidia-cuda-core-cuda-hgemv-optimization-51c25927ad43) -- Tensor core utilization for GEMV is only 1/16
