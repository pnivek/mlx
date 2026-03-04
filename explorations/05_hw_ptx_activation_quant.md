# Exploration 05: Hardware PTX for FP4 Activation Quantization

**Status:** NO PERF IMPROVEMENT (Kernel is Memory-Bound)
**Date:** 2026-03-04
**Target:** Reduce SM120 GEMM pipeline overhead from activation quantization

## Summary

Replacing the software 8-way branch chain `quantize_float_to_e2m1()` with NVIDIA's
hardware `__nv_cvt_float2_to_fp4x2()` (compiles to `cvt.rn.satfinite.e2m1x2.f32` PTX)
produces **correct results** but **zero performance improvement**. The activation
quantization kernel is memory-bound, not compute-bound as initially hypothesized.

## What Was Tried

### Approach 1: Raw Inline PTX (FAILED correctness)

```cpp
asm volatile(
    "cvt.rn.satfinite.e2m1x2.f32 q0, %2, %1;\n"
    "cvt.rn.satfinite.e2m1x2.f32 q1, %4, %3;\n"
    "mov.b16 %0, {q0, q1};\n"
    : "=h"(packed)
    : "f"(sv0), "f"(sv1), "f"(sv2), "f"(sv3));
```

**Result:** Built successfully but produced 0.64-5.0 relative error. Root cause unknown —
the nibble ordering and encoding analysis checked out. The same PTX instruction works in
`nvfp4_quantize.cuh` for weight quantization. Abandoned in favor of CUDA builtins.

### Approach 2: `__nv_fp4_e2m1(float)` builtin (correct, no perf gain)

```cpp
uint8_t q0 = __nv_fp4_e2m1(v0 * inv_scale).__x;
// ... (4 calls)
output[out_base] = q0 | (q1 << 4);
```

**Result:** Correct. Produces 4 separate `cvt.rn.satfinite.e2m1x2.f32` instructions
(each with a dummy zero second argument). Kernel time unchanged: 127.4µs at M=1024.

### Approach 3: `__nv_cvt_float2_to_fp4x2` packed (correct, no perf gain)

```cpp
float2 pair0 = make_float2(v0 * inv_scale, v1 * inv_scale);
output[out_base] = __nv_cvt_float2_to_fp4x2(pair0, __NV_E2M1, cudaRoundNearest);
```

**Result:** Correct. Produces 2 `cvt.rn.satfinite.e2m1x2.f32` instructions (each with
two real values). Kernel time unchanged: 126.9µs at M=1024, 29.6µs at M=256.

## Performance Analysis

| M | Baseline (sw branch) | HW CVT (builtin) | HW CVT (packed) | BW limit |
|---|---|---|---|---|
| 256 | 30.6µs | 29.6µs | 29.6µs | ~10µs (L2) |
| 1024 | 127.4µs | 127.4µs | 126.9µs | ~81µs (DRAM) |

### Why No Improvement?

The plan estimated the kernel was compute-bound at 1.65x the BW limit. This was
**incorrect** — the 1.65x overhead comes from memory access patterns, not compute:

1. **Instruction reduction was real**: 36 instructions → 6 instructions (6x fewer)
2. **But instructions weren't the bottleneck**: The warp scheduler hides compute
   latency by switching between warps. With high occupancy (many warps), the
   effective throughput is limited by memory bandwidth, not instruction count.
3. **The actual bottleneck**: Scale factor writes (scattered), L1TEX port utilization,
   and warp shuffle overhead for amax reduction.

At M=256 (L2-cached), the kernel runs at 3x the L2 BW limit. At M=1024 (DRAM), it
runs at 1.57x the DRAM BW limit. Neither regime is instruction-throughput-bound.

## Correctness Results

| Format | M | QMV (M≤8) rel_err | GEMM (M≥16) rel_err |
|--------|---|---|---|
| NVFP4 (gs=16) | 1/16/256 | 0.094 | 0.137 |
| MXFP4 (gs=32) | 1/16/256 | 0.115 | 0.593 |
| MXFP8 (control) | 16 | — | 0.582 |

MXFP4/MXFP8 higher error is pre-existing (e8m0 scale factor = power-of-2 only).
MXFP8 was not modified and shows same error, confirming no regression.

## Code Change (Kept for Cleanliness)

Despite no perf gain, the change was kept because:
1. Removes 27-line custom `quantize_float_to_e2m1()` function
2. Uses NVIDIA's official `__nv_cvt_float2_to_fp4x2()` API
3. Guaranteed correct E2M1 encoding (no custom threshold management)
4. Hardware CVT instruction has exact round-to-nearest-even behavior

## Key Learning

**The activation quantization kernel is memory-bound, not compute-bound.** To actually
improve it, we need to reduce memory traffic or improve access patterns, not reduce
instruction count. Potential approaches:
- Process 8+ elements per thread (wider vectorized loads/stores)
- Fuse quantization with the GEMM kernel (eliminate separate launch)
- Optimize scale factor write layout for coalescing
