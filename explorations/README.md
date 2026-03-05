# QMV FP4 L2 Bandwidth Gap: Exploration Results

**Problem:** FP4 QMV achieves only 51% of L2 bandwidth on SM121 (vs FP8's 83%).
**Root cause:** FP4 has 2.5x the L1TEX traffic per iteration (80B vs 32B), creating
port congestion that limits throughput when data is L2-resident.

## Exploration Summary

| # | Approach | Verdict | Measured Gain | Risk |
|---|----------|---------|---------------|------|
| 01 | [Software Pipelining](01_software_pipelining.md) | **NOT VIABLE** | 0% | Occupancy loss |
| 02 | [n_per_thread=8](02_n_per_thread_8.md) | **IMPLEMENTED (conditional)** | 7-17% DRAM | Register pressure |
| 03 | [Tensor Core GEMV](03_tensor_core_gemv.md) | **NOT RECOMMENDED** | N/A | Extreme complexity |
| 04 | [FP16 Accumulation (PTX)](04_fp16_accum_ptx.md) | **IMPLEMENTED** | 16-19% L2, 9% DRAM | Precision (verified OK) |
| 05 | [HW PTX Activation Quant](05_hw_ptx_activation_quant.md) | **NO PERF GAIN** | 0% (kernel is memory-bound) | Low |
| 06 | [cudaGraphExecUpdate Bug](06_cudaGraphExecUpdate_bug_sm121.md) | **WORKAROUND** | N/A (driver bug) | N/A |

## Implementation Results (nsys kernel-level, A/B tested)

### Optimization 1: n_per_thread=8 (persistent kernel only)

n=8 increases ILP by loading 2x more data per thread per iteration. However, the
higher register pressure (67% vs 83% occupancy) hurts L2-cached shapes where TLP matters.
**Solution:** conditionally use n=8 only for the persistent kernel (DRAM-bound >24MB).

| fp_qmv_persistent | Baseline (n=4) | n=8 persistent | Change |
|--------------------|---------------|----------------|--------|
| NVFP4 median | 159.5µs | 154.3µs | **-3.3%** |
| MXFP4 median | 147.8µs | 129.9µs | **-12.1%** |
| MXFP8 median | 240.8µs | 224.2µs | **-6.9%** |

### Optimization 2: FP16 accumulation with hardware PTX dequant

Uses inline `cvt.rn.f16x2.e2m1x2` (F2FP SASS instruction) + `__hfma2` to reduce
instruction count from 10 to 4 SASS ops per 4 FP4 values. Previous attempt using
software `__half` LUT failed; this uses the **same hardware instruction** that
`__nv_fp4x4_e2m1::operator float4()` uses internally, but stays in FP16 instead of
widening to FP32.

| fp_qmv_single (L2) | Baseline | FP16 accum | Change |
|---------------------|----------|-----------|--------|
| NVFP4 median | 14.2µs | **11.5µs** | **-19%** |
| MXFP4 median | 13.3µs | **11.2µs** | **-16%** |
| MXFP8 (control) | 12.3µs | 11.8µs | ~neutral |

| fp_qmv_persistent (DRAM) | Baseline | FP16 accum + n=8 | Change |
|--------------------------|----------|-----------------|--------|
| NVFP4 median | 159.5µs | **144.6µs** | **-9%** |
| MXFP4 median | 147.8µs | **134.0µs** | **-9%** |
| MXFP8 (control) | 240.8µs | 234.4µs | ~neutral |

## Key Learnings

1. **The kernel is bandwidth-bound, not latency-bound.** Optimizations that increase
   ILP to cover latency (software pipelining) don't help when the memory pipe is already
   saturated by TLP (32+ warps).

2. **L1TEX port congestion is the bottleneck for L2-cached FP4.** FP4 reads 64B activation
   + 16B weight = 80B per iteration through L1TEX, vs FP8's 32B. The persistent kernel's
   shared-memory activation trick addresses this for DRAM shapes but not L2 shapes.

3. **CUTLASS's own GEMV doesn't use tensor cores.** Even NVIDIA uses scalar
   `cvt.rn.f16x2.e2m1x2` + `fma.rn.f16x2` for block-scaled GEMV, confirming that
   tensor core GEMV is not a practical approach.

4. **`__nv_fp4x4_e2m1::operator float4()` already uses hardware `cvt.rn.f16x2.e2m1x2`
   internally.** The overhead is 4 additional `HADD2.F32` widening instructions per
   4 values. By staying in FP16 with `__hfma2`, we eliminate those widenings entirely.

5. **n_per_thread=8 helps DRAM but hurts L2.** The reduced occupancy (67% vs 83%)
   directly reduces L2 throughput, but the extra ILP hides DRAM latency. Conditional
   dispatch based on matrix size is the right approach.

6. **FP16 precision is sufficient for FP4 scale groups.** Max accumulation per group:
   32 values × max_product ≈ 32 × 60 = 1920, well within FP16 range (65504).

## Files Modified

| File | Changes |
|------|---------|
| `quantized_utils.cuh` | Added `dequant_fp4_half2x4()` using inline PTX (~20 lines) |
| `qmv.cu` | FP16 accum in `fp_qmv_impl` + `fp_qmv_persistent`, conditional n=8 dispatch |
| `gather_qmv.cu` | FP16 accum in `gather_qmv_impl`, reverted to n=4 max |

## MXFP4 Dispatch Investigation (2026-03-04)

bench_qmm.py showed MXFP4 0.86x at M=64 (slower than dense). Investigation with proper
timing (10 trials × 100 iters, median) showed this was a **measurement artifact**:

| M | NVFP4 vs Dense | MXFP4 vs Dense |
|---|---|---|
| 1 | 1.96x | 2.28x |
| 8 | 2.37x | 2.26x |
| 64 | 1.60x | **1.74x** (was "0.86x") |
| 256 | 2.01x | 2.02x |
| 1024 | 2.41x | 2.39x |

**Conclusion:** MXFP4 dispatch is correct. No M-threshold needed. The erratic numbers
were from bench_qmm.py's poor methodology (mean of 20 iters, no trials). Fixed benchmark
now uses median of 10 trials × 50 iters.

## Remaining Opportunity

**Shared memory activation for non-persistent kernel** could further improve L2-cached
shapes by separating weight loads (L1TEX) from activation loads (smem). But with FP16
accum already giving 16-19% improvement, the marginal gain may not justify the complexity
and occupancy cost.
