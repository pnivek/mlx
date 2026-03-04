# Exploration 02: Increasing n_per_thread from 4 to 8

**Status:** CONDITIONALLY VIABLE -- Worth Prototyping
**Date:** 2026-03-04
**Target:** Close FP4 L2 BW gap (51% vs FP8's 83%)

## Summary

Increasing `n_per_thread` from 4 to 8 doubles the work per loop iteration, providing
8 independent dequant+FMA chains for ILP (vs 4). The kernel body requires **zero code
changes** -- only dispatch logic needs updating. Scale group math is correct for all
group sizes. The main risk is register pressure dropping occupancy from 83% to 67%.

## Concept

Each thread in the warp currently processes 4 packed `uint32_t` values per loop iteration
(FP4: 32 activation elements). With n=8, each thread processes 8 packed values (64 activation
elements), halving the number of loop iterations and doubling the compute between load stalls.

## Current vs Proposed Parameters (FP4)

| Parameter | n=4 (current) | n=8 (proposed) |
|-----------|---------------|----------------|
| `nv_per_thread` | 32 | 64 |
| Weight load size | 16B (1x LDG.128) | 32B (2x LDG.128) |
| Activation load size | 64B (4x LDG.128) | 128B (8x LDG.128) |
| Total load/iter | 80B (5 LDG.128) | 160B (10 LDG.128) |
| Dequant calls/iter | 8 | 16 |
| FMA ops/iter | 32 | 64 |
| Loop iterations | K/(32*32)=K/1024 | K/(32*64)=K/2048 |
| Independent compute chains | 4 | **8** |

## Register Pressure Analysis

### Estimated Budget

| Variable | n=4 (regs) | n=8 (regs) | Delta |
|----------|------------|------------|-------|
| `local_mat` (uint32_t[N]) | 4 | 8 | +4 |
| `local_vec` (__half[N*8]) | 16 | 32 | +16 |
| Other (sum, scales, ptrs, temps) | ~26 | ~26 | 0 |
| **Total** | **~46** | **~66** | **+20** |

### Occupancy Impact (SM121: 65536 regs, 48 warps max, block=256 threads)

| Regs/thread | Blocks/SM | Warps/SM | Occupancy |
|-------------|-----------|----------|-----------|
| 46 (n=4 est) | 5 | 40 | 83% |
| 60 (n=4 actual) | 4 | 32 | 67% |
| 66 (n=8 est) | 3-4 | 24-32 | 50-67% |
| 72 (n=8 + compiler) | 3 | 24 | 50% |

**Key risk:** If compiler uses >64 regs/thread, we drop to 3 blocks (50% occupancy).
Use `__launch_bounds__(256, 4)` to guide the compiler toward 4 blocks/SM.

## Scale Group Correctness

All group sizes produce valid constants with n=8:

| group_size | scales_per_step | n_per_step | Values per scale group |
|------------|-----------------|------------|----------------------|
| 16 (NVFP4) | 4 | 2 | 2 * 8 = 16 |
| 32 (MXFP4) | 2 | 4 | 4 * 8 = 32 |
| 64 | 1 | 8 | 8 * 8 = 64 |
| 128 | 1 | 8 | 8 * 8 = 64 (partial group, threads share scales) |

All correct. The template computes these at compile time via `constexpr`.

## K Alignment Requirements

**Constraint:** K must be divisible by 64 (for FP4 with n=8).

Derivation: `packed_cols = K / 8` must be a multiple of `n_per_thread = 8`, so K % 64 == 0.

| K dimension | K % 64 | Compatible? |
|-------------|--------|-------------|
| 4096 | 0 | Yes |
| 5120 | 0 | Yes |
| 7168 | 0 | Yes |
| 8192 | 0 | Yes |
| 11008 | 0 | Yes |
| 14336 | 0 | Yes |

**All standard LLM dimensions work.**

## Pointer Alignment

- **Weight load** `unsafe_load_vector<8, uint32_t>`: needs 32-byte alignment
  - Check: `cu::is_aligned<8>(mat_ptr)` (8 * sizeof(uint32_t) = 32B)
- **Activation load** `unsafe_load_vector<64, __half>`: needs 128-byte alignment
  - `cudaMalloc` guarantees 256-byte alignment -- satisfied by default
  - Check: `cu::is_aligned<64>(vec_ptr)` (64 * sizeof(__half) = 128B)
  - **Fallback:** If alignment fails, dispatch to n=4 (graceful degradation)

## Code Changes Required

### 1. Dispatch function (qmv.cu + gather_qmv.cu)

Replace `dispatch_1_2_4()` with `dispatch_1_2_4_8()`:
```cpp
template <typename F>
void dispatch_1_2_4_8(int n, F&& f) {
  switch (n) {
    case 1: f(std::integral_constant<int, 1>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
  }
}
```

### 2. Alignment check (qmv.cu ~line 370)

```cpp
int n = 1;
if (K % 64 == 0 && cu::is_aligned<8>(mat_ptr) &&
    ((bits == 4 && cu::is_aligned<64>(vec_ptr)) ||
     cu::is_aligned<8>(vec_ptr))) {
  n = 8;
} else if (K % 32 == 0 && cu::is_aligned<4>(mat_ptr) &&
    ((bits == 4 && cu::is_aligned<8>(vec_ptr)) ||
     cu::is_aligned<4>(vec_ptr))) {
  n = 4;
} else if (cu::is_aligned<2>(mat_ptr) && ...) {
  n = 2;
}
```

### 3. Kernel body: NO CHANGES

The kernel `fp_qmv_impl` is fully parametric on `n_per_thread`. With n=8, the template
generates correct code for load sizes, scale group iteration, and FMA counts automatically.
`unsafe_load_vector<8, uint32_t>` decomposes into 2x LDG.128 by the compiler.

### 4. Template instantiations

Adding n=8 adds ~12-18 new kernel variants across:
- `fp_qmv_kernel` (qmv.cu): 2 bits (4,8) x 3 group_sizes x 1 type = 6 variants per n value
- `fp_qmv_persistent` (qmv.cu): same
- `fp_gather_qmv_kernel` (gather_qmv.cu): same

Estimated compile time increase: ~30-60 seconds.

## Expected Performance

| Regime | Current | Expected with n=8 | Improvement |
|--------|---------|-------------------|-------------|
| FP4 L2-cached (4096x4096) | 51% L2 BW | 55-65% L2 BW | 5-15% |
| FP4 DRAM (8192x8192) | 81% DRAM BW | 80-83% DRAM BW | 0-3% |
| FP8 (control) | 79-83% | unchanged | -- |

Primary benefit is from **8 independent dequant+FMA chains** (vs 4), allowing better
instruction-level parallelism. Halving loop iterations also reduces overhead.

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Register spills (>64 regs) | HIGH | `__launch_bounds__(256, 4)` + check with `--ptxas-options=-v` |
| Occupancy drop to 50% | MEDIUM | Benchmark: if n=8 regresses, keep n=4 as default |
| Compile time increase | LOW | Acceptable (~30-60s extra) |
| Alignment failure on sliced tensors | LOW | Graceful fallback to n=4 |

## Recommended Approach

1. Add n=8 as an **additional** dispatch option (not replacing n=4)
2. Use `__launch_bounds__(256, 4)` to prevent compiler from using too many registers
3. Build with `--ptxas-options=-v` to verify actual register count
4. Benchmark L2-cached (4096x4096) and DRAM-bound (8192x8192) shapes
5. If n=8 helps L2 but hurts DRAM, add a size-based dispatch threshold
6. Also apply to persistent kernel and gather_qmv

## Alternative Consideration

The n_per_thread=8 agent noted that **double-buffering with n=4** might achieve the
same ILP benefit with similar register cost but keeps the loop body simpler. However,
n=8 is cleaner because the kernel is already parametric -- it's literally just changing
dispatch, not kernel code.
