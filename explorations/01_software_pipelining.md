# Exploration 01: Software Pipelining (Double-Buffering) for QMV

**Status:** NOT VIABLE
**Date:** 2026-03-04
**Target:** Close FP4 L2 BW gap (51% vs FP8's 83%)

## Summary

Manual register double-buffering in the QMV inner loop is **not viable**. The kernel
is memory-bandwidth-bound, not memory-latency-bound. The compiler (ptxas) already performs
instruction scheduling that achieves the same effect. Manual double-buffering would hurt
performance through reduced occupancy.

## Concept

Classic software pipelining: maintain two sets of `local_mat`/`local_vec` registers.
While computing on buffer A (current iteration), prefetch buffer B (next iteration).
Swap after each iteration. This overlaps load latency with compute.

## Analysis

### Current Inner Loop (FP4, n_per_thread=4, group_size=32)

Per iteration:
- **Loads:** 64B activation (4x LDG.128) + 16B weights (1x LDG.128) + ~1B scale = **~6 LDG instructions**
- **Compute:** ~82 instructions (8 dequant + 32 h2f conversions + 32 FFMA + scale multiply + loop control)

### Register Impact

| Configuration | Regs/Thread | Blocks/SM | Warps/SM | Occupancy |
|---------------|-------------|-----------|----------|-----------|
| Current (single buffer) | ~45-60 | 4-5 | 32-40 | 67-83% |
| Double-buffer (+20 regs) | ~65-80 | 3-4 | 24-32 | 50-67% |
| With spills (80+ regs) | ~90+ | 2 | 16 | 33% |

### Why It Doesn't Help

**The kernel is bandwidth-bound, not latency-bound.** Critical distinction:

1. **L2-cached shapes (4096x4096, the 51% BW case):**
   - L2 latency: ~30-50 cycles
   - 82 compute instructions can already cover 50 cycles at 1 IPC
   - The bottleneck is **L1TEX port congestion** (FP4 has 2.5x the L1TEX traffic of FP8: 80B vs 32B per iteration)
   - Software pipelining does NOT reduce L1TEX port pressure -- it adds MORE outstanding loads

2. **DRAM-bound shapes (8192x8192):**
   - DRAM latency: ~400-600 cycles
   - 82 compute instructions can only cover ~82 cycles (at 1 IPC)
   - Need ~500 cycles of work to hide one load -- **82 is 6x short**
   - TLP (32+ warps) already covers this: 32 warps x 82 cycles = 2624 cycles of work available
   - Adding ILP is redundant when TLP already hides latency

3. **Compiler already does it:**
   - The inner loop is fully unrolled (`#pragma unroll`)
   - ptxas sees loads at the top and ~82 compute instructions below
   - It almost certainly interleaves next-iteration loads with current-iteration compute
   - Evidence: 89-98% L1TEX scoreboard stalls even with compiler scheduling = memory pipe is saturated

### The Root Cause of FP4 L2 Gap

The FP4 L2 bandwidth gap is caused by **L1TEX port congestion**, not insufficient ILP:

- FP4: 64B activation + 16B weight = **80B per iteration through L1TEX**
- FP8: 16B activation + 16B weight = **32B per iteration through L1TEX**
- FP4 has **2.5x the L1TEX traffic**

The persistent kernel already addresses this by moving activations to shared memory (separate port).
For L2-resident shapes, the non-persistent kernel doesn't use shared memory, hence the gap.

## Implementation Sketch (For Reference)

```cuda
// NOT RECOMMENDED -- shown for documentation only
if (row < rows) {
    AlignedVector<T, nv_per_thread> vec_A, vec_B;
    AlignedVector<uint32_t, n_per_thread> mat_A, mat_B;

    // Prologue: load first iteration
    bool has_work = (col < packed_cols);
    if (has_work) {
        vec_A = unsafe_load_vector<nv_per_thread>(vec + ...);
        mat_A = unsafe_load_vector<n_per_thread>(mat + ...);
    }

    while (has_work) {
        // Prefetch next iteration into buffer B
        bool has_next = (next_col < packed_cols);
        if (has_next) {
            vec_B = unsafe_load_vector<nv_per_thread>(vec + ...);
            mat_B = unsafe_load_vector<n_per_thread>(mat + ...);
        }
        // Compute on buffer A ...
        // Swap: vec_A = vec_B; mat_A = mat_B;  (20 MOV instructions!)
    }
}
```

**Problems:** Buffer swap costs 20 MOV instructions/iter, conditional prefetch hurts optimization,
register spill risk is high, compiler may undo manual scheduling anyway.

## Verdict

| Factor | Assessment |
|--------|------------|
| Register cost | +20 regs, pushing to 65-80 total |
| Occupancy impact | 67% -> 50% (or worse with spills) |
| ILP gain potential | ~14% theoretical max (DRAM), 0% (L2, port-bound) |
| TLP loss | Offsets any ILP gains |
| Compiler already does it | Likely yes |
| L1TEX congestion | Software pipelining makes it WORSE |

**Do not implement.** Focus on approaches that reduce L1TEX port pressure or reduce
total instructions between loads.
