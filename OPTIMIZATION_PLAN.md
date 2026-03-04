# SM121 Kernel Optimization Plan — DGX Spark (GB10)

## Hardware Summary

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (SM121, Blackwell GeForce family) |
| SMs | 48 |
| L2 Cache | **24 MB** (confirmed via nsys `TARGET_INFO_GPU.l2CacheSize`) |
| DRAM | LPDDR5x, 273 GB/s sustained (546 GB/s theoretical) |
| Shared Memory | 100 KB per SM (101376 bytes from CUTLASS `arch.h`) |
| Registers | 65536 per SM |
| Max Warps/SM | 48 |
| Max Blocks/SM | 24 |
| Tensor Cores | `mma.sync.aligned.block_scale` (FP4/FP8), NOT `tcgen05` |
| TMA Multicast | No (cluster must be 1×1×1) |
| CUDA Toolkit | 13.0 |
| CUTLASS | v4.3.5 |

## Tools

- **nsys** (Nsight Systems 2025.3.2) — timeline + kernel durations
  - Must use `--cuda-graph-trace=node` for MLX (CUDA graphs)
  - Query `CUPTI_ACTIVITY_KIND_KERNEL` joined with `StringIds` via `shortName`
- **ncu** (Nsight Compute 2025.3.1) — per-kernel metrics
  - Requires `NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/`
  - Use `MLX_USE_CUDA_GRAPHS=0` env var to disable CUDA graphs for ncu
- **bench_qmm.py** — end-to-end benchmark (includes ~380-500µs framework overhead per M=1 call)

---

## Completed Work

### Phase 1: SM120 Block-Scaled GEMM (CUTLASS)

**Status: COMPLETE**

1. **NVFP4/MXFP4 Native GEMM** — `qmm_sm120.cu`
   - Activation quantization kernel (FP16 → FP4 + ue4m3/ue8m0 scales)
   - Scale factor reformat kernel (row-major → CUTLASS block-scaled layout)
   - CUTLASS `Sm120BlockScaledGemm` template with TileShape 128×128×128
   - N-padding for non-128-aligned weight dimensions
   - **Result**: 2-5× speedup over dequant+cuBLAS for NVFP4/MXFP4

2. **MXFP8 Native GEMM** — `qmm_sm120.cu`
   - FP8 activation quantization kernel (FP16 → e4m3 + ue8m0 scales)
   - TileShape 128×128×64 (K=64 for FP8, half of FP4's K=128)
   - N-padding support (same pattern as FP4)
   - **Result**: Native SM120 path instead of dequant+cuBLAS fallback

3. **Pingpong Schedule** — `qmm_sm120.cu`
   - Added `KernelTmaWarpSpecializedPingpong*Sm120` variants
   - AtomLayoutMNK = (2,2,1) = 4 warps, 128 threads (vs Cooperative's 256)
   - Both FP4 and FP8 Pingpong variants
   - **Result**: 2-4% improvement over Cooperative schedule

4. **Warp-Cooperative Activation Quantization** — `qmm_sm120.cu`
   - Warp-level parallel reduction for amax (32 threads per group)
   - Replaced sequential per-thread loop
   - **Result**: Reduced quantization overhead

### Phase 2: QMV/GatherQMV Dispatch Fixes (Correctness)

**Status: COMPLETE**

1. **bits=8 dispatch bug** — `qmv.cu`, `gather_qmv.cu`
   - Fixed: restructured if/else chain to check bits first, then group_size within each
   - Applied to all 3 dispatch sites (qmv single, qmv batched, gather_qmv)

2. **gs=128 support** — `qmv.cu`, `gather_qmv.cu`, `qmm.cu`
   - Added gs=128 branch to QMV/GatherQMV dispatch
   - Added gs=128 branch to `dispatch_groups()` in `qmm.cu`
   - Math verified: BK=128, GROUPS_PER_TILE=1, shared memory under 48KB

3. **Raised CuTe M cap** — `quantized.cpp`
   - Removed `group_size_ <= 64` guard
   - Raised M threshold to M≤512 for CuTe kernel dispatch

### Phase 3: Persistent QMV Kernel (Performance)

**Status: COMPLETE**

1. **`fp_qmv_persistent` kernel** — `qmv.cu`
   - 48 blocks (1 per SM), `extern __shared__` for activation vector
   - Cooperative shared memory load of activation vector
   - Contiguous row assignment per SM for sequential DRAM access
   - Grid: `{1, num_sms}`, dynamic shared memory = K × sizeof(T)
   - SM count cached via `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`

2. **L2 cache threshold**
   - Only use persistent kernel when `M == 1 && mat.nbytes() > 24MB && K*sizeof(T) <= 98304`
   - Shapes < 24MB benefit from original kernel (higher occupancy, L2-cached)
   - 24MB threshold matches actual L2 cache size

---

## Profiling Results

### QMV Kernel-Level Bandwidth (nsys, median kernel times)

**DRAM-limited shapes (>24MB, persistent kernel, 8192×8192):**

| Mode | Weight Bytes | Total Bytes | Kernel Time | Bandwidth | % of Peak |
|------|-------------|-------------|-------------|-----------|-----------|
| NVFP4 (gs=16) | 33.6 MB | 37.8 MB | 170 µs | 222 GB/s | **81.3%** |
| MXFP4 (gs=32) | 33.6 MB | 35.7 MB | ~170 µs | ~210 GB/s | **~77%** |
| MXFP8 (gs=32) | 67.1 MB | 69.2 MB | 321 µs | 215 GB/s | **78.8%** |

**L2-cached shapes (<24MB, original kernel, 4096×4096):**

| Mode | Total Bytes | Kernel Time | Effective BW | Notes |
|------|-------------|-------------|-------------|-------|
| NVFP4 (gs=16) | 9.4 MB | 9.2 µs | 1,022 GB/s | L2 BW, not DRAM |
| MXFP4 (gs=32) | 8.9 MB | 8.9 µs | 1,000 GB/s | L2 BW |
| MXFP8 (gs=32) | 17.3 MB | 10.4 µs | 1,663 GB/s | L2 BW |

### Key Insight: FP4 Dequant ALU Cost

The FP4 dequant (`__nv_fp4x4_e2m1` → float4) is more compute-intensive than FP8
(`__nv_fp8x4_e4m3` → float4): FP4 processes 8 values per uint32_t via two `dequant_fp4()`
calls with bit shifting, while FP8 processes 4 values per uint32_t with a single call.

**ncu verdict: FP4 dequant ALU is NOT the bottleneck.** All QMV kernels are
**L1TEX scoreboard limited** (89-98% of stall cycles), meaning they are purely
memory-bandwidth-bound. SM compute utilization is only 6-15%. The dequant operations
complete faster than memory can deliver the next batch of data.

**Impact depends on memory tier:**
- **DRAM-limited** (>24MB): FP4 and FP8 both achieve ~80% of DRAM peak. DRAM latency
  (~100ns) hides the dequant ALU cost. No significant bottleneck.
- **L2-cached** (<24MB): FP4 achieves ~51% of L2 BW while FP8 achieves ~83%. The faster
  L2 latency exposes the FP4 dequant cost. This is where optimization would help.

**FP4 IS faster than FP8 in wall time** because it reads half the bytes. The lower DRAM
utilization percentage does not mean FP4 is worse — it means FP4 has more compute per byte.

### ncu Deep Profiling Results

**MXFP8 Persistent (8192×8192, DRAM-limited):**
| Metric | Value |
|--------|-------|
| Duration | 394 µs |
| Dominant Stall | L1TEX scoreboard: **89%** of 29.8 cycles/inst |
| L1 Hit Rate | 3.2% (streaming, no cache reuse) |
| L2 Hit Rate | 2.8% (weight data >> 24MB L2) |
| Registers/Thread | 39 |
| Theoretical Occupancy | 83.3% |
| Achieved Occupancy | **16.4%** (1 block/SM, 8 warps) |
| Coalescing | 31.1/32 bytes = 97% (near perfect) |
| SM Busy | 6% |

**NVFP4 Single (4096×4096, L2-cached):**
| Metric | Value |
|--------|-------|
| Duration | 47-57 µs |
| Dominant Stall | L1TEX scoreboard: **91-95%** of 46-50 cycles/inst |
| L1 Hit Rate | **77.7%** (activation vector cached in L1) |
| L2 Hit Rate | 5.5% |
| Registers/Thread | 42 |
| Theoretical Occupancy | 83.3% |
| Achieved Occupancy | 73-86% (512 blocks, 35-41 warps/SM) |
| SM Busy | 12-15% |

**MXFP8 Single (4096×4096, L2-cached):**
| Metric | Value |
|--------|-------|
| Duration | 81-84 µs (**2× slower than NVFP4** for same shape) |
| Dominant Stall | L1TEX scoreboard: **94-98%** of 114-119 cycles/inst |
| L1 Hit Rate | 65.5% (lower than FP4 — larger working set evicts cache) |
| L2 Hit Rate | 3.2% |
| Registers/Thread | 36 |
| Theoretical Occupancy | **100%** |
| Achieved Occupancy | **88-92%** (highest, yet slowest!) |
| SM Busy | 7.4-7.6% |

**ncu Key Takeaways:**
1. Memory-bound at every operating point (L1TEX scoreboard = 89-98% of stalls)
2. Higher occupancy does NOT help — MXFP8 has 92% occupancy but 119 cycles/inst CPI
3. Data volume is the bottleneck: MXFP8 loads 2× bytes → 2× slower despite better occupancy
4. FP4's lower occupancy (73%) still wins because it reads half the bytes
5. Persistent kernel: 16.4% occupancy → 80% DRAM BW (contiguous access > occupancy)

### Persistent QMV Bank Conflict Analysis — INVESTIGATED, KEEP SMEM

**Problem**: Shared memory has 16-way bank conflicts. Thread k reads 64 bytes
at smem offset 64×k. Bank `(64k/4) % 32` yields only banks {0, 16} across 32
threads → 16 threads per bank. ncu reports 1.05M conflict cycles (50% of LDS).

**A/B Test: Remove smem, use L1 cache instead:**

| Kernel | With smem (baseline) | Without smem | Change |
|--------|:-------------------:|:----------:|:------:|
| NVFP4 gs=16 | 170 µs | 192 µs | **+13% regression** |
| MXFP4 gs=32 | ~170 µs | 193 µs | **+13% regression** |
| MXFP8 gs=32 | 321 µs | 317 µs | -1.3% (noise) |

**Root cause of regression**: Removing smem pushes activation reads through L1TEX
alongside weight reads. For FP4: activation = 80% of L1TEX traffic (64B activation
vs 16B weight per warp step). L1TEX port becomes 5× more loaded → bandwidth drops.
For MXFP8: activation is only 33% of traffic (64B / 192B total) → minimal impact.

**Conclusion**: Bank conflicts (16× LDS replay) add ~15 cycles per instruction, but
are completely hidden by 100-200ns DRAM latency on the weight L1TEX path. The separate
LDS port for smem is critical for FP4 because it offloads 80% of data access from
L1TEX. **Keep smem; bank conflicts are not on the critical path.**

Scale factor loads also have sub-optimal efficiency (50% for MXFP8 gs=32: thread pairs
access same scale byte via integer division), but scale data is <6% of total traffic
per step, so the impact is negligible.

### End-to-End vs Kernel-Level

| Measurement | NVFP4 8192×8192 M=1 | MXFP8 8192×8192 M=1 |
|------------|---------------------|---------------------|
| Kernel time (nsys) | 170 µs | 321 µs |
| Wall time (bench_qmm.py) | 849 µs | 807 µs |
| Framework overhead | ~679 µs | ~486 µs |

Framework overhead (MLX eval/graph/memory allocation) adds 380-680µs per M=1 call,
dominating the end-to-end measurement. This overhead varies by mode and shape, making
bench_qmm.py unreliable for kernel-level optimization validation. **Always use nsys
kernel-level times for optimization work.**

### Benchmark Methodology

bench_qmm.py follows the Marlin benchmark pattern correctly for large operations:
- `mx.synchronize()` before and after timing window
- `mx.eval(y)` in loop (necessary for MLX lazy evaluation; eval overhead is negligible
  for large ops — tested: 12.59ms/iter with vs without eval)
- `time.sleep(COOLDOWN)` between benchmarks (0.3s vs Marlin's 1.0s)

For M=1 measurements, framework overhead dominates. Use nsys for kernel-level analysis.

### Latest Benchmark Results (bench_qmm.py, with SFB cache)

**Highlights (M=1 decode, GB/s):**

| Shape | INT4-gs64 | NVFP4 | MXFP4 | MXFP8 |
|-------|-----------|-------|-------|-------|
| 7B (4096²) | 20.7 | 20.0 | 115.8 | 99.6 |
| 13B (5120²) | 16.1 | 66.1 | 62.7 | 44.2 |
| 70B (8192²) | 33.7 | 44.2 | 54.6 | 70.9 |

**Highlights (M=256 prefill, TFLOP/s):**

| Shape | INT4-gs64 | NVFP4 | MXFP4 | MXFP8 |
|-------|-----------|-------|-------|-------|
| 7B (4096²) | 7.5 | 35.3 | 45.0 | 18.6 |
| 13B (5120²) | 9.4 | 59.1 | 20.5 | 31.8 |
| 70B (8192²) | 12.5 | 58.3 | 39.3 | 22.2 |

**Highlights (M=1024 prefill, TFLOP/s):**

| Shape | INT4-gs64 | NVFP4 | MXFP4 | MXFP8 |
|-------|-----------|-------|-------|-------|
| 7B (4096²) | 32.3 | 37.2 | 36.5 | 36.7 |
| 13B (5120²) | 30.7 | 65.6 | 84.8 | 54.8 |
| 70B (8192²) | 38.3 | 92.3 | 92.6 | 63.1 |

**Note**: bench_qmm.py includes framework overhead (380-680µs at M=1), so M=1 numbers
are unreliable for kernel comparison. Small-M numbers (M=16, M=64, M=256) are more
meaningful since kernel time dominates. The nsys pipeline times above are the authoritative
kernel-level measurements.

**Key bench observations (70B shape, vs Dense):**
- NVFP4 M=256: **2.46×** vs Dense (58.3 TFLOP/s)
- NVFP4 M=1024: **1.77×** vs Dense (92.3 TFLOP/s)
- MXFP4 M=256: **1.66×** vs Dense
- MXFP4 M=1024: **1.77×** vs Dense (92.6 TFLOP/s — matches NVFP4)

---

## Remaining Gaps & Opportunities

### Gap 1: Framework Overhead at M=1 (HIGH IMPACT)

**Problem**: 380-680µs MLX eval/graph/memory overhead dominates M=1 wall time.
Kernel takes 10-321µs but end-to-end takes 70-849µs.

**Potential approaches:**
- Reduce CUDA graph compilation/replay overhead for small kernels
- Pre-allocate output buffers to eliminate allocation in critical path
- Batch multiple QMV calls (attention K/Q/V projections) into single graph
- This is a framework-level issue, not a kernel-level issue

**Impact**: Would improve bench_qmm.py numbers dramatically but requires MLX core changes.

### Gap 2: QMV Memory Latency Hiding (MEDIUM IMPACT)

**Problem**: ncu confirms ALL QMV kernels are **L1TEX scoreboard limited** (89-98%).
CPI is 30-119 cycles, with SM compute only 6-15% busy. This is pure memory-bandwidth-bound.

**What ncu rules out:**
- FP4 dequant ALU is NOT the bottleneck (SM only 6-15% busy)
- Coalescing is near-perfect (31.1/32 bytes = 97% sector utilization)
- Occupancy is not the answer (MXFP8 single has 92% occupancy but worst CPI of 119)
- ~~Shared memory bank conflicts~~ — 16-way conflicts exist but are hidden by DRAM
  latency. Removing smem **regressed** FP4 by 13% (L1TEX port congestion). See analysis above.

**Remaining optimization vectors:**
- **Increase ILP**: n=8 template to overlap more compute with in-flight memory loads
- **Prefetch with `__ldg()`**: Use read-only data cache path for weight loads
- **Double-buffer in registers**: Load next group while computing current group
- **Tensor core GEMV**: Use `mma.sync.aligned.block_scale` for M=1 — hardware FP4
  decompression, bypasses software dequant entirely
- **L2 residency control**: Use `cudaAccessPolicyWindow` to pin activation vector in L2

**Impact**: Would help 7B and 13B decode (L2-cached shapes). For 70B+, persistent at 80%.

### Gap 3: SM120 GEMM Pipeline Overhead — SFB CACHING APPLIED

**Status**: OPTIMIZED — `reformat_sf` eliminated via SFB caching (Next Steps #2).

#### Before: 3-Kernel Pipeline (nsys, N=K=8192)

**NVFP4:**
| M | quantize_fp4 | reformat_sf | GEMM kernel | Total | Overhead % |
|---|-------------|-------------|-------------|-------|------------|
| 16 | 3.6 µs | 63.0 µs | 267 µs | 334 µs | **20.0%** |
| 64 | 11.0 µs | 63.6 µs | 283 µs | 357 µs | **20.9%** |
| 128 | 20.7 µs | 63.9 µs | 293 µs | 378 µs | **22.4%** |
| 256 | 40.2 µs | 64.1 µs | 308 µs | 412 µs | **25.3%** |
| 512 | 79.4 µs | 63.7 µs | 336 µs | 479 µs | **29.9%** |
| 1024 | 159 µs | 64.8 µs | 481 µs | 705 µs | **31.8%** |

**MXFP8:**
| M | quantize_fp8 | reformat_sf | GEMM kernel | Total | Overhead % |
|---|-------------|-------------|-------------|-------|------------|
| 16 | 4.0 µs | 37.3 µs | 472 µs | 513 µs | **8.1%** |
| 64 | 12.7 µs | 32.8 µs | 471 µs | 517 µs | **8.9%** |
| 128 | 23.5 µs | 32.3 µs | 488 µs | 544 µs | **10.3%** |
| 256 | 39.8 µs | 32.6 µs | 610 µs | 682 µs | **10.6%** |
| 512 | 77 µs | 33 µs | 737 µs | 847 µs | **13.0%** |
| 1024 | 155 µs | 33.3 µs | 1,386 µs | 1,574 µs | **12.0%** |

#### After SFB Cache: 2-Kernel Pipeline (nsys, N=K=8192)

**NVFP4:**
| M | quantize_fp4 | GEMM kernel | Total | vs 3-kernel |
|---|-------------|-------------|-------|-------------|
| 16 | 3.8 µs | 214.8 µs | **218.6 µs** | **-34.6%** |
| 64 | 11.3 µs | 215.3 µs | **226.5 µs** | **-36.6%** |
| 128 | 21.0 µs | 225.2 µs | **246.2 µs** | **-34.9%** |
| 256 | 42.5 µs | 262.5 µs | **305.0 µs** | **-26.0%** |
| 512 | 87.1 µs | 322.3 µs | **409.4 µs** | **-14.5%** |
| 1024 | 167.0 µs | 461.3 µs | **628.3 µs** | **-10.9%** |

**MXFP8:**
| M | quantize_fp8 | GEMM kernel | Total | vs 3-kernel |
|---|-------------|-------------|-------|-------------|
| 16 | 7.6 µs | 476.5 µs | **484.0 µs** | **-5.7%** |
| 64 | 14.3 µs | 485.9 µs | **500.1 µs** | **-3.3%** |
| 128 | 27.3 µs | 505.0 µs | **532.3 µs** | **-2.2%** |
| 256 | 42.8 µs | 611.8 µs | **654.6 µs** | **-4.0%** |
| 512 | 81.9 µs | 729.7 µs | **811.6 µs** | **-4.2%** |
| 1024 | 172.5 µs | 1,246.1 µs | **1,418.7 µs** | **-9.9%** |

#### After Vectorized Quantization: 2-Kernel Pipeline (nsys, N=K=8192)

**NVFP4:**
| M | quantize_fp4 | GEMM kernel | Total | vs SFB-only | vs 3-kernel |
|---|-------------|-------------|-------|-------------|-------------|
| 16 | 3.1 µs | 210.3 µs | **213.4 µs** | **-2.4%** | **-36.1%** |
| 64 | 8.2 µs | 211.5 µs | **219.7 µs** | **-3.0%** | **-38.5%** |
| 128 | 14.5 µs | 221.8 µs | **236.3 µs** | **-4.0%** | **-37.5%** |
| 256 | 30.6 µs | 247.7 µs | **278.2 µs** | **-8.8%** | **-32.5%** |
| 512 | 61.6 µs | 294.4 µs | **355.4 µs** | **-13.2%** | **-25.8%** |
| 1024 | 127.4 µs | 458.0 µs | **584.9 µs** | **-6.9%** | **-17.0%** |

**MXFP8:**
| M | quantize_fp8 | GEMM kernel | Total | vs SFB-only | vs 3-kernel |
|---|-------------|-------------|-------|-------------|-------------|
| 16 | 3.4 µs | 439.2 µs | **441.6 µs** | **-8.8%** | **-13.9%** |
| 64 | 10.8 µs | 475.0 µs | **487.0 µs** | **-2.6%** | **-5.8%** |
| 128 | 18.7 µs | 492.6 µs | **511.4 µs** | **-3.9%** | **-6.0%** |
| 256 | 35.0 µs | 568.7 µs | **604.5 µs** | **-7.7%** | **-11.4%** |
| 512 | 72.0 µs | 713.4 µs | **786.1 µs** | **-3.1%** | **-7.2%** |
| 1024 | 146.7 µs | 1,248.1 µs | **1,396.4 µs** | **-1.6%** | **-11.3%** |

#### Analysis

1. **reformat_sf fully eliminated** (SFB caching): nsys confirms only warmup instances.
2. **Vectorized quantization** (4 elements/thread): 1.2-2.2x faster quantize kernels.
   FP4 quantize at M=1024: 167→127 µs, FP8: 173→147 µs.
3. **Cumulative pipeline improvement** from original 3-kernel baseline:
   - NVFP4 M=512: 479 µs → **355 µs** (**-25.8%**, 124 µs saved)
   - NVFP4 M=1024: 705 µs → **585 µs** (**-17.0%**, 120 µs saved)
   - MXFP8 M=16: 513 µs → **442 µs** (**-13.9%**)
4. **Quantization is now a smaller fraction of pipeline** (NVFP4):

   | M | quantize | GEMM | quantize % of total |
   |---|---------|------|---------------------|
   | 16 | 3.1 µs | 210.3 µs | **1.5%** |
   | 256 | 30.6 µs | 247.7 µs | **11.0%** |
   | 1024 | 127.4 µs | 458.0 µs | **21.8%** |

   Down from 26.6% to 21.8% at M=1024. Remaining 127 µs achieves ~161 GB/s (59% of DRAM
   peak). Further improvement possible with wider vectorization (uint4 = 16 bytes) or
   grid-stride loop, but diminishing returns — the GEMM is now 78% of pipeline time.

#### ncu SM120 GEMM Kernel Analysis (NVFP4 8192×8192, with SFB cache)

**M=256 (128 CTAs = 2 M-tiles × 64 N-tiles):**
| Metric | Value |
|--------|-------|
| Duration | 318 µs |
| SM Busy (Compute) | **25.7%** |
| Tensor Pipe | **25.7%** (dominant pipeline) |
| L1 Hit Rate | 84.3% |
| L2 Hit Rate | 70.9% |
| CPI | 28.3 cycles |
| Registers/Thread | **168** |
| Dynamic Smem | **88 KB/block** |
| Theoretical Occupancy | 25% (1 CTA/SM) |
| Block Limit | Registers AND Shared Memory (both = 1 block) |
| Local Memory Spilling | 2.2/32 bytes utilized per sector (register spill) |
| Wave Tail | **33.3%** (partial third wave) |
| Stall: blocked/sleeping | 68.4% of 28.3 avg cycles |

**M=1024 (512 CTAs = 8 M-tiles × 64 N-tiles):**
| Metric | Value |
|--------|-------|
| Duration | 601 µs |
| SM Busy (Compute) | **54.9%** |
| Tensor Pipe | **54.9%** (dominant pipeline) |
| L1 Hit Rate | 99.9% |
| L2 Hit Rate | 89.9% |
| CPI | 19.6 cycles |
| Same resource limits as M=256 |
| Stall: blocked/sleeping | 57.0% of 19.6 avg cycles |

**ncu GEMM Key Takeaways:**
1. **Register spilling exists but is well-behaved**: 168 regs/thread forces 1 CTA/SM and
   causes local memory spills (2.2/32 bytes per sector = 7% utilization per access). BUT
   L1 hit rate on local spills is 97.6-99.4%, so spills are absorbed by L1 and generate
   almost no DRAM traffic. Spill volume is tiny vs weight/activation TMA traffic (~83K
   sectors local vs ~4.9M sectors L2 at M=256). This is CUTLASS's known tradeoff: large
   tiles need many registers, pipeline stages hide the spill latency.
2. **Occupancy limits tensor pipe utilization**: 25% occupancy (1 CTA/SM, 12 warps) means
   not enough warps to keep tensor pipe continuously fed between pipeline stages. At M=1024
   the kernel crosses the roofline into compute-bound territory (arithmetic intensity 2320
   ops/byte vs crossover at ~1832), yet tensor cores are only 55% busy. This is an inherent
   ceiling from the CUTLASS tile choice — BM/BN≥128 is constrained, can't be changed.
3. **Wave quantization** hurts small M: M=256 has 128 CTAs on 48 SMs → 2.7 waves with
   67% last-wave efficiency. M=1024 has 10.7 waves → amortized.
4. **L2 hit rate improves with more waves**: 71% (M=256) vs 90% (M=1024) due to
   temporal locality of weight tiles across successive M-tile waves.
5. **Pingpong schedule works correctly**: blocked/sleeping stalls (57-68%) are the correct
   signal for warp-specialized kernels. Producer warps intentionally sleep while consumer
   warps run tensor ops, and vice versa. Combined with barrier sync between producer/consumer,
   these account for 72-82% of all stall cycles — the pipeline is overlapping TMA loads
   with MMA compute as designed.
6. **Tensor pipe utilization scales with M**: 25.7% (M=256) → 54.9% (M=1024). At M=1024
   the GEMM is reasonably efficient; small M is limited by wave quantization and occupancy.

### Gap 4: SM120 GEMM Tile Shape Tuning — INVESTIGATED, NO BENEFIT

**Status: CLOSED** — All alternatives tested, none provide meaningful improvement.

**Current**: 128×128×128 tile, Pingpong schedule, 1 CTA/SM.

#### Tile Shape Experiments (2025-03)

| Approach | Result | Detail |
|----------|--------|--------|
| **64×128×128** (reduce BM) | FAILS to compile | CUTLASS TMA scale factor layout requires BM ≥ 128 |
| **128×64×128** (reduce BN) | FAILS to compile | CUTLASS TMA scale factor layout requires BN ≥ 128 |
| **StreamK scheduler** | <6% benefit | Cooperative+StreamK tested at GEMM kernel level via nsys |

**CUTLASS constraint**: SM120 block-scaled TMA descriptors require **both BM ≥ 128 AND BN ≥ 128**.
The minimum tile shape is 128×128×K for all scale factor types (ue4m3, ue8m0).
Error at `copy_traits_sm90_tma.hpp:1227`: "too many initializer values" for scale factor TMA.

#### StreamK A/B Results (nsys kernel-level, NVFP4, N=K=8192)

| M | Pingpong µs | StreamK µs | Speedup | PP Grid | SK Grid |
|---|------------|-----------|---------|---------|---------|
| 64 | 239 | 238 | 1.00x | 1×64=64 | 48×1=48 |
| 128 | 251 | 237 | 1.06x | 1×64=64 | 48×1=48 |
| 256 | 296 | 286 | 1.03x | 2×64=128 | 2×64=128 |
| 512 | 327 | 336 | 0.97x | 4×64=256 | 48×5=240 |

StreamK uses Cooperative schedule (Pingpong has a static_assert blocking StreamK).
The marginal gain on bandwidth-limited LPDDR5x doesn't justify the complexity.
Additionally, StreamK workspace allocation breaks CUDA graph replay (produces NaN).

**Conclusion**: 128×128×128 Pingpong is optimal for SM121/LPDDR5x. The bottleneck is
DRAM bandwidth, not wave utilization. Future CUTLASS versions or hardware with higher
compute/memory ratio might change this.

### Gap 5: CUTLASS Upgrade Assessment

**Current**: CUTLASS v4.3.5 (Jan 2026)
**Latest**: CUTLASS v4.4.1 (Feb 2026)

**4.4.0 adds for SM120/SM121:**
- `scale_vec::4X` block-scaled PTX support (requires CUDA 13.1)
  - Packs 4× more scale factors per MMA instruction
  - Could improve FP4 block-scaled throughput
- `cute.experimental` fragment-free programming (Python DSL)
- AoT compilation support
- GB300/SM103 support

**4.4.1**: Bugfix only (aarch64 tvm-ffi segfault fix)

**Assessment**: Upgrade worthwhile only if using CUDA 13.1 for `scale_vec::4X`.
On CUDA 13.0 (current), no material benefit for SM121 kernels. All SM120/SM121
kernel code paths have been stable since CUTLASS 4.2.0.

**Key CUTLASS references:**
- [Example 79a: Blackwell GeForce NVFP4 GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- [Issue #2947: SM121 lacks tcgen05](https://github.com/NVIDIA/cutlass/issues/2947) (hardware limitation, not fixable)
- [Issue #2800: BlockScaledMmaOp restricts FP4 to sm_100a in Python DSL](https://github.com/NVIDIA/cutlass/issues/2800)
- [Community: 129 TFLOPS FP4 on DGX Spark](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600)

### Gap 6: INT4 Affine Performance

INT4 affine modes (gs=64, gs=128) consistently underperform FP modes. The CuTe
kernel path (M≤512) helps, but large M performance lags behind NVFP4/MXFP4.
The dequantize-then-cuBLAS fallback for M>512 loses to native FP4.

### Gap 7: Real-Model End-to-End Validation (NOT DONE)

No tok/s measurements on actual LLM inference. All benchmarks are isolated matmul
microbenchmarks. Need to validate that kernel improvements translate to model-level
speedups. Framework overhead may dominate for decode (M=1).

### Gap 8: GatherQMM / MoE Status (NOT ADDRESSED)

GatherQMM (fused gather+QMM for MoE models like DeepSeek V3) has not been optimized
for SM120. The gather_qmv.cu dispatch fixes (Phase 2) are done, but there is no
SM120 native GatherQMM path. MoE models fall back to dequant+cuBLAS for the
gathered matmul. **Explicitly a future work item, not a current priority.**

### Gap 10: Activation Quantization Kernel Overhead — VECTORIZED

**Status**: OPTIMIZED — Vectorized 4-elements-per-thread activation quantization.

**Problem (before)**: Each thread processed 1 element, no vectorized loads, 50% FP4 write
divergence (only even lanes wrote). At M=1024 (NVFP4, N=K=8192): 167 µs, ~96 GB/s.

**Solution**: Restructured all 4 activation quantization kernels (FP4×2, FP8×2) to process
4 elements per thread via `uint2` vectorized loads (8 bytes):
- **FP4 NVFP4** (SFV=16): THREADS_PER_GROUP 16→4, GROUPS_PER_WARP 2→8, shuffle iters 4→2
- **FP4 MXFP4** (SFV=32): THREADS_PER_GROUP 32→8, GROUPS_PER_WARP 1→4, shuffle iters 5→3
- **FP8** (SFV=32): THREADS_PER_GROUP 32→8, GROUPS_PER_WARP 1→4, shuffle iters 5→3
- FP4 write utilization: 50% (even lanes) → **100%** (every thread writes 2 packed bytes)
- FP8 write: 1 byte/thread → **4 bytes via uint32_t store**

**Kernel-level improvement (nsys, N=K=8192):**

NVFP4:
| M | Before quant | After quant | Speedup | Before total | After total | Pipeline |
|---|-------------|------------|---------|-------------|------------|----------|
| 16 | 3.8 µs | 3.1 µs | 1.2x | 218.6 µs | **213.4 µs** | **-2.4%** |
| 64 | 11.3 µs | 8.2 µs | 1.4x | 226.5 µs | **219.7 µs** | **-3.0%** |
| 256 | 42.5 µs | 30.6 µs | 1.4x | 305.0 µs | **278.2 µs** | **-8.8%** |
| 512 | 87.1 µs | 61.6 µs | 1.4x | 409.4 µs | **355.4 µs** | **-13.2%** |
| 1024 | 167.0 µs | 127.4 µs | 1.3x | 628.3 µs | **584.9 µs** | **-6.9%** |

MXFP8:
| M | Before quant | After quant | Speedup | Before total | After total | Pipeline |
|---|-------------|------------|---------|-------------|------------|----------|
| 16 | 7.6 µs | 3.4 µs | 2.2x | 484.0 µs | **441.6 µs** | **-8.8%** |
| 64 | 14.3 µs | 10.8 µs | 1.3x | 500.1 µs | **487.0 µs** | **-2.6%** |
| 256 | 42.8 µs | 35.0 µs | 1.2x | 654.6 µs | **604.5 µs** | **-7.7%** |
| 1024 | 172.5 µs | 146.7 µs | 1.2x | 1418.7 µs | **1396.4 µs** | **-1.6%** |

**Quantize kernel bandwidth**: At M=1024, FP4: 20.5 MB / 127 µs = **161 GB/s** (was 96 GB/s,
+68%). Still below 273 GB/s peak — remaining gap likely from scale factor layout writes
(scattered via CuTe layout_sfa).

### Gap 9: L2 Threshold Brittleness

The persistent QMV kernel threshold is exactly 24MB (L2 cache size). In real inference:
- Other kernels (attention, normalization) compete for L2 space
- Actual effective L2 capacity during inference may be 16-18MB
- The 24MB threshold might need to be lowered to 16-18MB for production use
- Needs profiling under real inference workload to determine optimal threshold

---

## Architecture Notes

### SM120 vs SM121
Nearly identical. SM120 = GeForce RTX 50-series, SM121 = DGX Spark GB10.
Both use `mma.sync.aligned.block_scale` (NOT `tcgen05`). CUTLASS treats them
identically. No TMA multicast, no TMEM, cluster must be 1×1×1.

### QMV Kernel Architecture
- `fp_qmv_single` (original): N/8 blocks × M blocks, 256 threads/block, 0 smem
- `fp_qmv_persistent` (new): 1×48 blocks, 256 threads/block, K×sizeof(T) smem
  - Smem has 16-way bank conflicts (hidden by DRAM latency — not on critical path)
  - Smem offloads activation reads from L1TEX port — critical for FP4 (80% of traffic)
- Template params: `<T, rows_per_block=8, n_per_thread={1,2,4}, bits, group_size, use_mx_scale>`
- Inner loop: load uint32_t weights → dequant → FMA with activation → warp reduce

### SM120 GEMM Pipeline
Two kernels per call (after SFB caching optimization):
1. `quantize_activation_{fp4,fp8}_kernel` — FP16 → quantized + scales
2. CUTLASS block-scaled GEMM — the actual matmul

Weight scale factor reformatting (`reformat_sf_kernel`) runs once on first call per
(scales_ptr, N, K, group_size) and is cached via `get_or_reformat_sfb<GemmType>()`.
Cache uses `cudaMalloc` (persistent) with thread-safe mutex. Call `clear_sm120_sf_cache()`
when unloading models to free GPU memory.

---

## Next Steps (Priority Order — per reviewer feedback)

1. ~~**Tile shape sweep for SM120 GEMM**~~ — **CLOSED (2025-03)**. All alternatives
   tested: 64×128×128 fails (BM≥128 TMA constraint), 128×64×128 fails (BN≥128),
   StreamK gives <6% GEMM-level gain (bandwidth-limited, not wave-limited).
   128×128×128 Pingpong is optimal for SM121/LPDDR5x.

1b. ~~**Persistent QMV bank conflicts**~~ — **CLOSED (2025-03)**. 16-way smem bank
   conflicts exist but are hidden by DRAM latency. Removing smem regressed FP4
   by 13% due to L1TEX port congestion. Keep smem. Scale load efficiency (50% for
   some configs) is <6% of total traffic — not worth fixing.

2. ~~**Cache reformatted weight scale factors**~~ — **DONE (2025-03)**. Caches reformatted
   SFB buffers keyed on (scales_ptr, N, K, group_size). `reformat_sf_kernel` runs once on
   first call (warmup), then zero overhead on all subsequent calls. Saves 64 µs/call (FP4)
   and 33 µs/call (FP8) — verified via nsys: 1 instance of reformat_sf across 101 GEMM
   calls. Correctness: bit-exact output across all formats and M values (cached vs fresh).
   Implementation: `get_or_reformat_sfb<GemmType>()` template + `SFBCacheKey` hash map in
   `qmm_sm120.cu`, `clear_sm120_sf_cache()` in header for model unloading.

3. **Framework overhead investigation** — 380-680µs per M=1 call. Highest user-visible
   impact (dominates decode latency) but requires MLX core changes. Profile CUDA graph
   creation/replay, memory allocation, and eval() overhead. Nothing else matters for
   decode tok/s until this is addressed.

4. ~~**Activation quantization kernel optimization**~~ — **DONE (2026-03)**. Vectorized all 4
   activation quantization kernels to process 4 elements/thread via `uint2` loads. Quantize
   kernel 1.2-2.2x faster. Pipeline improvement: NVFP4 M=512 -13.2%, M=1024 -6.9%.
   Cumulative from 3-kernel baseline: NVFP4 M=512 -25.8%, M=1024 -17.0%.
   Remaining quantize overhead at M=1024: 21.8% (down from 26.6%), ~161 GB/s.

5. **Real-model validation** — Becoming urgent. Kernel-level work is mature (3 ncu reports,
   comprehensive nsys data, SFB caching, closed tile/bank-conflict investigations). The next
   meaningful signal must come from actual model inference. Even a single Llama-70B
   decode/prefill comparison would validate the effort and identify whether attention/KV-cache/
   sampling are now the bottleneck.

6. **QMV ILP/latency hiding** (MEDIUM) — Remaining vectors: n=8 template, double-buffer in
   registers, `__ldg()` prefetch. Ceiling is modest — persistent QMV already at 81% DRAM BW
   on large shapes. Would mainly help L2-cached shapes (7B/13B decode).

7. **Tensor core GEMV** (LOW, HIGH-RISK) — Use `mma.sync.aligned.block_scale` for M=1.
   Ceiling is only ~23 GB/s improvement (222 → 245 GB/s). Complex to implement, may not
   yield measurable improvement given framework overhead dominance at M=1.

8. **L2 threshold tuning** — Profile persistent QMV under real inference workload to
   determine if 24MB threshold needs lowering to account for L2 contention.

---

## File Reference

| File | Purpose |
|------|---------|
| `mlx/backend/cuda/quantized/qmv.cu` | QMV kernels (original + persistent) |
| `mlx/backend/cuda/quantized/gather_qmv.cu` | GatherQMV kernel |
| `mlx/backend/cuda/quantized/qmm.cu` | CuTe QMM kernel (affine) |
| `mlx/backend/cuda/quantized/qmm_sm120.cu` | SM120 native GEMM (FP4/FP8) |
| `mlx/backend/cuda/quantized/quantized.cpp` | Dispatch logic (mode/shape → kernel) |
| `mlx/backend/cuda/quantized/quantized_utils.cuh` | dequant_fp4, dequant_fp8, dispatch helpers |
| `mlx/backend/cuda/device/utils.cuh` | AlignedVector, load_vector, unsafe_load_vector |
| `bench_qmm.py` | End-to-end benchmark script |
| `ncu_reports/gemm_sm120_sfb_cached.ncu-rep` | ncu GEMM: NVFP4 M=256 + M=1024 (with SFB cache) |
