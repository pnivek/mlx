# Exploration 06: cudaGraphExecUpdate Bug on SM121 / CUDA 13.0

**Status:** TWO FIXES IMPLEMENTED
**Date:** 2026-03-04 to 2026-03-05
**Platform:** DGX Spark (GB10), SM121, CUDA 13.0, Driver 570.86.16

## Summary

`cudaGraphExecUpdate` on SM121/CUDA 13.0 silently corrupts execution when updating a
cached `cudaGraphExec_t`. Two distinct manifestations were found:

1. **Cross-kernel corruption** (affects ALL MLX operations): When the graph cache key
   maps different kernel functions to the same entry, `cudaGraphExecUpdate` updates an
   exec from kernel A to kernel B. This returns success but produces degraded/corrupted
   execution. **Fix: include kernel function pointer in graph cache key.**

2. **Same-kernel corruption** (affects GatherQMM): Even with the same kernel function,
   updating buffer pointers and grid dimensions causes illegal memory access in
   multi-kernel graphs. **Workaround: direct kernel launch bypass.**

## Bug 1: Cross-Kernel Graph Cache Collision (upstream-worthy fix)

### Root cause

MLX's graph cache key (`graph_nodes_key_ + ":" + graph_deps_key_`) only encodes node
types (`"K"` for all kernels) and dependency structure. **All single-kernel operations
share the cache key `"K-:"`**, regardless of which kernel function they contain.

When different kernels hit the same cache entry, `cudaGraphExecUpdate` is called to
update the cached exec with a completely different kernel function. On SM121, this returns
`cudaGraphExecUpdateSuccess` but silently corrupts the exec, causing ~3-5x performance
degradation on subsequent launches.

### Reproduction

```python
import mlx.core as mx
import time

K, N = 4096, 4096  # Llama-7B

# Step 1: Run INT4 quantized matmul (uses cute_qmm kernel)
# This caches a graph exec under key "K-:"
w_fp = mx.random.normal(shape=(N, K)).astype(mx.float16)
w_q, s, b = mx.quantize(w_fp, group_size=64, bits=4)
x = mx.random.normal(shape=(1, K)).astype(mx.float16)
mx.eval(w_q, s, b, x)
for _ in range(20):  # warmup
    y = mx.quantized_matmul(x, w_q, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(y)

# Step 2: Run NVFP4 quantized matmul (uses fp_qmv_single kernel)
# Same graph topology "K-:" -> cudaGraphExecUpdate replaces cute_qmm with fp_qmv_single
# BUG: update returns success but exec is corrupted
result = mx.quantize(w_fp, group_size=16, bits=4, mode="nvfp4")
w_q2, s2 = result[0], result[1]
mx.eval(w_q2, s2)
for _ in range(20):
    y = mx.quantized_matmul(x, w_q2, s2, transpose=True, group_size=16, bits=4, mode="nvfp4")
    mx.eval(y)

# Measure: NVFP4 should be ~0.035ms but shows ~0.14ms (4x slower)
mx.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    y = mx.quantized_matmul(x, w_q2, s2, transpose=True, group_size=16, bits=4, mode="nvfp4")
    mx.eval(y)
mx.synchronize()
print(f"NVFP4 M=1: {(time.perf_counter() - t0) / 100 * 1000:.3f} ms")
# Expected: ~0.035ms, Actual: ~0.14ms
```

### Measured impact

Tested with bench_qmm.py which runs all quantization modes sequentially (same process):

| Shape | Without fix | With fix | Regression |
|-------|------------|----------|------------|
| Llama-7B NVFP4 M=1 | 0.147ms | 0.036ms | **4.1x** |
| Llama-13B NVFP4 M=1 | 0.151ms | 0.041ms | **3.7x** |
| DSv3-MLP NVFP4 M=1 | 0.143ms | 0.030ms | **4.8x** |
| Llama-7B MXFP4 M=1 | 0.031ms | 0.033ms | 1.0x (unaffected) |

MXFP4 appeared unaffected because it runs AFTER NVFP4 in the benchmark — the corrupted
NVFP4 exec causes MXFP4's update to fail (error return), triggering fresh re-instantiation
which works correctly.

Isolated test confirming the mechanism:

| Test | Without fix | With fix |
|------|------------|----------|
| NVFP4 alone (no prior ops) | 0.047ms | 0.047ms |
| MXFP4 after NVFP4 (cross-kernel) | **0.168ms** | 0.050ms |
| NVFP4 after INT4+flush | **0.088ms** | 0.047ms |
| NVFP4 after contamination | **0.091ms** | 0.048ms |

### Fix (upstream-worthy)

Include the kernel function pointer in the graph node type string, so different kernels
produce different graph cache keys:

```cpp
// device.cpp — add_kernel_node (cudaKernelNodeParams overload)
void CommandEncoder::add_kernel_node(const cudaKernelNodeParams& params) {
  cudaGraphNode_t node;
  CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&node, graph_, NULL, 0, &params));
  // Include kernel function pointer in key so different kernels get different
  // graph cache entries. Prevents cudaGraphExecUpdate from updating a cached
  // exec with a completely different kernel, which corrupts execution on SM121.
  auto key = fmt::format("K{:x}", reinterpret_cast<uintptr_t>(params.func));
  insert_graph_dependencies(GraphNode{node, key});
}

// Same change for CUfunction (CUDA_KERNEL_NODE_PARAMS) overload.
```

**Before:** All single-kernel ops share `"K-:"` → cross-kernel `cudaGraphExecUpdate`
**After:** Each kernel gets its own key (e.g., `"K7f3a4c0800-:"`) → only same-kernel updates

**Trade-off:** More graph cache entries (one per unique kernel function), but each entry
is more likely to be correctly reusable. The LRU cache handles eviction. Even on GPUs
without the SM121 bug, this is strictly more correct — `cudaGraphExecUpdate` does less
work when only buffer pointers change vs. kernel function + all params.

### Why this is upstream-worthy

1. **Affects all MLX CUDA users on SM121**, not just quantized matmul. Any sequence of
   different single-kernel operations (layernorm → softmax → matmul) would trigger it.
2. **Silent performance degradation**, not a crash. Users would never know they're
   running 3-5x slower without profiling.
3. **The fix is correct on all GPUs**. Including function pointers in cache keys makes
   the cache more semantically accurate, even where `cudaGraphExecUpdate` works correctly.
4. **6-line change** in `device.cpp`, no behavioral change for same-kernel updates.

## Bug 2: Same-Kernel Graph Exec Corruption (GatherQMM-specific workaround)

### Root cause

Even with the graph cache key fix above, multi-kernel graph sequences with variable grid
dimensions and dynamically allocated buffers still trigger corruption on SM121. The sorted
gather QMV path (6 kernels) crashes with illegal memory access when `cudaGraphExecUpdate`
patches grid dims and buffer pointers for the same kernel functions.

### Workaround

Added `begin_direct_launch()` / `end_direct_launch()` mode to `CommandEncoder`. When
active, `add_kernel_node` uses `cudaLaunchKernel` directly instead of adding to the
CUDA graph. The sorted gather QMV path uses this:

```cpp
enc.commit();                 // flush pending graph work
enc.begin_direct_launch();    // bypass graph capture
// ... sort + QMV + scatter kernels (6 kernels) ...
enc.end_direct_launch();      // resume normal graph capture
```

Performance impact: < 1% (~30-60µs launch overhead vs 8-72ms kernel time).

This workaround is NOT upstream-worthy in isolation — it's specific to GatherQMM's sorted
QMV path which only exists in our fork. However, the `begin_direct_launch` API could be
useful upstream if other multi-kernel paths hit the same issue.

## Environment details

```
GPU: NVIDIA GB10 (DGX Spark)
  Compute Capability: 12.1 (sm_121)
  L2 Cache: 24 MB
  SMs: 48
  Memory: 128 GB LPDDR5x (273 GB/s sustained)

CUDA Toolkit: 13.0
CUDA Driver: 570.86.16
CUTLASS: 4.3.5
OS: Ubuntu 24.04 (aarch64)
```

## Suggested NVIDIA investigation

1. `cudaGraphExecUpdate` returns `cudaGraphExecUpdateSuccess` but produces corrupted/
   degraded execution when updating a cached exec with a different kernel function on
   SM121. Verified by comparing kernel timing before and after cross-kernel update.
2. Even with the same kernel function, updating grid dims and buffer pointers in
   multi-kernel graphs causes illegal memory access.
3. `compute-sanitizer` cannot reproduce because it disables CUDA graphs. Consider adding
   a graph-aware validation mode.
4. The bugs do NOT reproduce with `cudaLaunchKernel` directly, confirming they are
   graph-specific.

## Files modified

| File | Change | Upstream? |
|------|--------|-----------|
| `mlx/backend/cuda/device.cpp` | Kernel func ptr in graph cache key | **Yes** |
| `mlx/backend/cuda/device.cpp` | `direct_launch_` check in `add_kernel_node` | Maybe |
| `mlx/backend/cuda/device.h` | `begin/end_direct_launch()`, `disable_graph_update()` | Maybe |
| `mlx/backend/cuda/quantized/quantized.cpp` | Sorted gather QMV uses direct launch | No (fork-specific) |
| `mlx/backend/cuda/quantized/gather_qmm.cu` | `sort_gather_indices`, `scatter_gather_output` | No (fork-specific) |
| `mlx/backend/cuda/quantized/gather_qmm.h` | `SortedGatherIndices` struct | No (fork-specific) |
