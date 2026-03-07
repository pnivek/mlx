#!/usr/bin/env python3
"""
Focused benchmark for n_per_thread=8 evaluation.
Tests L2-cached vs DRAM-bound shapes for FP4 and FP8 (control).

Usage:
    python bench_npt.py
"""
import time
import mlx.core as mx

def bench(name, fn, warmup=50, trials=10, iters_per_trial=500):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    times = []
    for _ in range(trials):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters_per_trial):
            fn()
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) / iters_per_trial * 1e6)
    mn = min(times)
    md = sorted(times)[len(times) // 2]
    print(f"  {name}: min={mn:.1f}  median={md:.1f} us/iter")
    return mn

print("=" * 70)
print("n_per_thread=8 Benchmark")
print("=" * 70)

# Test shapes: (N, K) -- M=1 for all
shapes = [
    ("4096x4096 (L2-cached)", 4096, 4096),
    ("5120x5120 (L2-cached)", 5120, 5120),
    ("8192x8192 (DRAM-bound)", 8192, 8192),
    ("14336x4096 (DRAM-bound)", 14336, 4096),
]

# Quant modes to test
modes = [
    ("NVFP4", "nvfp4", 4, 16),
    ("MXFP4", "mxfp4", 4, 32),
    ("MXFP8", "mxfp8", 8, 32),  # control -- should not change
]

for shape_name, N, K in shapes:
    print(f"\n--- {shape_name} ---")
    x = mx.random.normal((1, K)).astype(mx.float16)

    for mode_name, mode, bits, gs in modes:
        try:
            w_fp = mx.random.normal((N, K)).astype(mx.float16)
            result = mx.quantize(w_fp, group_size=gs, bits=bits, mode=mode)
            if len(result) == 3:
                w, s, b = result
                fn = lambda w=w, s=s, b=b, x=x, gs=gs, bits=bits, mode=mode: \
                    mx.quantized_matmul(x, w, s, b, transpose=True,
                                        group_size=gs, bits=bits, mode=mode)
            else:
                w, s = result
                fn = lambda w=w, s=s, x=x, gs=gs, bits=bits, mode=mode: \
                    mx.quantized_matmul(x, w, s, transpose=True,
                                        group_size=gs, bits=bits, mode=mode)
            bench(mode_name, fn)
        except Exception as e:
            print(f"  {mode_name}: SKIP ({e})")

print("\n" + "=" * 70)
print("Done. Compare min times against baseline to evaluate n_per_thread=8.")
print("MXFP8 is the control -- should be unchanged between runs.")
