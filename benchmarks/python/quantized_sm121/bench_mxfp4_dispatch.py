#!/usr/bin/env python3
"""
MXFP4 vs NVFP4 dispatch investigation.
Profiles at M values where MXFP4 showed erratic performance in bench_qmm.py.

Usage:
    # Quick wall-clock (eval per iteration):
    python bench_mxfp4_dispatch.py

    # nsys kernel-level:
    nsys profile --cuda-graph-trace=node -o /tmp/mxfp4_dispatch python bench_mxfp4_dispatch.py
"""
import time
import mlx.core as mx


def bench(name, fn, warmup=20, iters=100):
    """Benchmark with eval per iteration."""
    for _ in range(warmup):
        mx.eval(fn())

    times = []
    for _ in range(10):  # 10 trials
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        times.append((time.perf_counter() - t0) / iters * 1e6)

    times.sort()
    median = times[len(times) // 2]
    mn = times[0]
    print(f"  {name:20s}: min={mn:.1f}us  median={median:.1f}us")
    return median


print("=" * 70)
print("MXFP4 vs NVFP4 Dispatch Investigation")
print("=" * 70)

# Llama-70B shape (where M=64 MXFP4 showed 0.86x)
K, N = 8192, 8192

# M values: QMV cutoff at M=8, then SM120 GEMM
test_M = [1, 4, 8, 9, 16, 32, 64, 128, 256, 512, 1024]

modes = [
    ("NVFP4",  "nvfp4", 4, 16),
    ("MXFP4",  "mxfp4", 4, 32),
    ("MXFP8",  "mxfp8", 8, 32),
    ("Dense",  None,    16, 0),
]

for M in test_M:
    print(f"\n--- M={M}, K={K}, N={N} ---")

    x = mx.random.normal((M, K)).astype(mx.float16)
    mx.eval(x)

    dense_time = None

    for mode_name, mode, bits, gs in modes:
        try:
            if mode is None:
                # Dense matmul
                w = mx.random.normal((N, K)).astype(mx.float16)
                mx.eval(w)
                fn = lambda w=w, x=x: x @ w.T
                dense_time = bench(mode_name, fn)
            else:
                w_fp = mx.random.normal((N, K)).astype(mx.float16)
                result = mx.quantize(w_fp, group_size=gs, bits=bits, mode=mode)
                if len(result) == 3:
                    w, s, b = result
                    mx.eval(w, s, b)
                    fn = lambda w=w, s=s, b=b, x=x, gs=gs, bits=bits, mode=mode: \
                        mx.quantized_matmul(x, w, s, b, transpose=True,
                                            group_size=gs, bits=bits, mode=mode)
                else:
                    w, s = result
                    mx.eval(w, s)
                    fn = lambda w=w, s=s, x=x, gs=gs, bits=bits, mode=mode: \
                        mx.quantized_matmul(x, w, s, transpose=True,
                                            group_size=gs, bits=bits, mode=mode)
                t = bench(mode_name, fn)
                if dense_time and dense_time > 0:
                    print(f"    -> vs dense: {dense_time/t:.2f}x")
        except Exception as e:
            print(f"  {mode_name:20s}: SKIP ({e})")

print("\n" + "=" * 70)
print("Done.")
