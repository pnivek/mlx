#!/usr/bin/env python3
"""
nsys-friendly benchmark for n_per_thread=8 A/B testing.
Runs each shape+mode 100 times with mx.eval() to force discrete kernel launches.
Profile with: nsys profile --cuda-graph-trace=node -o /tmp/npt_test python bench_npt_nsys.py
Then extract kernel times from the sqlite database.

For quick comparison without nsys, uses wall-clock timing with eval per iteration.
"""
import time
import mlx.core as mx

def bench(name, fn, warmup=20, iters=200):
    """Benchmark with eval per iteration to measure actual kernel time."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        mx.eval(result)

    # Timed run
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
        mx.eval(result)
    mx.synchronize()
    t1 = time.perf_counter()
    avg = (t1 - t0) / iters * 1e6
    print(f"  {name}: {avg:.1f} us/iter (avg of {iters})")
    return avg

print("=" * 70)
print("n_per_thread A/B Benchmark (eval per iteration)")
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
    mx.eval(x)

    for mode_name, mode, bits, gs in modes:
        try:
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
            bench(mode_name, fn)
        except Exception as e:
            print(f"  {mode_name}: SKIP ({e})")

print("\n" + "=" * 70)
print("Done. MXFP8 is the control -- should be unchanged between runs.")
