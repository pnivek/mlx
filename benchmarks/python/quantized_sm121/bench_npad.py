#!/usr/bin/env python3
"""E009: Measure N-padding overhead for non-128-aligned N."""
import mlx.core as mx
import time

WARMUP = 20
ITERS = 50
TRIALS = 10

def bench(M, K, N, mode, gs, bits=4):
    w_fp = mx.random.normal((N, K)).astype(mx.float16); mx.eval(w_fp)
    r = mx.quantize(w_fp, group_size=gs, bits=bits, mode=mode)
    w_q, s = r[0], r[1]; mx.eval(w_q, s)
    x = mx.random.normal((M, K)).astype(mx.float16); mx.eval(x)
    def go():
        y = mx.quantized_matmul(x, w_q, s, transpose=True, group_size=gs, bits=bits, mode=mode)
        mx.eval(y)
    for _ in range(WARMUP): go()
    times = []
    for _ in range(TRIALS):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS): go()
        mx.synchronize()
        times.append((time.perf_counter()-t0)/ITERS*1e6)
    times.sort()
    return times[len(times)//2]

print("N-padding overhead: aligned vs non-aligned N")
print(f"{'Mode':<8} {'M':>5} {'K':>5} {'N':>5} {'Time(us)':>10} {'Notes'}")
print("-" * 60)

K = 7168
for mode, gs in [('nvfp4', 16), ('mxfp4', 32)]:
    for M in [1, 16, 64, 256, 1024]:
        t1 = bench(M, K, 1407, mode, gs)
        t2 = bench(M, K, 1408, mode, gs)
        overhead = (t1 / t2 - 1) * 100
        print(f"{mode:<8} {M:>5} {K:>5} {1407:>5} {t1:>10.1f}  N=1407 (padded to 1536)")
        print(f"{mode:<8} {M:>5} {K:>5} {1408:>5} {t2:>10.1f}  N=1408 (aligned)  overhead: {overhead:+.0f}%")
        print()
