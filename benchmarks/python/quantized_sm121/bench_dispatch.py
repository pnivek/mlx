#!/usr/bin/env python3
"""
Dispatch investigation benchmark — targeted A/B testing at dispatch boundaries.

Use this when bench_qmm.py shows unexpected behavior at specific M values,
e.g. MXFP4 performing worse than NVFP4 at M=64 but better at M=16.
This probes the QMV↔GEMM transition threshold and mode-specific crossovers.

Also useful as an nsys target to see which kernel actually fires at each M.

Usage:
    python bench_dispatch.py                      # run with default shapes
    python bench_dispatch.py --K 8192 --N 8192    # specific shape
    nsys profile --cuda-graph-trace=node -o /tmp/dispatch python bench_dispatch.py
"""
import argparse
import time
import mlx.core as mx


def bench(name, fn, warmup=20, trials=10, iters=100):
    """Benchmark with eval per iteration. Returns (min_us, median_us)."""
    for _ in range(warmup):
        mx.eval(fn())

    times = []
    for _ in range(trials):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        times.append((time.perf_counter() - t0) / iters * 1e6)

    times.sort()
    return times[0], times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8192, help="K dim (input features)")
    parser.add_argument("--N", type=int, default=8192, help="N dim (output features)")
    args = parser.parse_args()

    K, N = args.K, args.N

    # Probe M values around the QMV↔GEMM transition (typically M=8 or M=16)
    # and other interesting crossover points
    test_M = [1, 2, 4, 6, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]

    modes = [
        ("NVFP4",  "nvfp4", 4, 16),
        ("MXFP4",  "mxfp4", 4, 32),
        ("MXFP8",  "mxfp8", 8, 32),
        ("Dense",  None,    16, 0),
    ]

    print(f"Dispatch Investigation: K={K}, N={N}")
    print(f"{'M':>5}  {'Mode':<8}  {'min(us)':>9}  {'med(us)':>9}  {'vs Dense':>9}")
    print("-" * 50)

    for M in test_M:
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(x)

        dense_min = None

        for mode_name, mode, bits, gs in modes:
            try:
                if mode is None:
                    w = mx.random.normal((N, K)).astype(mx.float16)
                    mx.eval(w)
                    fn = lambda w=w, x=x: x @ w.T
                    mn, md = bench(mode_name, fn)
                    dense_min = mn
                    print(f"{M:>5}  {mode_name:<8}  {mn:>9.1f}  {md:>9.1f}  {'(ref)':>9}")
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
                        w, s = result[0], result[1]
                        mx.eval(w, s)
                        fn = lambda w=w, s=s, x=x, gs=gs, bits=bits, mode=mode: \
                            mx.quantized_matmul(x, w, s, transpose=True,
                                                group_size=gs, bits=bits, mode=mode)
                    mn, md = bench(mode_name, fn)
                    vs = f"{dense_min/mn:.2f}x" if dense_min else "?"
                    print(f"{M:>5}  {mode_name:<8}  {mn:>9.1f}  {md:>9.1f}  {vs:>9}")
            except Exception as e:
                print(f"{M:>5}  {mode_name:<8}  SKIP: {e}")

        print()

    print("Done. Check for M values where mode ranking flips — that's a dispatch boundary.")
    print("Run under nsys to confirm which kernel actually fires at each M.")


if __name__ == "__main__":
    main()
