#!/usr/bin/env python3
"""
Profiling script for SM121 quantized matmul kernels.
Designed to be run under ncu or nsys on DGX Spark.

Do NOT use in the automated research loop — ncu injects instrumentation
that makes timing useless. Use this only for targeted investigation after
bench_qmm.py identifies something unexpected (big win or big loss).

Usage:
  # Timeline — shows kernel breakdown, gaps, pipeline stalls (~30s):
  nsys profile -o qmm_profile python3 profile_kernels.py

  # Per-kernel metrics for QMV (M=1, bandwidth bottleneck) (~2min):
  ncu --set full -o qmm_ncu python3 profile_kernels.py --mode qmv

  # Per-kernel metrics for QMM (M=1024, compute bottleneck) (~5min):
  ncu --set full -o qmm_ncu python3 profile_kernels.py --mode qmm

  # Dense baseline for comparison (~2min):
  ncu --set full -o dense_ncu python3 profile_kernels.py --mode dense
"""

import argparse
import mlx.core as mx
import time

# Representative shapes: square + FFN layers (asymmetric N≠K hits different dispatch)
SHAPES = [
    (4096,  4096),   # Llama-7B hidden/attn
    (11008, 4096),   # Llama-7B FFN up/gate
    (4096,  11008),  # Llama-7B FFN down
    (8192,  8192),   # Llama-70B hidden
    (28672, 8192),   # Llama-70B FFN up/gate
]

# (mode_name, group_size, bits)
MODES = [
    ("nvfp4", 16, 4),
    ("mxfp4", 32, 4),
    ("mxfp8", 32, 8),
]


def make_quantized(N, K, mode, group_size, bits):
    """Create quantized weight matrices using mx.quantize."""
    w_float = mx.random.normal((N, K)).astype(mx.float16)
    mx.eval(w_float)
    result = mx.quantize(w_float, group_size=group_size, bits=bits, mode=mode)
    if len(result) == 3:
        w_q, scales, biases = result
    else:
        w_q, scales = result
        biases = None
    mx.eval(w_q, scales)
    if biases is not None:
        mx.eval(biases)
    return w_q, scales, biases


def profile_qmv(shapes, modes, warmup=3, iters=10):
    """Profile QMV (M=1 decode) kernels — bandwidth bound."""
    print("=== QMV Profiling (M=1, bandwidth-bound) ===")
    print(f"{'Mode':<8} {'N':>6} {'K':>6} {'gs':>4}  {'Time(us)':>10}  {'BW(GB/s)':>10}  {'%DRAM':>7}")
    print("-" * 60)

    for N, K in shapes:
        for mode_name, gs, bits in modes:
            x = mx.random.normal((1, K)).astype(mx.float16)
            w_q, scales, biases = make_quantized(N, K, mode_name, group_size=gs, bits=bits)
            mx.eval(x)

            for _ in range(warmup):
                out = mx.quantized_matmul(
                    x, w_q, scales, biases,
                    transpose=True, group_size=gs, bits=bits, mode=mode_name
                )
                mx.eval(out)

            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out = mx.quantized_matmul(
                    x, w_q, scales, biases,
                    transpose=True, group_size=gs, bits=bits, mode=mode_name
                )
                mx.eval(out)
            mx.synchronize()
            dt = (time.perf_counter() - t0) / iters

            weight_bytes = w_q.nbytes
            scale_bytes  = scales.nbytes
            bias_bytes   = biases.nbytes if biases is not None else 0
            vec_bytes    = K * 2   # fp16 input
            out_bytes    = N * 2   # fp16 output
            total_bytes  = weight_bytes + scale_bytes + bias_bytes + vec_bytes + out_bytes
            bw_gbps = total_bytes / dt / 1e9
            pct_dram = bw_gbps / 273 * 100

            print(f"{mode_name:<8} {N:>6} {K:>6} {gs:>4}  {dt*1e6:>10.1f}  {bw_gbps:>10.1f}  {pct_dram:>6.1f}%")


def profile_qmm(shapes, modes, warmup=3, iters=10):
    """Profile QMM (prefill) kernels at various M — compute bound."""
    print("\n=== QMM Profiling (prefill, compute-bound) ===")
    M_values = [64, 128, 256, 512, 1024, 2048]
    print(f"{'Mode':<8} {'M':>5} {'N':>6} {'K':>6} {'gs':>4}  {'Time(us)':>10}  {'TFLOP/s':>10}")
    print("-" * 60)

    for N, K in shapes:
        for mode_name, gs, bits in modes:
            w_q, scales, biases = make_quantized(N, K, mode_name, group_size=gs, bits=bits)
            for M in M_values:
                x = mx.random.normal((M, K)).astype(mx.float16)
                mx.eval(x)

                for _ in range(warmup):
                    out = mx.quantized_matmul(
                        x, w_q, scales, biases,
                        transpose=True, group_size=gs, bits=bits, mode=mode_name
                    )
                    mx.eval(out)

                mx.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    out = mx.quantized_matmul(
                        x, w_q, scales, biases,
                        transpose=True, group_size=gs, bits=bits, mode=mode_name
                    )
                    mx.eval(out)
                mx.synchronize()
                dt = (time.perf_counter() - t0) / iters

                flops  = 2 * M * N * K
                tflops = flops / dt / 1e12
                print(f"{mode_name:<8} {M:>5} {N:>6} {K:>6} {gs:>4}  {dt*1e6:>10.1f}  {tflops:>10.2f}")


def profile_dense(shapes, warmup=3, iters=10):
    """Profile dense fp16 matmul baseline for comparison."""
    print("\n=== Dense FP16 Baseline ===")
    M_values = [1, 64, 128, 256, 512, 1024, 2048]

    for N, K in shapes:
        for M in M_values:
            x = mx.random.normal((M, K)).astype(mx.float16)
            w = mx.random.normal((N, K)).astype(mx.float16)
            mx.eval(x, w)

            for _ in range(warmup):
                out = x @ w.T
                mx.eval(out)

            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out = x @ w.T
                mx.eval(out)
            mx.synchronize()
            dt = (time.perf_counter() - t0) / iters

            if M == 1:
                total_bytes = N * K * 2 + K * 2 + N * 2
                bw_gbps = total_bytes / dt / 1e9
                print(f"  dense  M={M:>5} N={N:>6} K={K:>6}: {dt*1e6:>8.1f} us  BW={bw_gbps:.1f} GB/s ({bw_gbps/273*100:.0f}%)")
            else:
                flops  = 2 * M * N * K
                tflops = flops / dt / 1e12
                print(f"  dense  M={M:>5} N={N:>6} K={K:>6}: {dt*1e6:>8.1f} us  {tflops:.2f} TFLOP/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["qmv", "qmm", "dense", "all"],
                        help="Which kernels to profile")
    parser.add_argument("--iters",  type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    print(f"MLX: {mx.__version__}  Device: {mx.default_device()}")
    print(f"Shapes: {SHAPES}")
    print(f"Modes:  {MODES}")
    print()

    if args.mode in ("qmv", "all"):
        profile_qmv(SHAPES, MODES, args.warmup, args.iters)

    if args.mode in ("qmm", "all"):
        profile_qmm(SHAPES, MODES, args.warmup, args.iters)

    if args.mode in ("dense", "all"):
        profile_dense(SHAPES, args.warmup, args.iters)


if __name__ == "__main__":
    main()
