#!/usr/bin/env python3
"""
Benchmark script for MLX quantized matmul kernels on CUDA (GB10 / DGX Spark).

Methodology: 10 trials of 50 iterations each, reports median.
Each iteration calls mx.eval() to force discrete kernel execution.
DRAM BW% column assumes 273 GB/s sustained (DGX Spark LPDDR5x).

Usage:
    python bench_qmm.py              # run all benchmarks
    python bench_qmm.py --qmm       # only QuantizedMatmul
    python bench_qmm.py --gather     # only GatherQMM
    python bench_qmm.py --sweep      # batch size sweep
"""
import argparse
import time
import sys

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP = 20
TRIALS = 10
ITERS_PER_TRIAL = 50
COOLDOWN = 0.1  # seconds between benchmarks
DRAM_BW_GBS = 273  # DGX Spark sustained DRAM BW (GB/s)

# DeepSeek V3.1 MoE layer dimensions
DSV3_K = 7168
DSV3_N = 1407
DSV3_E = 256
DSV3_TOPK = 8

# Typical LLM hidden sizes
MODELS = {
    "Llama-7B":  {"K": 4096,  "N": 4096},
    "Llama-13B": {"K": 5120,  "N": 5120},
    "Llama-70B": {"K": 8192,  "N": 8192},
    "DSv3-MLP":  {"K": DSV3_K, "N": DSV3_N},
}

QUANT_MODES = [
    {"mode": "affine",  "bits": 4, "group_size": 64,  "label": "INT4-gs64"},
    {"mode": "affine",  "bits": 4, "group_size": 128, "label": "INT4-gs128"},
    {"mode": "affine",  "bits": 8, "group_size": 64,  "label": "INT8-gs64"},
    {"mode": "nvfp4",   "bits": 4, "group_size": 16,  "label": "NVFP4"},
    {"mode": "mxfp4",   "bits": 4, "group_size": 32,  "label": "MXFP4"},
    {"mode": "mxfp8",   "bits": 8, "group_size": 32,  "label": "MXFP8"},
]


def tflops(M, N, K, secs):
    """Compute TFLOP/s for a matmul of shape (M,K) x (K,N)."""
    return 2.0 * M * N * K / secs / 1e12


def gbps(M, N, K, bits, secs):
    """Approximate effective bandwidth: read quantized W + read X + write Y."""
    w_bytes = N * K * bits / 8  # quantized weight
    x_bytes = M * K * 2         # fp16 input
    y_bytes = M * N * 2         # fp16 output
    return (w_bytes + x_bytes + y_bytes) / secs / 1e9


def bench_quantized_matmul(M, K, N, mode_cfg, dtype=mx.float16):
    """Benchmark mx.quantized_matmul for a single (M,K)x(N,K)^T configuration."""
    mode = mode_cfg["mode"]
    bits = mode_cfg["bits"]
    gs = mode_cfg["group_size"]

    # Create random weight and quantize
    w_fp = mx.random.normal(shape=(N, K)).astype(dtype)
    mx.eval(w_fp)

    if mode == "affine":
        w_q, scales, biases = mx.quantize(w_fp, group_size=gs, bits=bits)
        mx.eval(w_q, scales, biases)
    else:
        result = mx.quantize(w_fp, group_size=gs, bits=bits, mode=mode)
        w_q, scales = result[0], result[1]
        biases = None
        mx.eval(w_q, scales)

    x = mx.random.normal(shape=(M, K)).astype(dtype)
    mx.eval(x)

    def run_once():
        if mode == "affine":
            y = mx.quantized_matmul(x, w_q, scales, biases, transpose=True,
                                     group_size=gs, bits=bits)
        else:
            y = mx.quantized_matmul(x, w_q, scales, transpose=True,
                                     group_size=gs, bits=bits, mode=mode)
        mx.eval(y)

    for _ in range(WARMUP):
        run_once()

    trial_times = []
    for _ in range(TRIALS):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS_PER_TRIAL):
            run_once()
        mx.synchronize()
        trial_times.append((time.perf_counter() - t0) / ITERS_PER_TRIAL)

    trial_times.sort()
    elapsed = trial_times[len(trial_times) // 2]  # median
    return elapsed, tflops(M, N, K, elapsed), gbps(M, N, K, bits, elapsed)


def bench_dense_matmul(M, K, N, dtype=mx.float16):
    """Benchmark dense fp16 matmul for speedup reference."""
    x = mx.random.normal(shape=(M, K)).astype(dtype)
    w = mx.random.normal(shape=(N, K)).astype(dtype)
    mx.eval(x, w)

    for _ in range(WARMUP):
        y = x @ w.T
        mx.eval(y)

    trial_times = []
    for _ in range(TRIALS):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS_PER_TRIAL):
            y = x @ w.T
            mx.eval(y)
        mx.synchronize()
        trial_times.append((time.perf_counter() - t0) / ITERS_PER_TRIAL)

    trial_times.sort()
    elapsed = trial_times[len(trial_times) // 2]  # median
    return elapsed, tflops(M, N, K, elapsed)


def bench_gather_qmm(M, K, N, E, topk, mode_cfg, dtype=mx.float16):
    """Benchmark mx.gather_qmm (MoE expert dispatch)."""
    mode = mode_cfg["mode"]
    bits = mode_cfg["bits"]
    gs = mode_cfg["group_size"]

    # Create E expert weights and quantize.
    # Quantize as flat (E*N, K) then reshape to (E, N, ...) to avoid mx.stack
    # which triggers JIT compilation that may fail on some systems.
    w_fp = mx.random.normal(shape=(E * N, K)).astype(dtype)
    mx.eval(w_fp)

    if mode == "affine":
        w_q_flat, s_flat, b_flat = mx.quantize(w_fp, group_size=gs, bits=bits)
        mx.eval(w_q_flat, s_flat, b_flat)
        w_q = w_q_flat.reshape(E, N, -1)
        scales = s_flat.reshape(E, N, -1)
        biases = b_flat.reshape(E, N, -1)
        mx.eval(w_q, scales, biases)
    else:
        w_q_flat, s_flat = mx.quantize(w_fp, group_size=gs, bits=bits, mode=mode)[:2]
        mx.eval(w_q_flat, s_flat)
        w_q = w_q_flat.reshape(E, N, -1)
        scales = s_flat.reshape(E, N, -1)
        biases = None
        mx.eval(w_q, scales)

    # Simulate MoE routing: B tokens, each selecting topk experts
    B = M  # tokens
    x = mx.random.normal(shape=(B, 1, K)).astype(dtype)
    mx.eval(x)

    # lhs_indices: which token (identity for prefill)
    # rhs_indices: which expert
    rhs_idx = mx.random.randint(0, E, shape=(B, topk)).astype(mx.uint32)
    lhs_idx = mx.broadcast_to(mx.arange(B).reshape(-1, 1), (B, topk)).astype(mx.uint32)
    mx.eval(rhs_idx, lhs_idx)

    def run_once():
        if mode == "affine":
            y = mx.gather_qmm(x, w_q, scales, biases,
                               lhs_indices=lhs_idx, rhs_indices=rhs_idx,
                               transpose=True, group_size=gs, bits=bits)
        else:
            y = mx.gather_qmm(x, w_q, scales,
                               lhs_indices=lhs_idx, rhs_indices=rhs_idx,
                               transpose=True, group_size=gs, bits=bits,
                               mode=mode)
        mx.eval(y)

    for _ in range(WARMUP):
        run_once()

    trial_times = []
    for _ in range(TRIALS):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS_PER_TRIAL):
            run_once()
        mx.synchronize()
        trial_times.append((time.perf_counter() - t0) / ITERS_PER_TRIAL)

    trial_times.sort()
    elapsed = trial_times[len(trial_times) // 2]  # median

    effective_flops = 2.0 * B * topk * N * K
    eff_tflops = effective_flops / elapsed / 1e12
    return elapsed, eff_tflops


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def run_qmm_benchmarks():
    """Run QuantizedMatmul benchmarks across model sizes and quant modes."""
    print_header("QuantizedMatmul Benchmarks")

    batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]

    for model_name, dims in MODELS.items():
        K, N = dims["K"], dims["N"]
        print(f"\n--- {model_name} (K={K}, N={N}) ---")
        print(f"{'Mode':<14} {'M':>5}  {'Time(ms)':>10}  {'TFLOP/s':>9}  {'GB/s':>8}  {'%DRAM':>6}  {'vs Dense':>9}")
        print("-" * 76)

        for M in batch_sizes:
            # Dense reference
            dense_t, dense_tflops = bench_dense_matmul(M, K, N)

            for mcfg in QUANT_MODES:
                try:
                    t, tf, bw = bench_quantized_matmul(M, K, N, mcfg)
                    speedup = dense_t / t
                    pct_dram = bw / DRAM_BW_GBS * 100
                    print(f"{mcfg['label']:<14} {M:>5}  {t*1000:>10.3f}  {tf:>9.2f}  {bw:>8.1f}  {pct_dram:>5.0f}%  {speedup:>8.2f}x")
                except Exception as e:
                    print(f"{mcfg['label']:<14} {M:>5}  {'ERROR':>10}  {str(e)[:30]}")

                time.sleep(COOLDOWN)

            print()


def run_gather_qmm_benchmarks():
    """Run GatherQMM benchmarks simulating MoE prefill workloads."""
    print_header("GatherQMM (MoE) Benchmarks")

    K, N, E, topk = DSV3_K, DSV3_N, DSV3_E, DSV3_TOPK

    # Only test modes that are realistic for MoE
    moe_modes = [
        {"mode": "affine", "bits": 4, "group_size": 64,  "label": "INT4-gs64"},
        {"mode": "affine", "bits": 4, "group_size": 128, "label": "INT4-gs128"},
        {"mode": "nvfp4",  "bits": 4, "group_size": 16,  "label": "NVFP4"},
    ]

    # Test token counts typical for prefill
    token_counts = [1, 4, 8, 32, 128, 512, 1024, 2048]

    print(f"\nDeepSeek V3.1 MoE: E={E}, topk={topk}, K={K}, N={N}")
    print(f"{'Mode':<14} {'Tokens':>7}  {'Time(ms)':>10}  {'Eff TFLOP/s':>12}")
    print("-" * 55)

    for mcfg in moe_modes:
        for B in token_counts:
            try:
                t, tf = bench_gather_qmm(B, K, N, E, topk, mcfg)
                print(f"{mcfg['label']:<14} {B:>7}  {t*1000:>10.3f}  {tf:>12.2f}")
            except Exception as e:
                print(f"{mcfg['label']:<14} {B:>7}  {'ERROR':>10}  {str(e)[:30]}")

            time.sleep(COOLDOWN)
        print()


def run_sweep():
    """Sweep batch sizes for a single config to show scaling."""
    print_header("Batch Size Sweep: INT4-gs64, Llama-70B (K=8192, N=8192)")

    K, N = 8192, 8192
    mcfg = {"mode": "affine", "bits": 4, "group_size": 64, "label": "INT4-gs64"}

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"{'M':>6}  {'Quant(ms)':>10}  {'Dense(ms)':>10}  {'TFLOP/s':>9}  {'GB/s':>8}  {'%DRAM':>6}  {'Speedup':>8}")
    print("-" * 72)

    for M in batch_sizes:
        try:
            dense_t, dense_tf = bench_dense_matmul(M, K, N)
            qt, qtf, qbw = bench_quantized_matmul(M, K, N, mcfg)
            speedup = dense_t / qt
            pct_dram = qbw / DRAM_BW_GBS * 100
            print(f"{M:>6}  {qt*1000:>10.3f}  {dense_t*1000:>10.3f}  {qtf:>9.2f}  {qbw:>8.1f}  {pct_dram:>5.0f}%  {speedup:>7.2f}x")
        except Exception as e:
            print(f"{M:>6}  ERROR: {e}")
        time.sleep(COOLDOWN)


def main():
    parser = argparse.ArgumentParser(description="MLX QMM Benchmark")
    parser.add_argument("--qmm", action="store_true", help="Run QuantizedMatmul benchmarks")
    parser.add_argument("--gather", action="store_true", help="Run GatherQMM benchmarks")
    parser.add_argument("--sweep", action="store_true", help="Run batch size sweep")
    args = parser.parse_args()

    # Print system info
    print(f"MLX version: {mx.__version__}")
    print(f"Default device: {mx.default_device()}")
    print(f"Methodology: median of {TRIALS} trials x {ITERS_PER_TRIAL} iters, {WARMUP} warmup")

    # If no flags, run everything
    run_all = not (args.qmm or args.gather or args.sweep)

    if run_all or args.qmm:
        run_qmm_benchmarks()
    if run_all or args.gather:
        run_gather_qmm_benchmarks()
    if run_all or args.sweep:
        run_sweep()


if __name__ == "__main__":
    main()
