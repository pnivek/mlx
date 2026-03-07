"""
Profiling script for SM121 quantized matmul kernels.
Designed to be run under ncu or nsys on DGX Spark.

Usage:
  # Timeline (nsys):
  nsys profile -o qmm_profile python3 profile_kernels.py

  # Per-kernel metrics (ncu):
  ncu --set full -o qmm_ncu python3 profile_kernels.py --mode qmv
  ncu --set full -o qmm_ncu python3 profile_kernels.py --mode qmm
"""

import argparse
import mlx.core as mx
import time

def make_quantized(N, K, mode, group_size=None, bits=None):
    """Create quantized weight matrices using mx.quantize."""
    w_float = mx.random.normal((N, K)).astype(mx.bfloat16)
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
    """Profile QMV (M=1 decode) kernels."""
    print("=== QMV Profiling (M=1 bandwidth) ===")
    for N, K in shapes:
        for mode_name, gs, bits in modes:
            x = mx.random.normal((1, K)).astype(mx.bfloat16)
            w_q, scales, biases = make_quantized(N, K, mode_name, group_size=gs, bits=bits)
            mx.eval(x)

            # Warmup
            for _ in range(warmup):
                out = mx.quantized_matmul(
                    x, w_q, scales, biases,
                    transpose=True, group_size=gs, bits=bits, mode=mode_name
                )
                mx.eval(out)

            # Timed iterations
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

            # Compute theoretical bandwidth
            weight_bytes = w_q.nbytes
            scale_bytes = scales.nbytes
            bias_bytes = biases.nbytes if biases is not None else 0
            vec_bytes = K * 2  # bf16 input
            out_bytes = N * 2  # bf16 output
            total_bytes = weight_bytes + scale_bytes + bias_bytes + vec_bytes + out_bytes
            bw_gbps = total_bytes / dt / 1e9

            print(f"  {mode_name:6s} N={N:5d} K={K:5d} gs={gs:3d}: "
                  f"{dt*1e6:8.1f} us  BW={bw_gbps:6.1f} GB/s "
                  f"({bw_gbps/273*100:4.1f}% of 273 GB/s)")

def profile_qmm(shapes, modes, warmup=3, iters=10):
    """Profile QMM (prefill) kernels at various M."""
    print("\n=== QMM Profiling (compute utilization) ===")
    M_values = [64, 128, 256, 512, 1024, 2048]
    for N, K in shapes:
        for mode_name, gs, bits in modes:
            w_q, scales, biases = make_quantized(N, K, mode_name, group_size=gs, bits=bits)
            for M in M_values:
                x = mx.random.normal((M, K)).astype(mx.bfloat16)
                mx.eval(x)

                # Warmup
                for _ in range(warmup):
                    out = mx.quantized_matmul(
                        x, w_q, scales, biases,
                        transpose=True, group_size=gs, bits=bits, mode=mode_name
                    )
                    mx.eval(out)

                # Timed
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

                # Compute TFLOPS (2*M*N*K FLOPs for matmul)
                flops = 2 * M * N * K
                tflops = flops / dt / 1e12
                print(f"  {mode_name:6s} M={M:5d} N={N:5d} K={K:5d} gs={gs:3d}: "
                      f"{dt*1e6:8.1f} us  {tflops:6.2f} TFLOPS")

def profile_dense(shapes, warmup=3, iters=10):
    """Profile dense matmul baseline for comparison."""
    print("\n=== Dense Matmul Baseline ===")
    M_values = [1, 64, 128, 256, 512, 1024, 2048]
    for N, K in shapes:
        for M in M_values:
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            w = mx.random.normal((N, K)).astype(mx.bfloat16)
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
                print(f"  dense  M={M:5d} N={N:5d} K={K:5d}: "
                      f"{dt*1e6:8.1f} us  BW={bw_gbps:6.1f} GB/s")
            else:
                flops = 2 * M * N * K
                tflops = flops / dt / 1e12
                print(f"  dense  M={M:5d} N={N:5d} K={K:5d}: "
                      f"{dt*1e6:8.1f} us  {tflops:6.2f} TFLOPS")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["qmv", "qmm", "dense", "all"],
                        help="Which kernels to profile")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    # Representative LLM shapes
    shapes = [
        (4096, 4096),    # Llama-style hidden
        (11008, 4096),   # Llama-style FFN up/gate
        (4096, 11008),   # Llama-style FFN down
    ]

    # (mode_name, group_size, bits)
    # nvfp4: gs=16, bits=4
    # mxfp4: gs=32, bits=4
    # mxfp8: gs=32, bits=8
    modes = [
        ("nvfp4", 16, 4),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
    ]

    if args.mode in ("qmv", "all"):
        profile_qmv(shapes, modes, args.warmup, args.iters)

    if args.mode in ("qmm", "all"):
        profile_qmm(shapes, modes, args.warmup, args.iters)

    if args.mode in ("dense", "all"):
        profile_dense(shapes, args.warmup, args.iters)

if __name__ == "__main__":
    main()
