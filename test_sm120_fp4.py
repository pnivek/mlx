"""
SM120 FP4 GEMM correctness test.

Tests the native block-scaled FP4 GEMM path against FP16 dense matmul reference.
Runs on DGX Spark (SM121) only.
"""
import mlx.core as mx
import numpy as np

def cos_sim(a, b):
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    a_flat = a_np.flatten()
    b_flat = b_np.flatten()
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)

def rel_error(a, b):
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    diff = np.abs(a_np - b_np)
    ref = np.abs(b_np)
    mask = ref > 1e-6
    if mask.sum() == 0:
        return 0.0
    return np.mean(diff[mask] / ref[mask])

def test_quantized_matmul(M, N, K, group_size, mode, dtype=mx.float16):
    """Test quantized matmul by quantizing random weights and comparing output."""
    # Create random FP16 weight matrix
    np.random.seed(42)
    w_fp16 = mx.array(np.random.randn(N, K).astype(np.float16) * 0.1)
    x = mx.array(np.random.randn(M, K).astype(np.float16) * 0.1)

    # Reference: dense FP16 matmul
    ref = x @ w_fp16.T
    mx.eval(ref)

    # Quantize weights to FP4
    result = mx.quantize(w_fp16, group_size=group_size, bits=4, mode=mode)
    w_q, scales = result[0], result[1]
    mx.eval(w_q, scales)

    # Quantized matmul (will dispatch to SM120 if available and K%128==0)
    out = mx.quantized_matmul(
        x, w_q, scales=scales, transpose=True,
        group_size=group_size, bits=4, mode=mode
    )
    mx.eval(out)

    # Also test with dequantized reference
    w_deq = mx.dequantize(w_q, scales, group_size=group_size, bits=4, mode=mode)
    ref_deq = x @ w_deq.T
    mx.eval(ref_deq)

    cs = cos_sim(out, ref_deq)
    re = rel_error(out, ref_deq)
    has_nan = bool(mx.any(mx.isnan(out)))
    has_inf = bool(mx.any(mx.isinf(out)))

    return cs, re, has_nan, has_inf

def main():
    print("=" * 80)
    print("  SM120 FP4 GEMM Correctness Test")
    print("=" * 80)

    N = 8192  # Fixed N

    # Test configurations
    configs = [
        # (K, group_size, mode_str, mode)
        (8192, 32, "MXFP4", "mxfp4"),
        (8192, 16, "NVFP4", "nvfp4"),
        (2880, 32, "MXFP4-2880", "mxfp4"),  # K%128!=0 (CuTe dequant path)
    ]

    M_values = [1, 4, 8, 16, 32, 64, 128, 256, 512]

    for K, gs, label, mode in configs:
        print(f"\n{label} (K={K}, gs={gs}):")
        print(f"{'M':>6s}  {'cos_sim':>10s}  {'rel_err':>10s}  {'NaN':>4s}  {'Inf':>4s}  {'Status':>8s}")
        print("-" * 55)

        for M in M_values:
            try:
                cs, re, has_nan, has_inf = test_quantized_matmul(M, N, K, gs, mode)
                status = "PASS" if cs > 0.95 and not has_nan and not has_inf else "FAIL"
                print(f"{M:6d}  {cs:10.6f}  {re:10.6f}  {'Y' if has_nan else 'N':>4s}  {'Y' if has_inf else 'N':>4s}  {status:>8s}")
            except Exception as e:
                print(f"{M:6d}  ERROR: {e}")

    # Also test MXFP8 as regression check
    print(f"\nMXFP8 (K=8192, gs=32) — regression check:")
    print(f"{'M':>6s}  {'cos_sim':>10s}  {'rel_err':>10s}  {'NaN':>4s}  {'Inf':>4s}  {'Status':>8s}")
    print("-" * 55)
    for M in [16, 64, 128, 256]:
        try:
            # Use bits=8 for FP8
            np.random.seed(42)
            w_fp16 = mx.array(np.random.randn(N, 8192).astype(np.float16) * 0.1)
            x = mx.array(np.random.randn(M, 8192).astype(np.float16) * 0.1)

            result = mx.quantize(w_fp16, group_size=32, bits=8, mode="mxfp8")
            w_q, scales = result[0], result[1]
            mx.eval(w_q, scales)

            out = mx.quantized_matmul(
                x, w_q, scales=scales, transpose=True,
                group_size=32, bits=8, mode="mxfp8"
            )
            mx.eval(out)

            w_deq = mx.dequantize(w_q, scales, group_size=32, bits=8, mode="mxfp8")
            ref_deq = x @ w_deq.T
            mx.eval(ref_deq)

            cs = cos_sim(out, ref_deq)
            re = rel_error(out, ref_deq)
            has_nan = bool(mx.any(mx.isnan(out)))
            has_inf = bool(mx.any(mx.isinf(out)))
            status = "PASS" if cs > 0.95 and not has_nan and not has_inf else "FAIL"
            print(f"{M:6d}  {cs:10.6f}  {re:10.6f}  {'Y' if has_nan else 'N':>4s}  {'Y' if has_inf else 'N':>4s}  {status:>8s}")
        except Exception as e:
            print(f"{M:6d}  ERROR: {e}")

if __name__ == "__main__":
    main()
