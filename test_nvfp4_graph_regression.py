"""Test NVFP4 M=1 regression caused by graph cache key collision.

Investigating whether flush_l2_cache() (a 24MB fill kernel) contaminates
the graph cache entry for QMV kernels, since both produce single-kernel
graphs with the same topology key "K-:".
"""
import time
import mlx.core as mx

K, N = 4096, 4096  # Llama-7B shape
M = 1
WARMUP = 20
TRIALS = 10
ITERS = 50
L2_CACHE_BYTES = 24 * 1024 * 1024

def flush_l2_cache():
    """Same as bench_qmm.py — writes 24MB+ to evict L2."""
    buf = mx.zeros((L2_CACHE_BYTES // 2 + 1,), dtype=mx.float16)
    mx.eval(buf)
    del buf

def make_quantized(mode, bits, gs):
    w_fp = mx.random.normal(shape=(N, K)).astype(mx.float16)
    mx.eval(w_fp)
    if mode == "affine":
        result = mx.quantize(w_fp, group_size=gs, bits=bits)
        return result[0], result[1], result[2]
    else:
        result = mx.quantize(w_fp, group_size=gs, bits=bits, mode=mode)
        mx.eval(result[0], result[1])
        return result[0], result[1], None

def bench_qmv(w_q, scales, biases, mode, bits, gs, label=""):
    x = mx.random.normal(shape=(M, K)).astype(mx.float16)
    mx.eval(x)

    def run():
        if mode == "affine":
            y = mx.quantized_matmul(x, w_q, scales, biases, transpose=True,
                                     group_size=gs, bits=bits)
        else:
            y = mx.quantized_matmul(x, w_q, scales, transpose=True,
                                     group_size=gs, bits=bits, mode=mode)
        mx.eval(y)

    for _ in range(WARMUP):
        run()

    trial_times = []
    for _ in range(TRIALS):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            run()
        mx.synchronize()
        trial_times.append((time.perf_counter() - t0) / ITERS)

    trial_times.sort()
    median_ms = trial_times[len(trial_times) // 2] * 1000
    print(f"  {label:40s} {median_ms:8.3f} ms")
    return median_ms


print(f"Shape: M={M}, K={K}, N={N}")
print(f"Methodology: median of {TRIALS} trials x {ITERS} iters, {WARMUP} warmup")
print()

# Test 1: NVFP4 alone (no L2 flush)
print("Test 1: NVFP4 M=1 alone")
w_q, scales, _ = make_quantized("nvfp4", 4, 16)
t1 = bench_qmv(w_q, scales, None, "nvfp4", 4, 16, "NVFP4 (no flush)")

# Test 2: L2 flush then NVFP4
print("\nTest 2: L2 flush then NVFP4 M=1")
flush_l2_cache()
w_q, scales, _ = make_quantized("nvfp4", 4, 16)
t2 = bench_qmv(w_q, scales, None, "nvfp4", 4, 16, "NVFP4 (after L2 flush)")

# Test 3: L2 flush then MXFP4 (control)
print("\nTest 3: L2 flush then MXFP4 M=1 (control)")
flush_l2_cache()
w_q, scales, _ = make_quantized("mxfp4", 4, 32)
t3 = bench_qmv(w_q, scales, None, "mxfp4", 4, 32, "MXFP4 (after L2 flush)")

# Test 4: L2 flush then MXFP8 (control)
print("\nTest 4: L2 flush then MXFP8 M=1 (control)")
flush_l2_cache()
w_q, scales, _ = make_quantized("mxfp8", 8, 32)
t4 = bench_qmv(w_q, scales, None, "mxfp8", 8, 32, "MXFP8 (after L2 flush)")

# Test 5: INT4 then flush then NVFP4 (full bench_qmm sequence)
print("\nTest 5: INT4 → L2 flush → NVFP4 (bench_qmm.py pattern)")
w_q_int4, scales_int4, biases_int4 = make_quantized("affine", 4, 64)
bench_qmv(w_q_int4, scales_int4, biases_int4, "affine", 4, 64, "INT4-gs64")
flush_l2_cache()
w_q, scales, _ = make_quantized("nvfp4", 4, 16)
t5 = bench_qmv(w_q, scales, None, "nvfp4", 4, 16, "NVFP4 (after INT4 + flush)")

# Test 6: quantize() then NVFP4 (quantize creates graphs too)
print("\nTest 6: Fresh quantize + NVFP4 (with flush)")
flush_l2_cache()
w_fp = mx.random.normal(shape=(N, K)).astype(mx.float16)
mx.eval(w_fp)
result = mx.quantize(w_fp, group_size=16, bits=4, mode="nvfp4")
w_q, scales = result[0], result[1]
mx.eval(w_q, scales)
t6 = bench_qmv(w_q, scales, None, "nvfp4", 4, 16, "NVFP4 (fresh quantize + flush)")

# Test 7: NVFP4 at end (graph cache warmed from above)
print("\nTest 7: NVFP4 again (graph cache warmed)")
w_q, scales, _ = make_quantized("nvfp4", 4, 16)
t7 = bench_qmv(w_q, scales, None, "nvfp4", 4, 16, "NVFP4 (warmed cache)")

print(f"\nSummary:")
print(f"  NVFP4 alone:           {t1:.3f} ms")
print(f"  NVFP4 after flush:     {t2:.3f} ms  (ratio: {t2/t1:.1f}x)")
print(f"  MXFP4 after flush:     {t3:.3f} ms")
print(f"  MXFP8 after flush:     {t4:.3f} ms")
print(f"  NVFP4 INT4+flush:      {t5:.3f} ms  (ratio: {t5/t1:.1f}x)")
print(f"  NVFP4 quantize+flush:  {t6:.3f} ms  (ratio: {t6/t1:.1f}x)")
print(f"  NVFP4 warmed:          {t7:.3f} ms  (ratio: {t7/t1:.1f}x)")
