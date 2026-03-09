"""Diagnostic: Test MXFP4 + INT8 affine quantized_matmul correctness on CUDA.
The gpt-oss MXFP4-Q8 model uses:
  - MoE experts: MXFP4 (group_size=32, bits=4)
  - Attention Q/K/V/O: INT8 affine (group_size=64, bits=8)
"""
import mlx.core as mx
import numpy as np
import json
import os
import sys

model_dir = os.path.expanduser("~/.local/share/exo/models/mlx-community--gpt-oss-20b-MXFP4-Q8")
if not os.path.exists(model_dir):
    print(f"Model not found at {model_dir}")
    sys.exit(1)

with open(os.path.join(model_dir, "config.json")) as f:
    config = json.load(f)
print(f"Model: {config.get('model_type')}, hidden_size={config.get('hidden_size')}")

# Use mx.load which handles bfloat16 natively
shard = os.path.join(model_dir, "model-00001-of-00003.safetensors")
print(f"Loading {shard}...")
weights = mx.load(shard)
print(f"Loaded {len(weights)} tensors")

# ============================================================
# Test 1: INT8 Affine Attention (q_proj)
# ============================================================
print("\n" + "="*60)
print("TEST 1: INT8 Affine Quantized Matmul (Attention q_proj)")
print("="*60)

attn_key = "model.layers.0.self_attn.q_proj"
w_attn = weights[f"{attn_key}.weight"]
s_attn = weights[f"{attn_key}.scales"]
b_attn = weights[f"{attn_key}.biases"]

print(f"Weight: shape={w_attn.shape}, dtype={w_attn.dtype}")
print(f"Scales: shape={s_attn.shape}, dtype={s_attn.dtype}")
print(f"Biases: shape={b_attn.shape}, dtype={b_attn.dtype}")

N_attn = w_attn.shape[0]
K_attn = w_attn.shape[1] * 4  # INT8: 4 values per uint32 (8 bits each)
print(f"N={N_attn}, K={K_attn}, group_size=64, bits=8, mode=affine")
print(f"CuTe aligned? N%128={N_attn%128}, K%64={K_attn%64}")

for M in [1, 4, 16, 64]:
    x = mx.random.normal(shape=(M, K_attn)) * 0.01
    x = x.astype(mx.float16)

    # Quantized matmul (affine)
    try:
        qmm_out = mx.quantized_matmul(
            x, w_attn, s_attn, b_attn, transpose=True, group_size=64, bits=8
        )
        mx.eval(qmm_out)
    except Exception as e:
        print(f"  M={M}: quantized_matmul FAILED: {e}")
        continue

    # Dequantized reference
    try:
        w_dense = mx.dequantize(w_attn, s_attn, b_attn, group_size=64, bits=8)
        ref_out = x @ w_dense.T
        mx.eval(ref_out)

        qmm_np = np.array(qmm_out).astype(np.float32)
        ref_np = np.array(ref_out).astype(np.float32)

        rel_err = np.abs(qmm_np - ref_np).sum() / (np.abs(ref_np).sum() + 1e-10)
        cos_sim = np.dot(qmm_np.flatten(), ref_np.flatten()) / (
            np.linalg.norm(qmm_np.flatten()) * np.linalg.norm(ref_np.flatten()) + 1e-10
        )
        has_nan = np.isnan(qmm_np).any()
        has_inf = np.isinf(qmm_np).any()
        print(f"  M={M:3d}: rel_err={rel_err:.6f}, cos_sim={cos_sim:.6f}, "
              f"nan={has_nan}, inf={has_inf}, "
              f"qmm=[{qmm_np.min():.4f},{qmm_np.max():.4f}] "
              f"ref=[{ref_np.min():.4f},{ref_np.max():.4f}]")
    except Exception as e:
        print(f"  M={M}: reference FAILED: {e}")

# ============================================================
# Test 2: MXFP4 MoE Expert (gate_proj)
# ============================================================
print("\n" + "="*60)
print("TEST 2: MXFP4 Quantized Matmul (MoE gate_proj)")
print("="*60)

moe_key = "model.layers.0.mlp.experts.gate_proj"
w_moe = weights[f"{moe_key}.weight"]
s_moe = weights[f"{moe_key}.scales"]

print(f"Weight: shape={w_moe.shape}, dtype={w_moe.dtype}")
print(f"Scales: shape={s_moe.shape}, dtype={s_moe.dtype}")

# MoE experts are batched: shape is (num_experts, N, K_packed)
# Slicing triggers JIT gather kernel which is broken on this env.
# Use synthetic MXFP4 data instead to test the matmul kernel directly.
if w_moe.ndim == 3:
    n_experts = w_moe.shape[0]
    N_moe = w_moe.shape[1]
    K_packed = w_moe.shape[2]
    K_moe = K_packed * 8  # FP4: 8 values per uint32
    print(f"Num experts: {n_experts}, creating synthetic MXFP4 data")
    print(f"  Using same dimensions: N={N_moe}, K={K_moe}")
    # Create synthetic quantized weights (random uint32 + e8m0 scales)
    w_exp0 = mx.random.randint(0, 2**32 - 1, shape=(N_moe, K_packed)).astype(mx.uint32)
    # e8m0 scales: byte values around 127 (scale ~1.0)
    s_exp0 = mx.full((N_moe, K_moe // 32), 127, dtype=mx.uint8)
    mx.eval(w_exp0, s_exp0)
else:
    w_exp0 = w_moe
    s_exp0 = s_moe
    N_moe = w_exp0.shape[0]
    K_moe = w_exp0.shape[1] * 8
print(f"N={N_moe}, K={K_moe}, group_size=32, bits=4, mode=mxfp4")

for M in [1, 4, 16]:
    x = mx.random.normal(shape=(M, K_moe)) * 0.01
    x = x.astype(mx.float16)

    try:
        qmm_out = mx.quantized_matmul(
            x, w_exp0, s_exp0, transpose=True, group_size=32, bits=4, mode="mxfp4"
        )
        mx.eval(qmm_out)
    except Exception as e:
        print(f"  M={M}: quantized_matmul FAILED: {e}")
        continue

    try:
        w_dense = mx.dequantize(w_exp0, s_exp0, group_size=32, bits=4, mode="mxfp4")
        ref_out = x @ w_dense.T
        mx.eval(ref_out)

        qmm_np = np.array(qmm_out).astype(np.float32)
        ref_np = np.array(ref_out).astype(np.float32)

        rel_err = np.abs(qmm_np - ref_np).sum() / (np.abs(ref_np).sum() + 1e-10)
        cos_sim = np.dot(qmm_np.flatten(), ref_np.flatten()) / (
            np.linalg.norm(qmm_np.flatten()) * np.linalg.norm(ref_np.flatten()) + 1e-10
        )
        has_nan = np.isnan(qmm_np).any()
        has_inf = np.isinf(qmm_np).any()
        print(f"  M={M:3d}: rel_err={rel_err:.6f}, cos_sim={cos_sim:.6f}, "
              f"nan={has_nan}, inf={has_inf}, "
              f"qmm=[{qmm_np.min():.4f},{qmm_np.max():.4f}] "
              f"ref=[{ref_np.min():.4f},{ref_np.max():.4f}]")
    except Exception as e:
        print(f"  M={M}: reference FAILED: {e}")

# ============================================================
# Test 3: Error accumulation (MoE gate_proj x5, square matrix)
# ============================================================
print("\n" + "="*60)
print("TEST 3: Error accumulation (MoE gate_proj x5)")
print("="*60)

x = mx.random.normal(shape=(1, K_moe)) * 0.01
x = x.astype(mx.float16)
x_qmm = x
x_ref = x

try:
    w_dense_moe = mx.dequantize(w_exp0, s_exp0, group_size=32, bits=4, mode="mxfp4")
    mx.eval(w_dense_moe)
except Exception as e:
    print(f"  dequantize FAILED: {e}")
    print("\nDiagnostic complete.")
    sys.exit(0)

for i in range(5):
    x_qmm = mx.quantized_matmul(
        x_qmm, w_exp0, s_exp0, transpose=True, group_size=32, bits=4, mode="mxfp4"
    )
    x_ref = x_ref @ w_dense_moe.T
    x_qmm = x_qmm / (mx.abs(x_qmm).max() + 1e-8)
    x_ref = x_ref / (mx.abs(x_ref).max() + 1e-8)
    mx.eval(x_qmm, x_ref)

    qmm_np = np.array(x_qmm).astype(np.float32)
    ref_np = np.array(x_ref).astype(np.float32)
    cos_sim = np.dot(qmm_np.flatten(), ref_np.flatten()) / (
        np.linalg.norm(qmm_np.flatten()) * np.linalg.norm(ref_np.flatten()) + 1e-10
    )
    rel_err = np.abs(qmm_np - ref_np).sum() / (np.abs(ref_np).sum() + 1e-10)
    print(f"  Layer {i+1}: cos_sim={cos_sim:.6f}, rel_err={rel_err:.6f}")

print("\nDiagnostic complete.")
