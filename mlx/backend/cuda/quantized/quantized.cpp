// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/gather_qmm.h"
#include "mlx/backend/cuda/quantized/gather_qmv.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/quantized/qmm.h"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/common/matmul.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

// Helper: dequantize weights to FP16 and run cuBLAS GEMM.
static void dequant_cublas_gemm(
    const array& x,
    const array& w_dequant,
    array& out,
    int M, int N, int K,
    cu::CommandEncoder& enc) {
  auto [batch_shape, a_batch_strides, b_batch_strides] =
      collapse_batches(x, w_dequant);

  CublasGemm gemm(
      enc.device(),
      out.dtype(),
      /* a_transposed */ false,
      /* a_rows (M) */ static_cast<uint64_t>(M),
      /* a_cols (K) */ static_cast<uint64_t>(K),
      /* lda */ static_cast<int64_t>(K),
      /* b_transposed */ true,
      /* b_rows (K) */ static_cast<uint64_t>(K),
      /* b_cols (N) */ static_cast<uint64_t>(N),
      /* ldb */ static_cast<int64_t>(K),
      /* batch_count */ static_cast<int32_t>(batch_shape.back()),
      /* a_batch_stride */ a_batch_strides.back(),
      /* b_batch_stride */ b_batch_strides.back());

  gemm.run(
      enc,
      out,
      x,
      w_dequant,
      batch_shape,
      a_batch_strides,
      b_batch_strides,
      /* alpha */ 1.0f);
}

// Helper: allocate temporary dequantized weight buffer.
static array alloc_dequant_buffer(
    int N, int K, Dtype dtype,
    cu::CommandEncoder& enc) {
  array w_dequant(
      {static_cast<int>(N), static_cast<int>(K)},
      dtype,
      nullptr,
      {});
  w_dequant.set_data(
      cu::malloc_async(N * K * size_of(dtype), enc));
  enc.add_temporary(w_dequant);
  return w_dequant;
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(cu::malloc_async(out.nbytes(), enc));

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  // CuTe kernel alignment requirements.
  bool cute_aligned = (N % 128 == 0) && (K % 64 == 0);

  // FP quantization modes (MXFP4, NVFP4, MXFP8): use QMV for small M.
  if (transpose_ && M <= 8 && mode_ != QuantizationMode::Affine) {
    fp_qmv(w, scales, x, out, bits_, group_size_, M, N, K, enc);
    return;
  }

  // MXFP4 / NVFP4: CuTe kernel for small M (aligned), dequant+cuBLAS otherwise.
  if (transpose_ && (mode_ == QuantizationMode::Mxfp4 || mode_ == QuantizationMode::Nvfp4)) {
    if (cute_aligned && M <= 64) {
      cute_qmm_fp4(x, w, scales, out, bits_, group_size_, enc);
    } else {
      array w_dequant = alloc_dequant_buffer(N, K, out.dtype(), enc);
      fp_dequantize(w, scales, w_dequant, group_size_, bits_, enc, s);
      dequant_cublas_gemm(x, w_dequant, out, M, N, K, enc);
    }
    return;
  }

  // Affine quantization: CuTe kernel for small M (aligned), dequant+cuBLAS otherwise.
  if (transpose_ && biases && mode_ == QuantizationMode::Affine) {
    if (cute_aligned && M <= 64) {
      cute_qmm(x, w, scales, *biases, out, bits_, group_size_, enc);
    } else {
      array w_dequant = alloc_dequant_buffer(N, K, out.dtype(), enc);
      affine_dequantize(w, scales, *biases, w_dequant, group_size_, bits_, enc, s);
      dequant_cublas_gemm(x, w_dequant, out, M, N, K, enc);
    }
    return;
  }

  throw std::runtime_error("QMM NYI");
}


void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherQMM::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(cu::malloc_async(out.nbytes(), enc));

  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;

  // Inputs layout: x, w, scales, [biases], lhs_indices, rhs_indices
  // Affine mode has biases; FP modes do not.
  if (mode_ == QuantizationMode::Affine) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  int M = x.shape(-2);
  int K = x.shape(-1);
  int N = out.shape(-1);
  int B = lhs_indices.size();

  // Fused on-device path for FP modes during decode (small M, small B).
  // Uses fp_gather_qmv which reads indices on-device — no host sync.
  // B<=64 threshold: decode has B=num_experts_per_tok (e.g. 4), prefill has
  // B=seq_len*num_experts which benefits from batched dequant+cuBLAS.
  if (transpose_ && M <= 8 && B <= 64 && mode_ != QuantizationMode::Affine) {
    // Make indices contiguous (MoE produces broadcast/strided 3D indices).
    array lhs_flat = ensure_row_contiguous(lhs_indices, enc, s);
    array rhs_flat = ensure_row_contiguous(rhs_indices, enc, s);
    cu::fp_gather_qmv(
        w, scales, x, lhs_flat, rhs_flat, out,
        bits_, group_size_, M, N, K, B, enc);
    return;
  }

  // Fallback: host-side grouping path for large M or affine mode.
  gather_qmm_gpu(
      x, w, scales, biases,
      lhs_indices, rhs_indices,
      out, transpose_, group_size_, bits_, mode_,
      enc, s);
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Quantize::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(cu::malloc_async(w.nbytes(), enc));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      fp_dequantize(wq, scales, w, group_size_, bits_, enc, s);
    }
  } else {
    auto w = ensure_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      fp_quantize(w, wq, scales, group_size_, bits_, enc, s);
    }
  }
}

} // namespace mlx::core
