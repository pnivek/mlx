// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
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

  if (transpose_ && M <= 8 && mode_ != QuantizationMode::Affine) {
    fp_qmv(w, scales, x, out, bits_, group_size_, M, N, K, enc);
    return;
  }

  if (transpose_ && biases && mode_ == QuantizationMode::Affine) {
    if (M > 64) {
      // Large M (prefill): dequantize INT4→FP16, then cuBLAS GEMM.
      // This avoids redundant per-tile dequantization in the fused kernel.

      // 1. Allocate temporary FP16 weight buffer: w is (N, K_packed), dequant
      //    produces (N, K) in out.dtype() (float16 or bfloat16).
      array w_dequant(
          {static_cast<int>(N), static_cast<int>(K)},
          out.dtype(),
          nullptr,
          {});
      w_dequant.set_data(
          cu::malloc_async(N * K * size_of(out.dtype()), enc));
      enc.add_temporary(w_dequant);

      // 2. Dequantize weights: INT4 packed → FP16
      affine_dequantize(
          w, scales, *biases, w_dequant, group_size_, bits_, enc, s);

      // 3. cuBLAS FP16 GEMM: out = x @ w_dequant^T
      //    x is (M, K) row-major, w_dequant is (N, K) row-major
      //    We need C(M,N) = A(M,K) * B(N,K)^T
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
    } else {
      cute_qmm(x, w, scales, *biases, out, bits_, group_size_, enc);
    }
    return;
  }

  throw std::runtime_error("QMM NYI");
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
