// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/quantized/qmm/cute_qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm_sm120.h"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/common/matmul.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

// Helper: dequantize weights to FP16 and run cuBLAS GEMM.
static void dequant_cublas_gemm(
    const array& x,
    const array& w_dequant,
    array& out,
    int M,
    int N,
    int K,
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
    int N,
    int K,
    Dtype dtype,
    cu::CommandEncoder& enc) {
  array w_dequant(
      {static_cast<int>(N), static_cast<int>(K)}, dtype, nullptr, {});
  w_dequant.set_data(cu::malloc_async(N * K * size_of(dtype), enc));
  enc.add_temporary(w_dequant);
  return w_dequant;
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& enc = cu::get_command_encoder(s);
  auto& d = enc.device();

  out.set_data(cu::malloc_async(out.nbytes(), enc));

  // Make sure the last two dims of x and w, s, b are contiguous.
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() > 3) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  // --- SM120+ dispatch (DGX Spark / GeForce Blackwell) ---
  // These paths use native block-scaled tensor cores and our custom QMV.
  if (d.compute_capability_major() >= 12 && transpose_) {
    // FP quantization modes: QMV for small M.
    if (mode_ != QuantizationMode::Affine) {
      int qmv_threshold = 8;
      if (mode_ == QuantizationMode::Mxfp8) {
        int64_t mat_elems = static_cast<int64_t>(N) * K;
        constexpr int64_t kSmallMatrix = 32LL * 1024 * 1024;
        if (mat_elems < kSmallMatrix) {
          qmv_threshold = 16;
        }
      }
      if (M <= qmv_threshold) {
        cu::fp_qmv(w, scales, x, out, bits_, group_size_, M, N, K, enc);
        return;
      }
    }

    // MXFP4 / NVFP4: SM120 native > CuTe > dequant+cuBLAS.
    if (mode_ == QuantizationMode::Mxfp4 ||
        mode_ == QuantizationMode::Nvfp4) {
      bool cute_aligned = (N % 128 == 0) && (K % 64 == 0);
      if (K % 128 == 0) {
        cute_qmm_fp4_sm120(x, w, scales, out, bits_, group_size_, enc);
      } else if (cute_aligned) {
        cute_qmm_fp4(x, w, scales, out, bits_, group_size_, enc);
      } else {
        array w_dequant = alloc_dequant_buffer(N, K, out.dtype(), enc);
        fp_dequantize(
            w, scales, w_dequant, group_size_, bits_, std::nullopt, enc, s);
        dequant_cublas_gemm(x, w_dequant, out, M, N, K, enc);
      }
      return;
    }

    // MXFP8: SM120 native (M<=2048) > dequant+cuBLAS.
    if (mode_ == QuantizationMode::Mxfp8) {
      if ((K % 128 == 0) && M <= 2048) {
        cute_qmm_fp8_sm120(x, w, scales, out, group_size_, enc);
        return;
      }
      array w_dequant = alloc_dequant_buffer(N, K, out.dtype(), enc);
      fp_dequantize(
          w, scales, w_dequant, group_size_, bits_, std::nullopt, enc, s);
      dequant_cublas_gemm(x, w_dequant, out, M, N, K, enc);
      return;
    }

    // Affine INT4: CuTe (M<=64, aligned) > dequant+cuBLAS.
    if (biases && mode_ == QuantizationMode::Affine) {
      bool cute_aligned = (N % 128 == 0) && (K % 64 == 0);
      if (cute_aligned && M <= 64) {
        cute_qmm(x, w, scales, *biases, out, bits_, group_size_, enc);
      } else {
        array w_dequant = alloc_dequant_buffer(N, K, out.dtype(), enc);
        affine_dequantize(
            w, scales, *biases, w_dequant, group_size_, bits_, enc, s);
        dequant_cublas_gemm(x, w_dequant, out, M, N, K, enc);
      }
      return;
    }
  }

  // --- SM90 dispatch (Hopper) ---
  auto supports = [&](auto&& f) {
    return f(
        x,
        w,
        scales,
        biases,
        out,
        transpose_,
        bits_,
        group_size_,
        mode_,
        d);
  };
  bool can_use_qmm_sm90 = supports(supports_qmm_sm90);
  bool can_use_qmm_sm80 = supports(supports_qmm_sm80);
  bool can_use_fp_qmv = supports(supports_fp_qmv);
  bool can_use_qmv = supports(supports_qmv) || can_use_fp_qmv;

  auto call_qmm_sm90 = [&]() {
    qmm_sm90(x, w, scales, *biases, out, bits_, group_size_, enc, s);
  };
  auto call_qmm_sm80 = [&]() {
    qmm_sm80(x, w, scales, biases, out, bits_, group_size_, mode_, enc);
  };
  auto call_qmv = [&]() {
    if (can_use_fp_qmv) {
      fp_qmv(x, w, scales, out, bits_, group_size_, enc, s);
    } else {
      qmv(x, w, scales, biases, out, bits_, group_size_, mode_, enc);
    }
  };

  int B = out.size() / (M * N);

  if (can_use_qmm_sm90) {
    if (can_use_qmv && (M == 1 && B == 1 && N <= 16384 && K <= 16384)) {
      call_qmv();
    } else {
      call_qmm_sm90();
    }
    return;
  }

  if (can_use_qmm_sm80) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_sm80();
    }
    return;
  }

  if (can_use_qmv) {
    call_qmv();
    return;
  }

  throw std::runtime_error(
      fmt::format(
          "[quantized_matmul] No implementation for "
          "problem shape: {}x{}x{}x{}, transpose: {}, "
          "activation: {}, bits: {}, group size: {}, mode: \"{}\".",
          M,
          N,
          K,
          B,
          transpose_,
          dtype_to_string(x.dtype()),
          bits_,
          group_size_,
          quantization_mode_to_string(mode_)));
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
      // 0 -- xq, 1 -- scales, 2 -- could be global scale for nvfp4
      bool use_global_scale =
          mode_ == QuantizationMode::Nvfp4 && inputs.size() > 2;
      std::optional<array> global_scale =
          use_global_scale ? std::make_optional(inputs[2]) : std::nullopt;
      fp_dequantize(wq, scales, w, group_size_, bits_, global_scale, enc, s);
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
      bool use_global_scale =
          mode_ == QuantizationMode::Nvfp4 && inputs.size() > 1;
      std::optional<array> global_scale =
          use_global_scale ? std::make_optional(inputs[1]) : std::nullopt;
      fp_quantize(w, wq, scales, group_size_, bits_, global_scale, enc, s);
    }
  }
}

} // namespace mlx::core
