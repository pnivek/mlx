// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/gather_qmm.h"
#include "mlx/backend/cuda/quantized/qmm/cute_qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm_sm120.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

// SM120/SM121 (DGX Spark, Desktop Blackwell) native block-scaled GEMM.
// Covers the FP quantization modes; K must be a multiple of the 128-wide
// MMA tile. MXFP8 beyond M=2048 is faster through the regular chain.
bool supports_qmm_sm120(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (device.compute_capability_major() < 12) {
    return false;
  }
  if (!transpose || mode == QuantizationMode::Affine) {
    return false;
  }
  if (w.ndim() != 2) {
    return false;
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    return false;
  }
  if (x.shape(-1) % 128 != 0) {
    return false;
  }
  int64_t m_total = out.size() / out.shape(-1);
  int64_t weight_elems =
      static_cast<int64_t>(w.shape(-2)) * x.shape(-1);
  if (mode == QuantizationMode::Mxfp8) {
    // Sweep vs upstream sm80 (2026-07): the FP8 SM120 GEMM only wins at
    // large M on large weights (N*K >= 32M separates winners exactly).
    if (m_total < 512 || m_total > 2048 ||
        weight_elems < 32ll * 1024 * 1024) {
      return false;
    }
  } else {
    // FP4: sm80 wins the small-M band; SM120 wins from ~M=64-128 up.
    if (m_total > 8 && m_total < 64) {
      return false;
    }
    // NVFP4 with sub-alignment N: upstream's nvfp4 chain beats the
    // minimal-pad path at every M (MXFP4 odd-N stays ours: 2.3-5.4x wins).
    if (mode == QuantizationMode::Nvfp4 &&
        out.shape(-1) % (16 / out.itemsize()) != 0) {
      return false;
    }
  }
  return true;
}

} // namespace

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  array x = ensure_row_contiguous(inputs[0], encoder, s);
  const array& w = inputs[1];
  const array& scales = inputs[2];
  std::optional<array> biases;
  if (inputs.size() > 3) {
    biases = inputs[3];
  }

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
        encoder.device());
  };
  bool can_use_qmm_sm120 = supports(supports_qmm_sm120);
  bool can_use_qmm_sm90 = supports(supports_qmm_sm90);
  bool can_use_qmm_sm80 = supports(supports_qmm_sm80);
  bool can_use_qmm_naive = supports(supports_qmm_naive);
  bool can_use_fp_qmv = supports(supports_fp_qmv);
  bool can_use_qmv = supports(supports_qmv) || can_use_fp_qmv;

  auto call_qmm_sm120 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    // The SM120 GEMM launches kernels directly on the stream (not as graph
    // nodes). Flush pending graph ops so inputs produced in this eval are
    // materialized before the eager kernels read them, and drain in-flight
    // work: allocations made inside the direct-launch window are not tied to
    // graph completion, so without the drain the pool can hand them memory
    // still referenced by pending kernels (observed as whole-buffer
    // corruption of live inputs under allocation churn).
    encoder.commit();
    encoder.synchronize();
    encoder.begin_direct_launch();
    if (bits_ == 4) {
      cute_qmm_fp4_sm120(x, w, scales, out, bits_, group_size_, encoder);
    } else {
      cute_qmm_fp8_sm120(x, w, scales, out, group_size_, encoder);
    }
    encoder.end_direct_launch();
  };

  auto call_qmm_sm90 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    qmm_sm90(x, w, scales, *biases, out, bits_, group_size_, encoder, s);
  };
  auto call_qmm_sm80 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    qmm_sm80(
        x,
        w,
        scales,
        biases,
        std::nullopt,
        std::nullopt,
        out,
        bits_,
        group_size_,
        mode_,
        encoder);
  };
  auto call_qmm_naive = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    qmm_naive(
        x,
        w,
        scales,
        biases,
        std::nullopt,
        std::nullopt,
        out,
        transpose_,
        bits_,
        group_size_,
        mode_,
        encoder);
  };
  auto call_qmv = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    if (can_use_fp_qmv) {
      fp_qmv(x, w, scales, out, bits_, group_size_, encoder, s);
    } else {
      qmv(x,
          w,
          scales,
          biases,
          std::nullopt,
          out,
          bits_,
          group_size_,
          mode_,
          encoder);
    }
  };

  int M = out.ndim() > 1 ? out.shape(-2) : 1;
  int N = out.shape(-1);
  int K = x.shape(-1);
  int B = out.size() / (M * N);

  if (can_use_qmm_sm120) {
    // Small batches decode faster through QMV (reads weights once per row).
    if (can_use_qmv && (M * B <= 8)) {
      call_qmv();
    } else {
      call_qmm_sm120();
    }
    return;
  }

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

  if (can_use_qmm_naive) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_naive();
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

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherQMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  array x = ensure_row_contiguous(inputs[0], encoder, s);
  const array& w = inputs[1];
  const array& scales = inputs[2];
  std::optional<array> biases;
  if (inputs.size() == 6) {
    biases = inputs[3];
  }
  array lhs_indices =
      ensure_row_contiguous(inputs[inputs.size() - 2], encoder, s);
  array rhs_indices =
      ensure_row_contiguous(inputs[inputs.size() - 1], encoder, s);

  int M = out.ndim() > 1 ? out.shape(-2) : 1;
  int N = out.shape(-1);
  int K = x.shape(-1);
  int B = out.size() / (M * N);

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
        encoder.device());
  };
  bool can_use_qmm_sm80 = supports(supports_qmm_sm80);
  bool can_use_qmm_naive = supports(supports_qmm_naive);
  bool can_use_qmv = supports(supports_qmv);

  // SM120 (DGX Spark / Desktop Blackwell) MoE prefill paths for FP modes.
  // At prefill scale each expert's weights are read once, so either a single
  // CUTLASS grouped block-scaled GEMM (aligned FP4) or the host-sync
  // per-expert dequant+cuBLAS loop beats the row-wise chain below.
  if (encoder.device().compute_capability_major() >= 12 && transpose_ &&
      mode_ != QuantizationMode::Affine && (M * B > 2048) &&
      (x.dtype() == float16 || x.dtype() == bfloat16)) {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    if (bits_ == 4 && (K % 128 == 0)) {
      gather_qmm_grouped_gpu(
          x,
          w,
          scales,
          lhs_indices,
          rhs_indices,
          out,
          group_size_,
          bits_,
          mode_,
          encoder,
          s);
    } else {
      gather_qmm_gpu(
          x,
          w,
          scales,
          biases,
          lhs_indices,
          rhs_indices,
          out,
          transpose_,
          group_size_,
          bits_,
          mode_,
          encoder,
          s);
    }
    return;
  }

  auto call_qmm_sm80 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    qmm_sm80(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        bits_,
        group_size_,
        mode_,
        encoder);
  };
  auto call_qmm_naive = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    qmm_naive(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        bits_,
        group_size_,
        mode_,
        encoder);
  };
  auto call_qmv = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    gather_qmv(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        bits_,
        group_size_,
        mode_,
        encoder);
  };

  if (can_use_qmm_sm80) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_sm80();
    }
    return;
  }

  if (can_use_qmm_naive) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_naive();
    }
    return;
  }

  if (can_use_qmv) {
    call_qmv();
    return;
  }

  throw std::runtime_error(
      fmt::format(
          "[gather_qmm] No implementation for "
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
  auto& enc = cu::get_command_encoder(s);
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
