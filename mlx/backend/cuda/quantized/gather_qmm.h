// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

#include <optional>

namespace mlx::core {

enum class QuantizationMode;

// Legacy path: host sync + per-expert cuBLAS GEMM loop.
void gather_qmm_gpu(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    QuantizationMode mode,
    cu::CommandEncoder& enc,
    const Stream& s);

// Fused path: on-device counting sort + batch dequant + CUTLASS grouped GEMM.
// Eliminates the host sync. Requires dequanting all E experts at once.
void gather_qmm_gpu_fused(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    QuantizationMode mode,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
