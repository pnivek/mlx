// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

#include <optional>

namespace mlx::core {

enum class QuantizationMode;

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

} // namespace mlx::core
