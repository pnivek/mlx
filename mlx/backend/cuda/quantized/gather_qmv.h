// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

// Fused GatherQMV: quantized vector-matrix multiply with on-device index
// lookup. Each threadblock reads lhs_indices/rhs_indices to select which
// x batch and expert to use — no host synchronization needed.
void fp_gather_qmv(
    const array& w,
    const array& scales,
    const array& x,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    int B,
    CommandEncoder& encoder);

} // namespace mlx::core::cu
