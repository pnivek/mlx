// Copyright © 2025 Apple Inc.
// SM120 (DGX Spark / GeForce Blackwell) grouped block-scaled GEMM for MoE.

#pragma once

#include "mlx/backend/cuda/device.h"

#include <optional>

namespace mlx::core {

enum class QuantizationMode;

// On-device counting sort of gather indices by expert ID.
// Returns sorted lhs/rhs indices and a permutation for scattering output.
struct SortedGatherIndices {
  array sorted_lhs;
  array sorted_rhs;
  array sorted_perm;
};

SortedGatherIndices sort_gather_indices(
    const array& lhs_indices,
    const array& rhs_indices,
    int B,
    int E,
    cu::CommandEncoder& enc,
    const Stream& s);

// Scatter output rows from sorted order back to original positions.
void scatter_gather_output(
    const array& src,
    const array& sorted_perm,
    array& dst,
    int B,
    int M,
    int N,
    cu::CommandEncoder& enc);

// Single CUTLASS SM120 grouped block-scaled GEMM launch for all experts.
void gather_qmm_grouped_gpu(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    QuantizationMode mode,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
