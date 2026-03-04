// Copyright © 2025 Apple Inc.

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
