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

// Host-sync path: per-expert dequant + cuBLAS GEMM loop.
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


// E003: Device-side sorted path with SM120 native FP4/FP8 GEMM.
void gather_qmm_sm120_gpu(
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

// E003-grouped: Single CUTLASS grouped GEMM launch for all experts at once.
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
