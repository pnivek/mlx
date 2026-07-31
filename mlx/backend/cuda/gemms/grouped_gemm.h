// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/dtype.h"

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

class array;

void cutlass_grouped_gemm_unaligned(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int group_count,
    const array& a,
    const array& b,
    const array& indices,
    array& out,
    cu::CommandEncoder& encoder);

void cutlass_segmented_mm(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int num_segments,
    int M,
    int N,
    const array& a,
    const array& b,
    const array& segments,
    array& out,
    cu::CommandEncoder& encoder);

// Lower-level grouped GEMM: takes pre-computed GPU args (problem sizes,
// leading dimensions, and data pointers already on device).
// problem_sizes_gpu must point to group_count × {int, int, int} structs
// matching the layout of cutlass::gemm::GemmCoord.
void cutlass_grouped_gemm_run(
    bool a_transposed,
    bool b_transposed,
    int group_count,
    void* problem_sizes_gpu,
    int64_t* a_lds_gpu,
    int64_t* b_lds_gpu,
    int64_t* out_lds_gpu,
    void** a_ptrs_gpu,
    void** b_ptrs_gpu,
    void** out_ptrs_gpu,
    Dtype dtype,
    int N,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
