// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core {

// SM120 native block-scaled FP4 GEMM (MXFP4/NVFP4).
// Uses SM120 tensor cores directly on packed FP4 data with hardware
// block scaling — no dequantization needed.
void cute_qmm_fp4_sm120(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder);

// SM120 native block-scaled FP8 GEMM (MXFP8).
// Uses SM120 tensor cores directly on packed FP8 data with hardware
// block scaling.
void cute_qmm_fp8_sm120(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    cu::CommandEncoder& encoder);

// Clear cached reformatted weight scale factors.
// Call when unloading a model to free GPU memory.
void clear_sm120_sf_cache();

} // namespace mlx::core
