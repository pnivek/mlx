// Copyright © 2026 pnivek
// E003-grouped: CUTLASS SM120 grouped block-scaled GEMM for MoE.
//
// Single-launch grouped GEMM that handles all experts in one kernel launch,
// eliminating the per-expert loop and host sync for expert counts.
// Based on FlashInfer's SM120 grouped GEMM approach.
//
// Key features:
// - Device-side argument computation (no host sync for expert shapes)
// - PDL (Programmatic Dependent Launch) for zero-sync kernel chaining
// - CUTLASS GemmUniversalMode::kGrouped with PtrArray schedule
// - Native FP4/FP8 block-scaled tensor cores (no dequantization)

#include "mlx/backend/cuda/quantized/gather_qmm.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS includes for grouped GEMM
#if defined(__CUDACC__)

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/arch/arch.h"

#include "cute/tensor.hpp"

// SM120 block-scaled GEMM support check
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED) || \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200)
#define MLX_SM120_GROUPED_GEMM_ENABLED 1
#endif

#endif // __CUDACC__

namespace mlx::core {

#ifdef MLX_SM120_GROUPED_GEMM_ENABLED

using namespace cute;

// ============================================================================
// Device-side kernel to compute per-group GEMM arguments
// ============================================================================

// This kernel runs on GPU to set up problem shapes, strides, and pointers
// for each expert group. Uses GDC barriers for zero-sync chaining with
// the subsequent CUTLASS grouped GEMM kernel.
template <
    int ScaleGranularity,
    typename ScaleConfig,
    typename ElementA,
    typename ElementB,
    typename ElementSFA,
    typename ElementSFB,
    typename ElementD,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideD,
    typename LayoutSFA,
    typename LayoutSFB>
__global__ void compute_grouped_gemm_args(
    ElementA* A,          // gathered activations (sorted by expert)
    ElementB* B,          // expert weights [E, N, K_packed]
    ElementSFA* SFA,      // activation scale factors (sorted)
    ElementSFB* SFB,      // weight scale factors [E, N, K/gs]
    ElementD* D,          // output buffer (sorted order)
    int* m_indptr,        // expert offsets: [0, count_e0, count_e0+count_e1, ...]
    int N,
    int K,
    int num_groups,
    ProblemShape* problem_sizes,
    const ElementA** A_ptr,
    const ElementB** B_ptr,
    const ElementSFA** SFA_ptr,
    const ElementSFB** SFB_ptr,
    ElementD** D_ptr,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideD* stride_D,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_groups) return;

  // GDC: wait for sort kernel to complete
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  constexpr size_t alignment_mn = 128;
  constexpr size_t alignment_k = static_cast<size_t>(ScaleGranularity) * 4;

  size_t sf_n = (static_cast<size_t>(N) + alignment_mn - 1) / alignment_mn * alignment_mn;
  size_t swizzled_k = (static_cast<size_t>(K) + alignment_k - 1) / alignment_k * alignment_k;
  size_t sf_k = swizzled_k / static_cast<size_t>(ScaleGranularity);

  int m_offset = m_indptr[i];
  int m_next = m_indptr[i + 1];
  size_t M = static_cast<size_t>(m_next) - static_cast<size_t>(m_offset);

  // SF M offset aligned for block-scaled layout
  size_t sf_m_offset =
      (static_cast<size_t>(m_offset) + static_cast<size_t>(i) * (alignment_mn - 1)) /
      alignment_mn * alignment_mn;

  problem_sizes[i] = typename ProblemShape::UnderlyingProblemShape(M, N, K);

  stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(M), K, 1});
  stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(M), N, 1});

  // Pointer offsets for activations and output (sorted by expert)
  A_ptr[i] = safe_ptr_offset(A, static_cast<size_t>(m_offset) * K);
  D_ptr[i] = safe_ptr_offset(D, static_cast<size_t>(m_offset) * N);

  // Weight pointer: each expert is at offset i in the first dimension
  B_ptr[i] = safe_ptr_offset(B, static_cast<size_t>(i) * N * K);

  // Scale factor layouts
  auto problem_shape_for_sf = make_shape(static_cast<int>(M), N, K, 1);
  layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(problem_shape_for_sf);
  layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(problem_shape_for_sf);

  SFA_ptr[i] = safe_ptr_offset(SFA, sf_m_offset * sf_k);
  SFB_ptr[i] = safe_ptr_offset(SFB, static_cast<size_t>(i) * sf_n * sf_k);
}

// Helper for sub-byte pointer arithmetic
template <typename T>
__device__ __forceinline__ const T* safe_ptr_offset(const T* ptr, size_t offset) {
  constexpr int adj = (cutlass::sizeof_bits<T>::value < 8) ? (8 / cutlass::sizeof_bits<T>::value) : 1;
  return ptr + offset / adj;
}

template <typename T>
__device__ __forceinline__ T* safe_ptr_offset(T* ptr, size_t offset) {
  constexpr int adj = (cutlass::sizeof_bits<T>::value < 8) ? (8 / cutlass::sizeof_bits<T>::value) : 1;
  return ptr + offset / adj;
}

#endif // MLX_SM120_GROUPED_GEMM_ENABLED

// TODO: Implement gather_qmm_grouped_gpu() that:
// 1. Sorts indices on device (sort_gather_indices)
// 2. Computes m_indptr from sorted expert IDs (prefix sum on device)
// 3. Gathers activations + quantizes them in sorted order
// 4. Launches compute_grouped_gemm_args via PDL
// 5. Launches CUTLASS grouped GEMM
// 6. Scatters output back to original order
//
// This replaces the per-expert loop in gather_qmm_sm120_gpu with a
// single CUTLASS grouped GEMM launch.

} // namespace mlx::core
