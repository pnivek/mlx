// Copyright © 2026 Apple Inc.
//
// SM120 native block-scaled quantized GEMM for GeForce/DGX Spark (SM120/SM121).
// Uses CUTLASS 3.x collective builder with OpClassBlockScaledTensorOp to
// feed packed FP4/FP8 data directly to SM120 tensor cores with hardware
// block scaling — eliminating the entire dequant-to-shared-memory pipeline.
//
// Key insight: SM120's mma.sync.aligned.block_scale instruction requires BOTH
// operands in sub-byte block-scaled format (FP4 or FP8). For quantized matmul
// (fp16 activation × FP4 weight → fp16 output), we quantize activations
// on-the-fly to FP4 with per-block scale factors, then run the native GEMM.
//
// For NVFP4: Uses m16n8k64 MMA with ue4m3 scale factors, SFVecSize=16.
// For MXFP4: Uses m16n8k64 MMA with ue8m0 scale factors, SFVecSize=32.
// For MXFP8: Uses m16n8k32 MMA with ue8m0 scale factors, SFVecSize=32, TileK=128.
//
// References:
//   [1] CUTLASS Example 79a — blackwell_geforce_nvfp4_bf16_gemm
//       https://github.com/NVIDIA/cutlass/tree/main/examples/79a_blackwell_geforce_nvfp4_bf16_gemm
//   [2] Vincent Kaufmann's fp4-cuda-kernel (129 TFLOPS on DGX Spark)
//       https://github.com/VincentKaufmann/fp4-cuda-kernel
//       Key insight: CUTLASS TagToStrideB<ColumnMajor> = Stride<int64_t, _1, int64_t>
//       means K is contiguous — B stored as row-major [N,K], NO transpose needed.
//   [3] NVIDIA Developer Forums: Custom FP4 CUDA Kernel discussion
//       https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600

#include "mlx/backend/cuda/quantized/qmm_sm120.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype_utils.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Must include CUTLASS arch config BEFORE the guard check so the SM120/SM121
// macros are defined when compiling for the right target.
#include "cutlass/arch/config.h"

// SM120 block-scaled GEMM requires CUTLASS SM120 support.
// Guard everything so the file compiles cleanly on older toolkits.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

#include <mutex>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"

#include <fmt/format.h>

// NOTE: We do NOT "using namespace cute;" because cute::Shape conflicts with
// mlx::core::Shape. Instead, use explicit cute:: prefixes.

namespace mlx::core {

// ============================================================================
// CUTLASS 3.x GEMM kernel type for block-scaled SM120 matmul.
//
// IMPORTANT: This template MUST be at named-namespace scope (not anonymous
// namespace) so that device_kernel<GemmKernel> gets external linkage.
// Anonymous namespace causes internal linkage, which prevents the CUDA
// runtime from properly registering the kernel function pointer in shared
// libraries (cudaFuncGetAttributes returns error 400).
//
// Following Example 79a: BOTH A and B are block-scaled types.
// ElementQuant: Block-scaled quantized type wrapper
//               (nv_float4_t<float_e2m1_t> for NVFP4,
//                mx_float4_t<float_e2m1_t> for MXFP4,
//                mx_float8_t<float_e4m3_t> for MXFP8)
// ElementOut:   Output type (bfloat16_t or half_t)
// ============================================================================
template <
    typename ElementQuant,
    typename ElementOut,
    int AlignA,
    int AlignB,
    typename TileShape,
    typename KernelScheduleTag = cutlass::gemm::collective::KernelScheduleAuto>
struct Sm120BlockScaledGemm {
  // Both A (activation) and B (weight) use the same block-scaled type.
  using ElementA = ElementQuant;
  using ElementB = ElementQuant;
  // SM120 block-scaled GEMM requires TN layout (RowMajor A, ColumnMajor B).
  // IMPORTANT: Despite the "ColumnMajor" name, TagToStrideB<ColumnMajor>
  // produces Stride<int64_t, _1, int64_t> — K has stride 1 (contiguous).
  // make_packed_stride gives stride_B = (K, 1, 0), meaning B(n,k) = ptr[n*K + k].
  // This IS row-major [N,K] storage, which matches how MLX stores packed FP4
  // weights as (N, K/2) row-major. NO transpose needed.
  using LayoutATag = cutlass::layout::RowMajor;      // A: (M, K) row-major
  using LayoutBTag = cutlass::layout::ColumnMajor;    // B: (N, K) K-contiguous (TN)
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  static constexpr int AlignC =
      128 / cutlass::sizeof_bits<ElementOut>::value;
  static constexpr int AlignD = AlignC;

  // No multicast TMA on GeForce/SM121 — cluster must be 1×1×1.
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  // Epilogue: accumulator -> output conversion.
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          ElementOut, LayoutCTag, AlignC,
          ElementOut, LayoutDTag, AlignD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // Mainloop: handles TMA loads, scale factor routing, MMA pipeline.
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          ElementA, LayoutATag, AlignA,
          ElementB, LayoutBTag, AlignB,
          ElementAccumulator,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelScheduleTag>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, // ProblemShape: M, N, K, L
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Scale factor layout helpers.
  using Sm1xxBlkScaledConfig =
      typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
};


// ============================================================================
// MLX wrapper kernel for SM120 block-scaled GEMM.
//
// We define our own __global__ kernel wrapper instead of using
// cutlass::device_kernel<> so we can control launch bounds and pass params
// via __grid_constant__. The file is compiled with
// --device-entity-has-hidden-visibility=false (set in CMakeLists.txt) so that
// cudaFuncSetAttribute() can resolve the kernel function pointer for shared
// memory opt-in (>48KB). See run_sm120_gemm() for the launch logic.
// ============================================================================
template <typename GemmKernel>
__global__
#ifdef __CUDACC__
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmKernel::MinBlocksPerMultiprocessor)
#endif
void sm120_gemm_kernel(CUTLASS_GRID_CONSTANT typename GemmKernel::Params const params) {
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}

namespace {

// ============================================================================
// Cache for reformatted weight scale factors (CUTLASS interleaved layout).
//
// Weight SFs are reformatted from row-major to CUTLASS TMA layout by
// reformat_sf_kernel. This layout depends only on (N, K, group_size), NOT on M.
// Since weights are static during inference, we cache the reformatted SFs
// and skip the reformat kernel (~64 µs FP4, ~33 µs FP8) on all subsequent calls.
//
// Key: (raw_scales_ptr, N, K, group_size) — uniquely identifies a weight config.
// MLX quantized weights are persistent; their device pointers are stable.
// ============================================================================
struct SFBCacheKey {
  const void* scales_ptr;
  int N, K, group_size;
  bool operator==(const SFBCacheKey& o) const {
    return scales_ptr == o.scales_ptr && N == o.N && K == o.K &&
        group_size == o.group_size;
  }
};

struct SFBCacheKeyHash {
  size_t operator()(const SFBCacheKey& k) const {
    size_t h = std::hash<const void*>{}(k.scales_ptr);
    h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct SFBCacheEntry {
  void* device_ptr;
  size_t size_bytes;
};

static std::unordered_map<SFBCacheKey, SFBCacheEntry, SFBCacheKeyHash> sfb_cache;
static std::mutex sfb_cache_mutex;

// ============================================================================
// Scale factor reformatting kernel.
//
// Copies weight scale factors from MLX format (SFType, row-major) to
// CUTLASS interleaved layout for TMA. MLX stores FP scale factors as
// raw uint8 values in their native encoding (ue8m0 for MXFP4, ue4m3
// for NVFP4), which matches CUTLASS's SFType binary representation.
// ============================================================================
template <typename SFType, typename LayoutSF>
__global__ void reformat_sf_kernel(
    const SFType* __restrict__ src,  // (rows, num_groups) row-major, native SF
    SFType* __restrict__ dst,        // CUTLASS-format output buffer
    LayoutSF layout_sf,
    int rows,
    int num_groups,
    int sf_vec_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * num_groups;
  if (idx >= total) return;

  int row = idx / num_groups;
  int g = idx % num_groups;

  // Read scale factor in native encoding (no conversion needed).
  SFType scale_val = src[row * num_groups + g];

  // Write to CUTLASS interleaved layout.
  // layout_sf maps (N/M, K, L) where K is in full-K space.
  // Within each group of sf_vec_size K elements, the stride-0 dimension
  // means they share one scale factor. We address at g * sf_vec_size.
  auto offset = layout_sf(row, g * sf_vec_size, 0);
  dst[offset] = scale_val;
}

// Returns a device pointer to reformatted weight SFs in CUTLASS layout.
// On first call for a given (scales, N, K, gs), launches reformat_sf_kernel
// and caches the result. Subsequent calls return the cached pointer directly,
// eliminating the ~64 µs (FP4) / ~33 µs (FP8) reformat overhead.
template <typename GemmType>
static void* get_or_reformat_sfb(
    const void* raw_scales,
    int N,
    int K,
    int group_size,
    cudaStream_t stream) {
  using BlkConfig = typename GemmType::Sm1xxBlkScaledConfig;
  using LayoutSFB = typename GemmType::LayoutSFB;
  using ElementA = typename GemmType::ElementA;
  using SFType = typename ElementA::ScaleFactorType;

  SFBCacheKey key{raw_scales, N, K, group_size};

  {
    std::lock_guard<std::mutex> lock(sfb_cache_mutex);
    auto it = sfb_cache.find(key);
    if (it != sfb_cache.end()) {
      return it->second.device_ptr;
    }
  }

  // Cache miss — compute layout, allocate, reformat.
  int sf_vec_size = BlkConfig::SFVecSize;
  int num_wt_groups = N * (K / sf_vec_size);
  // M=1 placeholder: layout_SFB depends only on (N, K, BlkConfig), not M.
  auto problem_shape = cute::make_shape(1, N, K, 1);
  LayoutSFB layout_SFB = BlkConfig::tile_atom_to_shape_SFB(problem_shape);
  int sfb_size = cute::cosize(layout_SFB);
  size_t sfb_bytes = static_cast<size_t>(sfb_size) * sizeof(SFType);

  // Persistent allocation (outlives any single encoder/stream).
  void* sfb_ptr = nullptr;
  cudaError_t err = cudaMalloc(&sfb_ptr, sfb_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "[qmm_sm120] cudaMalloc failed for SFB cache entry");
  }

  constexpr int kThreads = 256;
  int wt_groups_per_row = K / group_size;
  int blocks = (num_wt_groups + kThreads - 1) / kThreads;
  reformat_sf_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const SFType*>(raw_scales),
      reinterpret_cast<SFType*>(sfb_ptr),
      layout_SFB, N, wt_groups_per_row, sf_vec_size);

  // Sync so the buffer is ready before caching (other streams may use it).
  cudaStreamSynchronize(stream);

  {
    std::lock_guard<std::mutex> lock(sfb_cache_mutex);
    auto it = sfb_cache.find(key);
    if (it != sfb_cache.end()) {
      cudaFree(sfb_ptr); // Another thread won the race.
      return it->second.device_ptr;
    }
    sfb_cache[key] = {sfb_ptr, sfb_bytes};
  }

  return sfb_ptr;
}

// ============================================================================
// Activation quantization kernels (warp-cooperative).
//
// Quantizes fp16/bf16 activations to FP4 (e2m1) or FP8 (e4m3) with per-block
// scale factors in CUTLASS interleaved layout.
//
// Design: Each WARP processes one group (sf_vec_size=32) or two groups
// (sf_vec_size=16). Each thread loads and quantizes exactly ONE element.
// Benefits:
//   - Perfectly coalesced reads: warp reads 32 consecutive half values = 64 bytes
//   - Perfectly coalesced writes: warp writes 32 (FP8) or 16 (FP4) bytes
//   - Warp shuffle for amax reduction: O(log n) instead of sequential
//   - Minimal register pressure: 1 value per thread (not 32)
//   - High occupancy: few registers per thread -> more warps per SM
// ============================================================================

// FP4 activation quantization — FP16 input, vectorized.
// Each thread processes ELEMS_PER_THREAD=4 consecutive elements via uint2 load.
// SF_VEC_SIZE: 16 (NVFP4) or 32 (MXFP4). Templated for compile-time optimization.
template <int SF_VEC_SIZE, int ELEMS_PER_THREAD, typename SFType, typename LayoutSF>
__global__ void quantize_activation_fp4_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    SFType* __restrict__ sf_out,
    LayoutSF layout_sfa,
    int M,
    int K) {
  static_assert(ELEMS_PER_THREAD == 4, "Only ELEMS_PER_THREAD=4 supported");
  constexpr int THREADS_PER_GROUP = SF_VEC_SIZE / ELEMS_PER_THREAD;
  constexpr int GROUPS_PER_WARP = 32 / THREADS_PER_GROUP;

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_tid / 32;
  int lane = global_tid % 32;
  int sub_group = lane / THREADS_PER_GROUP;
  int local_lane = lane % THREADS_PER_GROUP;

  int num_groups = K / SF_VEC_SIZE;
  int total_groups = M * num_groups;
  int group_idx = warp_id * GROUPS_PER_WARP + sub_group;

  bool valid = (group_idx < total_groups);
  int safe_idx = valid ? group_idx : (total_groups - 1);
  int m = safe_idx / num_groups;
  int g = safe_idx % num_groups;

  // Vectorized load: 4 consecutive halves (8 bytes) via uint2.
  int base_addr = m * K + g * SF_VEC_SIZE + local_lane * ELEMS_PER_THREAD;
  uint2 loaded = *reinterpret_cast<const uint2*>(&input[base_addr]);
  __half2 h01 = *reinterpret_cast<__half2*>(&loaded.x);
  __half2 h23 = *reinterpret_cast<__half2*>(&loaded.y);
  float v0 = __half2float(__low2half(h01));
  float v1 = __half2float(__high2half(h01));
  float v2 = __half2float(__low2half(h23));
  float v3 = __half2float(__high2half(h23));

  // Local amax across 4 values, then shuffle across THREADS_PER_GROUP.
  float local_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                           fmaxf(fabsf(v2), fabsf(v3)));
  unsigned sub_mask = (THREADS_PER_GROUP == 32) ? 0xffffffffu
      : (((1u << THREADS_PER_GROUP) - 1) << (sub_group * THREADS_PER_GROUP));
  #pragma unroll
  for (int offset = THREADS_PER_GROUP / 2; offset > 0; offset >>= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(sub_mask, local_amax, offset));
  }

  float scale = (local_amax > 0.0f) ? (local_amax / 6.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  if (valid && local_lane == 0) {
    auto sf_offset = layout_sfa(m, g * SF_VEC_SIZE, 0);
    sf_out[sf_offset] = static_cast<SFType>(scale);
  }

  // Quantize 4 values to 4 FP4 nibbles via hardware CVT, pack into 2 bytes.
  // __nv_cvt_float2_to_fp4x2 converts 2 floats to packed FP4x2 in 1 CVT
  // instruction, replacing the 8-way branch chain in quantize_float_to_e2m1().
  if (valid) {
    float2 pair0 = make_float2(v0 * inv_scale, v1 * inv_scale);
    float2 pair1 = make_float2(v2 * inv_scale, v3 * inv_scale);
    int out_base = m * (K / 2) + g * (SF_VEC_SIZE / 2) + local_lane * (ELEMS_PER_THREAD / 2);
    output[out_base]     = __nv_cvt_float2_to_fp4x2(pair0, __NV_E2M1, cudaRoundNearest);
    output[out_base + 1] = __nv_cvt_float2_to_fp4x2(pair1, __NV_E2M1, cudaRoundNearest);
  }
}

// FP4 activation quantization — BF16 input, vectorized.
template <int SF_VEC_SIZE, int ELEMS_PER_THREAD, typename SFType, typename LayoutSF>
__global__ void quantize_activation_fp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    SFType* __restrict__ sf_out,
    LayoutSF layout_sfa,
    int M,
    int K) {
  static_assert(ELEMS_PER_THREAD == 4, "Only ELEMS_PER_THREAD=4 supported");
  constexpr int THREADS_PER_GROUP = SF_VEC_SIZE / ELEMS_PER_THREAD;
  constexpr int GROUPS_PER_WARP = 32 / THREADS_PER_GROUP;

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_tid / 32;
  int lane = global_tid % 32;
  int sub_group = lane / THREADS_PER_GROUP;
  int local_lane = lane % THREADS_PER_GROUP;

  int num_groups = K / SF_VEC_SIZE;
  int total_groups = M * num_groups;
  int group_idx = warp_id * GROUPS_PER_WARP + sub_group;

  bool valid = (group_idx < total_groups);
  int safe_idx = valid ? group_idx : (total_groups - 1);
  int m = safe_idx / num_groups;
  int g = safe_idx % num_groups;

  // Vectorized load: 4 consecutive bf16 values (8 bytes) via uint2.
  int base_addr = m * K + g * SF_VEC_SIZE + local_lane * ELEMS_PER_THREAD;
  uint2 loaded = *reinterpret_cast<const uint2*>(&input[base_addr]);
  __nv_bfloat162 b01 = *reinterpret_cast<__nv_bfloat162*>(&loaded.x);
  __nv_bfloat162 b23 = *reinterpret_cast<__nv_bfloat162*>(&loaded.y);
  float v0 = __bfloat162float(__low2bfloat16(b01));
  float v1 = __bfloat162float(__high2bfloat16(b01));
  float v2 = __bfloat162float(__low2bfloat16(b23));
  float v3 = __bfloat162float(__high2bfloat16(b23));

  // Local amax across 4 values, then shuffle across THREADS_PER_GROUP.
  float local_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                           fmaxf(fabsf(v2), fabsf(v3)));
  unsigned sub_mask = (THREADS_PER_GROUP == 32) ? 0xffffffffu
      : (((1u << THREADS_PER_GROUP) - 1) << (sub_group * THREADS_PER_GROUP));
  #pragma unroll
  for (int offset = THREADS_PER_GROUP / 2; offset > 0; offset >>= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(sub_mask, local_amax, offset));
  }

  float scale = (local_amax > 0.0f) ? (local_amax / 6.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  if (valid && local_lane == 0) {
    auto sf_offset = layout_sfa(m, g * SF_VEC_SIZE, 0);
    sf_out[sf_offset] = static_cast<SFType>(scale);
  }

  // Hardware FP4 quantization (same as FP16 kernel above).
  if (valid) {
    float2 pair0 = make_float2(v0 * inv_scale, v1 * inv_scale);
    float2 pair1 = make_float2(v2 * inv_scale, v3 * inv_scale);
    int out_base = m * (K / 2) + g * (SF_VEC_SIZE / 2) + local_lane * (ELEMS_PER_THREAD / 2);
    output[out_base]     = __nv_cvt_float2_to_fp4x2(pair0, __NV_E2M1, cudaRoundNearest);
    output[out_base + 1] = __nv_cvt_float2_to_fp4x2(pair1, __NV_E2M1, cudaRoundNearest);
  }
}

// ============================================================================
// FP8 (e4m3) activation quantization — vectorized.
//
// Each thread processes ELEMS_PER_THREAD=4 elements via uint2 load.
// sf_vec_size is always 32 for MXFP8 -> THREADS_PER_GROUP = 32/4 = 8,
// GROUPS_PER_WARP = 32/8 = 4.
//
// e4m3 representable range: +/-[2^-9, 448]. Max magnitude = 448.0.
// ============================================================================

// FP8 activation quantization — FP16 input, vectorized.
template <int ELEMS_PER_THREAD, typename SFType, typename LayoutSF>
__global__ void quantize_activation_fp8_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    SFType* __restrict__ sf_out,
    LayoutSF layout_sfa,
    int M,
    int K) {
  static_assert(ELEMS_PER_THREAD == 4, "Only ELEMS_PER_THREAD=4 supported");
  constexpr int SF_VEC_SIZE = 32;
  constexpr int THREADS_PER_GROUP = SF_VEC_SIZE / ELEMS_PER_THREAD;  // 8
  constexpr int GROUPS_PER_WARP = 32 / THREADS_PER_GROUP;            // 4

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_tid / 32;
  int lane = global_tid % 32;
  int sub_group = lane / THREADS_PER_GROUP;
  int local_lane = lane % THREADS_PER_GROUP;

  int num_groups = K / SF_VEC_SIZE;
  int total_groups = M * num_groups;
  int group_idx = warp_id * GROUPS_PER_WARP + sub_group;

  bool valid = (group_idx < total_groups);
  int safe_idx = valid ? group_idx : (total_groups - 1);
  int m = safe_idx / num_groups;
  int g = safe_idx % num_groups;

  // Vectorized load: 4 consecutive halves (8 bytes) via uint2.
  int base_addr = m * K + g * SF_VEC_SIZE + local_lane * ELEMS_PER_THREAD;
  uint2 loaded = *reinterpret_cast<const uint2*>(&input[base_addr]);
  __half2 h01 = *reinterpret_cast<__half2*>(&loaded.x);
  __half2 h23 = *reinterpret_cast<__half2*>(&loaded.y);
  float v0 = __half2float(__low2half(h01));
  float v1 = __half2float(__high2half(h01));
  float v2 = __half2float(__low2half(h23));
  float v3 = __half2float(__high2half(h23));

  // Local amax across 4 values, then shuffle across THREADS_PER_GROUP=8.
  float local_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                           fmaxf(fabsf(v2), fabsf(v3)));
  unsigned sub_mask = (((1u << THREADS_PER_GROUP) - 1) << (sub_group * THREADS_PER_GROUP));
  #pragma unroll
  for (int offset = THREADS_PER_GROUP / 2; offset > 0; offset >>= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(sub_mask, local_amax, offset));
  }

  float scale = (local_amax > 0.0f) ? (local_amax / 448.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  if (valid && local_lane == 0) {
    auto sf_offset = layout_sfa(m, g * SF_VEC_SIZE, 0);
    sf_out[sf_offset] = static_cast<SFType>(scale);
  }

  // Quantize 4 values and write 4 bytes via uint32_t store.
  if (valid) {
    __nv_fp8_e4m3 fp8_0 = __nv_fp8_e4m3(v0 * inv_scale);
    __nv_fp8_e4m3 fp8_1 = __nv_fp8_e4m3(v1 * inv_scale);
    __nv_fp8_e4m3 fp8_2 = __nv_fp8_e4m3(v2 * inv_scale);
    __nv_fp8_e4m3 fp8_3 = __nv_fp8_e4m3(v3 * inv_scale);
    uint32_t packed = static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_0))
                    | (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_1)) << 8)
                    | (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_2)) << 16)
                    | (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_3)) << 24);
    int out_addr = m * K + g * SF_VEC_SIZE + local_lane * ELEMS_PER_THREAD;
    *reinterpret_cast<uint32_t*>(&output[out_addr]) = packed;
  }
}

// FP8 activation quantization — BF16 input, vectorized.
template <int ELEMS_PER_THREAD, typename SFType, typename LayoutSF>
__global__ void quantize_activation_fp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    SFType* __restrict__ sf_out,
    LayoutSF layout_sfa,
    int M,
    int K) {
  static_assert(ELEMS_PER_THREAD == 4, "Only ELEMS_PER_THREAD=4 supported");
  constexpr int SF_VEC_SIZE = 32;
  constexpr int THREADS_PER_GROUP = SF_VEC_SIZE / ELEMS_PER_THREAD;  // 8
  constexpr int GROUPS_PER_WARP = 32 / THREADS_PER_GROUP;            // 4

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_tid / 32;
  int lane = global_tid % 32;
  int sub_group = lane / THREADS_PER_GROUP;
  int local_lane = lane % THREADS_PER_GROUP;

  int num_groups = K / SF_VEC_SIZE;
  int total_groups = M * num_groups;
  int group_idx = warp_id * GROUPS_PER_WARP + sub_group;

  bool valid = (group_idx < total_groups);
  int safe_idx = valid ? group_idx : (total_groups - 1);
  int m = safe_idx / num_groups;
  int g = safe_idx % num_groups;

  // Vectorized load: 4 consecutive bf16 values (8 bytes) via uint2.
  int base_addr = m * K + g * SF_VEC_SIZE + local_lane * ELEMS_PER_THREAD;
  uint2 loaded = *reinterpret_cast<const uint2*>(&input[base_addr]);
  __nv_bfloat162 b01 = *reinterpret_cast<__nv_bfloat162*>(&loaded.x);
  __nv_bfloat162 b23 = *reinterpret_cast<__nv_bfloat162*>(&loaded.y);
  float v0 = __bfloat162float(__low2bfloat16(b01));
  float v1 = __bfloat162float(__high2bfloat16(b01));
  float v2 = __bfloat162float(__low2bfloat16(b23));
  float v3 = __bfloat162float(__high2bfloat16(b23));

  float local_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                           fmaxf(fabsf(v2), fabsf(v3)));
  unsigned sub_mask = (((1u << THREADS_PER_GROUP) - 1) << (sub_group * THREADS_PER_GROUP));
  #pragma unroll
  for (int offset = THREADS_PER_GROUP / 2; offset > 0; offset >>= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(sub_mask, local_amax, offset));
  }

  float scale = (local_amax > 0.0f) ? (local_amax / 448.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  if (valid && local_lane == 0) {
    auto sf_offset = layout_sfa(m, g * SF_VEC_SIZE, 0);
    sf_out[sf_offset] = static_cast<SFType>(scale);
  }

  if (valid) {
    __nv_fp8_e4m3 fp8_0 = __nv_fp8_e4m3(v0 * inv_scale);
    __nv_fp8_e4m3 fp8_1 = __nv_fp8_e4m3(v1 * inv_scale);
    __nv_fp8_e4m3 fp8_2 = __nv_fp8_e4m3(v2 * inv_scale);
    __nv_fp8_e4m3 fp8_3 = __nv_fp8_e4m3(v3 * inv_scale);
    uint32_t packed = static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_0))
                    | (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_1)) << 8)
                    | (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_2)) << 16)
                    | (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8_3)) << 24);
    int out_addr = m * K + g * SF_VEC_SIZE + local_lane * ELEMS_PER_THREAD;
    *reinterpret_cast<uint32_t*>(&output[out_addr]) = packed;
  }
}

} // anonymous namespace (helper __global__ kernels)

// ============================================================================
// Stride construction helper.
//
// CUTLASS 3.x uses typed strides: RowMajor has Stride<IntT, Int<1>, IntT>,
// ColumnMajor has Stride<Int<1>, IntT, IntT>. We detect which element is
// static vs dynamic and set the leading dimension accordingly.
//
// For GemmUniversal with shape (M, N, K, L=1):
//   A (RowMajor, M×K):    stride = (K, 1, 0)   ← K contiguous
//   B (ColumnMajor, N×K): stride = (K, 1, 0)   ← K contiguous (row-major!)
//   D (RowMajor, M×N):    stride = (N, 1, 0)   ← N contiguous
//
// IMPORTANT: This function and run_sm120_gemm/execute_sm120_fp4_gemm MUST
// be in the named mlx::core namespace (not anonymous namespace). When the
// implicit instantiation of sm120_gemm_kernel<GemmKernel> occurs in the
// anonymous namespace, nvcc gives the host-side registration stub internal
// linkage, which prevents cudaFuncGetAttributes/cudaFuncSetAttribute from
// resolving the function pointer in shared libraries.
// ============================================================================
template <typename Stride>
Stride make_packed_stride(int dim0, int dim1) {
  Stride stride{};
  using Elem0 = cute::tuple_element_t<0, Stride>;
  if constexpr (cute::is_static_v<Elem0>) {
    // ColumnMajor-like: first dim is contiguous (Int<1>).
    // Set stride for second dim = dim0 (number of rows).
    cute::get<1>(stride) = dim0;
  } else {
    // RowMajor-like: second dim is contiguous (Int<1>).
    // Set stride for first dim = dim1 (number of columns).
    cute::get<0>(stride) = dim1;
  }
  // Batch stride = 0 for non-batched (L=1).
  // If the stride element is static (Int<0>), it's already zero — skip.
  if constexpr (cute::tuple_size_v<Stride> > 2) {
    using Elem2 = cute::tuple_element_t<2, Stride>;
    if constexpr (!cute::is_static_v<Elem2>) {
      cute::get<2>(stride) = 0;
    }
  }
  return stride;
}

// ============================================================================
// CUTLASS GEMM launch helper.
//
// IMPORTANT: kernel_ptr must be obtained and configured (shared memory opt-in)
// from a NON-TEMPLATE context to avoid nvcc ODR issues. When nvcc compiles
// template functions, it may generate different cute::Int<> types between
// device and host passes (signed vs unsigned), causing the kernel function
// pointer to reference an unregistered instantiation. By taking the kernel
// pointer in the non-template cute_qmm_fp4_sm120() function and passing it
// here as void*, we ensure type consistency.
// ============================================================================
template <typename GemmType>
void run_sm120_gemm(
    void* kernel_ptr,
    int M,
    int N,
    int K,
    const void* a_ptr,
    const void* sfa_ptr,
    const void* b_ptr,
    const void* sfb_ptr,
    void* d_ptr,
    cu::CommandEncoder& encoder) {
  using Gemm = typename GemmType::Gemm;
  using GemmKernel = typename Gemm::GemmKernel;
  using BlkConfig = typename GemmType::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename GemmType::LayoutSFA;
  using LayoutSFB = typename GemmType::LayoutSFB;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using SFTypeA = typename GemmType::ElementA::ScaleFactorType;
  using SFTypeB = typename GemmType::ElementB::ScaleFactorType;

  auto problem_shape = cute::make_shape(M, N, K, 1);

  // Construct strides correctly for each layout:
  //   A (RowMajor, M×K):    stride = (K, 1, 0)  ← K contiguous
  //   B (ColumnMajor, N×K): stride = (K, 1, 0)  ← K contiguous (row-major [N,K])
  //   D (RowMajor, M×N):    stride = (N, 1, 0)  ← N contiguous
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  StrideA stride_A = make_packed_stride<StrideA>(M, K);
  StrideB stride_B = make_packed_stride<StrideB>(N, K);
  StrideD stride_D = make_packed_stride<StrideD>(M, N);

  // Scale factor layouts computed from problem shape.
  LayoutSFA layout_SFA = BlkConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = BlkConfig::tile_atom_to_shape_SFB(problem_shape);

  // Construct GEMM arguments.
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {// Mainloop arguments
       reinterpret_cast<const ElementA*>(a_ptr),
       stride_A,
       reinterpret_cast<const ElementB*>(b_ptr),
       stride_B,
       reinterpret_cast<const SFTypeA*>(sfa_ptr),
       layout_SFA,
       reinterpret_cast<const SFTypeB*>(sfb_ptr),
       layout_SFB},
      {// Epilogue arguments: alpha=1, beta=0 (no bias), C=nullptr, D=output
       {1.0f, 0.0f},
       nullptr,
       stride_D,
       reinterpret_cast<ElementD*>(d_ptr),
       stride_D}};

  // Allocate workspace if needed.
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace_ptr = allocate_workspace(encoder, workspace_size);
  }

  // Validate the problem can be implemented.
  {
    auto status = Gemm().can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] CUTLASS cannot implement: M={} N={} K={} (status={})",
          M, N, K, static_cast<int>(status)));
    }
  }

  // Initialize workspace (barrier init, persistent kernel setup, etc.).
  // This is step 1 of CUTLASS's initialize() — MUST be done before launch.
  {
    auto status = GemmKernel::initialize_workspace(
        arguments, workspace_ptr, encoder.stream(), /*cuda_adapter=*/nullptr);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] CUTLASS initialize_workspace failed: {}",
          static_cast<int>(status)));
    }
  }

  // Construct params with TMA descriptors (all host-side computation).
  // This is step 2 of CUTLASS's initialize().
  auto params = GemmKernel::to_underlying_arguments(arguments, workspace_ptr);

  // Step 3 of CUTLASS's initialize() is cudaFuncSetAttribute(device_kernel<GemmKernel>, ...),
  // which fails due to nvcc ODR issues with device_kernel<> in shared libraries.
  // We SKIP this — our sm120_gemm_kernel<> wrapper was already configured for
  // shared memory opt-in in the get_configured_kernel_*() functions.

  // Get launch configuration from CUTLASS.
  dim3 grid = GemmKernel::get_grid_shape(params);
  dim3 block = GemmKernel::get_block_shape();
  int smem_size = GemmKernel::SharedStorageSize;

  // Launch using our pre-configured kernel wrapper.
  // kernel_ptr was obtained from get_configured_kernel_*() which already called
  // cudaFuncSetAttribute for shared memory opt-in on sm120_gemm_kernel<GemmKernel>.
  void* kernel_args[] = {&params};
  cudaError_t err = cudaLaunchKernel(
      kernel_ptr, grid, block, kernel_args, smem_size, encoder.stream());
  if (err != cudaSuccess) {
    throw std::runtime_error(fmt::format(
        "[qmm_sm120] cudaLaunchKernel failed: {} (grid=({},{},{}), block=({},{},{}), smem={})",
        cudaGetErrorString(err),
        grid.x, grid.y, grid.z, block.x, block.y, block.z, smem_size));
  }
}

// ============================================================================
// Orchestration: quantize activations + reformat weight SFs + run GEMM.
// ============================================================================
template <typename GemmType, typename InputType>
void execute_sm120_fp4_gemm(
    void* kernel_ptr,
    const array& x,      // (M, K) fp16/bf16 activation
    const array& w,      // (N, K/2) packed FP4 weights
    const array& scales, // (N, K/gs) fp16 weight scale factors
    array& out,          // (M, N) output
    int group_size,
    cu::CommandEncoder& encoder) {
  using BlkConfig = typename GemmType::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename GemmType::LayoutSFA;
  using ElementA = typename GemmType::ElementA;
  using SFType = typename ElementA::ScaleFactorType;

  int M = out.shape(-2);
  int N = out.shape(-1);
  int K = x.shape(-1);
  int sf_vec_size = BlkConfig::SFVecSize;
  int num_act_groups = M * (K / sf_vec_size);

  auto problem_shape = cute::make_shape(M, N, K, 1);
  LayoutSFA layout_SFA = BlkConfig::tile_atom_to_shape_SFA(problem_shape);
  int sfa_size = cute::cosize(layout_SFA);

  // Allocate temporary buffers.
  // 1. Quantized activation data: (M, K/2) packed FP4.
  size_t x_q_bytes = static_cast<size_t>(M) * K / 2;
  auto x_q_alloc = cu::malloc_async(x_q_bytes, encoder);
  array x_q_buf(x_q_alloc, {M, K / 2}, uint8);
  encoder.add_temporary(x_q_buf);

  // 2. Activation scale factors in CUTLASS layout.
  size_t sfa_bytes = static_cast<size_t>(sfa_size) * sizeof(SFType);
  auto sfa_alloc = cu::malloc_async(sfa_bytes, encoder);
  array sfa_buf(sfa_alloc, {sfa_size}, uint8);
  encoder.add_temporary(sfa_buf);

  // NOTE: No weight transpose needed! CUTLASS ColumnMajor B with
  // TagToStrideB<ColumnMajor> = Stride<int64_t, _1, int64_t> gives
  // stride_B = (K, 1, 0), meaning B(n,k) = ptr[n*K + k] — K contiguous.
  // MLX stores weights as (N, K/2) row-major (K contiguous), which matches.

  auto& stream = encoder.stream();
  constexpr int kThreads = 256;

  // Step 1: Quantize activations to FP4 (vectorized, 4 elements/thread).
  // Each warp handles more groups than before due to vectorization.
  {
    constexpr int ELEMS_PER_THREAD = 4;
    constexpr int THREADS_PER_GROUP = BlkConfig::SFVecSize / ELEMS_PER_THREAD;
    constexpr int GROUPS_PER_WARP = 32 / THREADS_PER_GROUP;
    int total_warps = (num_act_groups + GROUPS_PER_WARP - 1) / GROUPS_PER_WARP;
    int total_threads = total_warps * 32;
    int blocks = (total_threads + kThreads - 1) / kThreads;
    if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
      quantize_activation_fp4_bf16_kernel<BlkConfig::SFVecSize, ELEMS_PER_THREAD>
          <<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data<void>()),
          reinterpret_cast<uint8_t*>(x_q_buf.data<void>()),
          reinterpret_cast<SFType*>(sfa_buf.data<void>()),
          layout_SFA, M, K);
    } else {
      quantize_activation_fp4_kernel<BlkConfig::SFVecSize, ELEMS_PER_THREAD>
          <<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const __half*>(x.data<void>()),
          reinterpret_cast<uint8_t*>(x_q_buf.data<void>()),
          reinterpret_cast<SFType*>(sfa_buf.data<void>()),
          layout_SFA, M, K);
    }
  }

  // Step 2: Get cached reformatted weight scale factors.
  // layout_SFB depends only on (N, K, BlkConfig), not M — safe to cache.
  void* sfb_ptr = get_or_reformat_sfb<GemmType>(
      scales.data<void>(), N, K, group_size, stream);

  // Step 3: Run CUTLASS block-scaled GEMM.
  // Weight data (w) is passed directly — no transpose needed.
  // CUTLASS ColumnMajor B stride = (K, 1, 0) expects K-contiguous data,
  // which matches MLX's row-major (N, K/2) packed FP4 storage.
  run_sm120_gemm<GemmType>(
      kernel_ptr,
      M, N, K,
      x_q_buf.data<void>(),
      sfa_buf.data<void>(),
      w.data<void>(),
      sfb_ptr,
      out.data<void>(),
      encoder);
}

// ============================================================================
// Execute SM120 FP8 GEMM: quantize activations to FP8, reformat SFs, run GEMM.
//
// Same structure as execute_sm120_fp4_gemm but with FP8:
// - x_q is M*K bytes (1 byte/element, no packing)
// - SFVecSize=32 with ue8m0 scale factors (same as MXFP4)
// ============================================================================
template <typename GemmType, typename InputType>
void execute_sm120_fp8_gemm(
    void* kernel_ptr,
    const array& x,      // (M, K) fp16/bf16 activation
    const array& w,      // (N, K) packed FP8 weights
    const array& scales, // (N, K/gs) weight scale factors
    array& out,          // (M, N) output
    int group_size,
    cu::CommandEncoder& encoder) {
  using BlkConfig = typename GemmType::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename GemmType::LayoutSFA;
  using ElementA = typename GemmType::ElementA;
  using SFType = typename ElementA::ScaleFactorType;

  int M = out.shape(-2);
  int N = out.shape(-1);
  int K = x.shape(-1);
  int sf_vec_size = BlkConfig::SFVecSize;
  int num_act_groups = M * (K / sf_vec_size);

  auto problem_shape = cute::make_shape(M, N, K, 1);
  LayoutSFA layout_SFA = BlkConfig::tile_atom_to_shape_SFA(problem_shape);
  int sfa_size = cute::cosize(layout_SFA);

  // 1. Quantized activation data: (M, K) FP8 — 1 byte per element (NOT M*K/2).
  size_t x_q_bytes = static_cast<size_t>(M) * K;
  auto x_q_alloc = cu::malloc_async(x_q_bytes, encoder);
  array x_q_buf(x_q_alloc, {M, K}, uint8);
  encoder.add_temporary(x_q_buf);

  // 2. Activation scale factors in CUTLASS layout.
  size_t sfa_bytes = static_cast<size_t>(sfa_size) * sizeof(SFType);
  auto sfa_alloc = cu::malloc_async(sfa_bytes, encoder);
  array sfa_buf(sfa_alloc, {sfa_size}, uint8);
  encoder.add_temporary(sfa_buf);

  auto& stream = encoder.stream();
  constexpr int kThreads = 256;

  // Step 1: Quantize activations to FP8 (vectorized, 4 elements/thread).
  // sf_vec_size=32, THREADS_PER_GROUP=8, GROUPS_PER_WARP=4.
  {
    constexpr int ELEMS_PER_THREAD = 4;
    constexpr int SF_VEC_SIZE_FP8 = 32;
    constexpr int THREADS_PER_GROUP = SF_VEC_SIZE_FP8 / ELEMS_PER_THREAD;  // 8
    constexpr int GROUPS_PER_WARP = 32 / THREADS_PER_GROUP;                // 4
    int total_warps = (num_act_groups + GROUPS_PER_WARP - 1) / GROUPS_PER_WARP;
    int total_threads = total_warps * 32;
    int blocks = (total_threads + kThreads - 1) / kThreads;
    if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
      quantize_activation_fp8_bf16_kernel<ELEMS_PER_THREAD>
          <<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data<void>()),
          reinterpret_cast<uint8_t*>(x_q_buf.data<void>()),
          reinterpret_cast<SFType*>(sfa_buf.data<void>()),
          layout_SFA, M, K);
    } else {
      quantize_activation_fp8_kernel<ELEMS_PER_THREAD>
          <<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const __half*>(x.data<void>()),
          reinterpret_cast<uint8_t*>(x_q_buf.data<void>()),
          reinterpret_cast<SFType*>(sfa_buf.data<void>()),
          layout_SFA, M, K);
    }
  }

  // Step 2: Get cached reformatted weight scale factors.
  void* sfb_ptr = get_or_reformat_sfb<GemmType>(
      scales.data<void>(), N, K, group_size, stream);

  // Step 3: Run CUTLASS block-scaled GEMM.
  // FP8 weight data stored as (N, K) row-major — K contiguous, same as FP4.
  run_sm120_gemm<GemmType>(
      kernel_ptr,
      M, N, K,
      x_q_buf.data<void>(),
      sfa_buf.data<void>(),
      w.data<void>(),
      sfb_ptr,
      out.data<void>(),
      encoder);
}

// ============================================================================
// Public API: SM120 native FP4 quantized matmul.
//
// Takes fp16/bf16 activations, packed FP4 weights, and fp16 weight scale
// factors. Internally quantizes activations to FP4, reformats scale factors
// to CUTLASS layout, and runs the native SM120 block-scaled GEMM.
// ============================================================================
// Kernel pointer capture and shared memory opt-in.
//
// CRITICAL: These functions capture the sm120_gemm_kernel<GemmKernel> function
// pointer and configure shared memory opt-in. They MUST NOT be template
// functions, because nvcc generates different cute::Int<> types (signed vs
// unsigned) between device and host compilation passes within template
// contexts. This causes the host-side function pointer to reference an
// unregistered kernel instantiation, making cudaFuncSetAttribute and
// cudaGetFuncBySymbol fail with "invalid device function".
//
// By using concrete (non-template) functions with fully-resolved types,
// both device and host passes agree on the GemmKernel type, and the
// kernel function pointer matches the CUDA-registered device stub.
// ============================================================================

// Tile shape used by all SM120 FP4 GEMM variants.
using Sm120FP4TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;

// Concrete GemmType aliases for the 4 kernel variants.
using NvFP4_BF16_Gemm = Sm120BlockScaledGemm<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t, 32, 32, Sm120FP4TileShape>;
using NvFP4_FP16_Gemm = Sm120BlockScaledGemm<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::half_t, 32, 32, Sm120FP4TileShape>;
using MxFP4_BF16_Gemm = Sm120BlockScaledGemm<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t, 32, 32, Sm120FP4TileShape>;
using MxFP4_FP16_Gemm = Sm120BlockScaledGemm<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>,
    cutlass::half_t, 32, 32, Sm120FP4TileShape>;

// Tile shape for SM120 FP8 GEMM: K=128 (same as FP4).
// Despite FP8 being 2x wider per element, CUTLASS CollectiveBuilder expects
// K=128 for block-scaled TMA layout compatibility (per CUTLASS example 79c).
// The physical data per tile is 128 bytes (vs 64 bytes for FP4).
using Sm120FP8TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;

// MXFP8: mx_float8_t with ue8m0 scale factors, SFVecSize=32.
// AlignA/B=16: 16 FP8 elements × 1 byte = 16 bytes (vs 32 FP4 × 0.5 bytes).
using MxFP8_BF16_Gemm = Sm120BlockScaledGemm<
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::bfloat16_t, 16, 16, Sm120FP8TileShape>;
using MxFP8_FP16_Gemm = Sm120BlockScaledGemm<
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::half_t, 16, 16, Sm120FP8TileShape>;

// Pingpong schedule variants for MXFP8.
// Pingpong uses 4 warps (128 threads) with overlapped producer/consumer phases
// instead of Cooperative's 8 warps (256 threads). Better latency hiding on LPDDR5x.
using MxFP8_BF16_Gemm_PP = Sm120BlockScaledGemm<
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::bfloat16_t, 16, 16, Sm120FP8TileShape,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using MxFP8_FP16_Gemm_PP = Sm120BlockScaledGemm<
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::half_t, 16, 16, Sm120FP8TileShape,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;

// Pingpong schedule variants for NVFP4.
using NvFP4_BF16_Gemm_PP = Sm120BlockScaledGemm<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t, 32, 32, Sm120FP4TileShape,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>;
using NvFP4_FP16_Gemm_PP = Sm120BlockScaledGemm<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::half_t, 32, 32, Sm120FP4TileShape,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>;

// Pingpong schedule variants for MXFP4.
using MxFP4_BF16_Gemm_PP = Sm120BlockScaledGemm<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t, 32, 32, Sm120FP4TileShape,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf4Sm120>;
using MxFP4_FP16_Gemm_PP = Sm120BlockScaledGemm<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>,
    cutlass::half_t, 32, 32, Sm120FP4TileShape,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf4Sm120>;

// NOTE: Tile shape sweep results (2025-03):
// - 64×128×128: FAILS — CUTLASS TMA scale factor layout requires BM >= 128
// - 128×64×128: FAILS — CUTLASS TMA scale factor layout requires BN >= 128
// Minimum tile for SM120 block-scaled GEMM is 128×128×K (all scale factor types).
// The only remaining avenue for improving wave utilization at small M is SplitK
// or Stream-K tile schedulers.

// Non-template helper: configure kernel and return void* pointer.
// Each variant is a separate non-template function to ensure consistent
// type resolution between nvcc's device and host compilation passes.
static void* get_configured_kernel_nvfp4_bf16() {
  using GemmKernel = NvFP4_BF16_Gemm::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for nvfp4_bf16: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_nvfp4_fp16() {
  using GemmKernel = NvFP4_FP16_Gemm::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for nvfp4_fp16: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp4_bf16() {
  using GemmKernel = MxFP4_BF16_Gemm::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp4_bf16: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp4_fp16() {
  using GemmKernel = MxFP4_FP16_Gemm::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp4_fp16: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp8_bf16() {
  using GemmKernel = MxFP8_BF16_Gemm::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp8_bf16: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp8_fp16() {
  using GemmKernel = MxFP8_FP16_Gemm::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp8_fp16: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

// Pingpong schedule kernel configurators.
static void* get_configured_kernel_mxfp8_bf16_pp() {
  using GemmKernel = MxFP8_BF16_Gemm_PP::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp8_bf16_pp: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp8_fp16_pp() {
  using GemmKernel = MxFP8_FP16_Gemm_PP::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp8_fp16_pp: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_nvfp4_bf16_pp() {
  using GemmKernel = NvFP4_BF16_Gemm_PP::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for nvfp4_bf16_pp: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_nvfp4_fp16_pp() {
  using GemmKernel = NvFP4_FP16_Gemm_PP::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for nvfp4_fp16_pp: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp4_bf16_pp() {
  using GemmKernel = MxFP4_BF16_Gemm_PP::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp4_bf16_pp: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

static void* get_configured_kernel_mxfp4_fp16_pp() {
  using GemmKernel = MxFP4_FP16_Gemm_PP::Gemm::GemmKernel;
  void* ptr = (void*)sm120_gemm_kernel<GemmKernel>;
  int smem = GemmKernel::SharedStorageSize;
  if (smem >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
      throw std::runtime_error(fmt::format(
          "[qmm_sm120] cudaFuncSetAttribute failed for mxfp4_fp16_pp: {} (smem={}B)",
          cudaGetErrorString(err), smem));
    }
  }
  return ptr;
}

// Helper: dispatch to the correct execute_sm120_fp4_gemm variant based on
// group_size and output dtype.
//
// NOTE: StreamK scheduler was tested (2025-03) and provides <6% GEMM-level
// improvement on DGX Spark (LPDDR5x bandwidth-limited). Additionally, StreamK
// workspace allocation breaks CUDA graph replay (NaN output). Not worth the
// complexity; Pingpong schedule is used for all M values.
static void dispatch_sm120_fp4(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    cu::CommandEncoder& encoder) {
  const char* tag = "[qmm_fp4_sm120]";

  if (group_size == 16) {
    // NVFP4: nv_float4_t with ue4m3 scale factors, SFVecSize=16.
    if (out.dtype() == bfloat16) {
      void* kernel_ptr = get_configured_kernel_nvfp4_bf16_pp();
      execute_sm120_fp4_gemm<NvFP4_BF16_Gemm_PP, __nv_bfloat16>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else if (out.dtype() == float16) {
      void* kernel_ptr = get_configured_kernel_nvfp4_fp16_pp();
      execute_sm120_fp4_gemm<NvFP4_FP16_Gemm_PP, __half>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else {
      throw std::runtime_error(
          fmt::format("{} Unsupported dtype for SM120 NVFP4 GEMM.", tag));
    }
  } else if (group_size == 32) {
    // MXFP4: mx_float4_t with ue8m0 scale factors, SFVecSize=32.
    if (out.dtype() == bfloat16) {
      void* kernel_ptr = get_configured_kernel_mxfp4_bf16_pp();
      execute_sm120_fp4_gemm<MxFP4_BF16_Gemm_PP, __nv_bfloat16>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else if (out.dtype() == float16) {
      void* kernel_ptr = get_configured_kernel_mxfp4_fp16_pp();
      execute_sm120_fp4_gemm<MxFP4_FP16_Gemm_PP, __half>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else {
      throw std::runtime_error(
          fmt::format("{} Unsupported dtype for SM120 MXFP4 GEMM.", tag));
    }
  } else {
    throw std::runtime_error(fmt::format(
        "{} Unsupported group_size {} for SM120 FP4 GEMM. "
        "Expected 16 (NVFP4) or 32 (MXFP4).",
        tag, group_size));
  }
}

void cute_qmm_fp4_sm120(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder) {
  const char* tag = "[qmm_fp4_sm120]";
  int M = out.shape(-2);
  int N = out.shape(-1);
  int K = x.shape(-1);

  if (bits != 4) {
    throw std::runtime_error(
        fmt::format("{} Only 4-bit quantization supported.", tag));
  }

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_output_array(out);

  // SM120 TMA requires N % 128 == 0 and K % 128 == 0.
  // When N is not 128-aligned (e.g. DSv3 N=1407), pad weights, scales, and
  // output to the next 128 multiple, run GEMM, then extract valid columns.
  // Zero-padded weight rows produce zero GEMM contributions.
  int N_padded = (N + 127) / 128 * 128;
  bool needs_n_pad = (N_padded != N);

  if (needs_n_pad) {
    auto& stream = encoder.stream();
    size_t elem_size = size_of(out.dtype());

    // Padded weight buffer: (N_padded, K/2) — first N rows from w, rest zero.
    int w_cols = w.shape(-1);
    array w_pad({N_padded, w_cols}, w.dtype(), nullptr, {});
    w_pad.set_data(cu::malloc_async(w_pad.nbytes(), encoder));
    encoder.add_temporary(w_pad);
    cudaMemsetAsync(w_pad.data<void>(), 0, w_pad.nbytes(), stream);
    cudaMemcpyAsync(
        w_pad.data<void>(), w.data<void>(), w.nbytes(),
        cudaMemcpyDeviceToDevice, stream);

    // Padded scale buffer: (N_padded, K/gs) — first N rows from scales, rest zero.
    int s_cols = scales.shape(-1);
    array s_pad({N_padded, s_cols}, scales.dtype(), nullptr, {});
    s_pad.set_data(cu::malloc_async(s_pad.nbytes(), encoder));
    encoder.add_temporary(s_pad);
    cudaMemsetAsync(s_pad.data<void>(), 0, s_pad.nbytes(), stream);
    cudaMemcpyAsync(
        s_pad.data<void>(), scales.data<void>(), scales.nbytes(),
        cudaMemcpyDeviceToDevice, stream);

    // Padded output: (M, N_padded).
    array out_pad({M, N_padded}, out.dtype(), nullptr, {});
    out_pad.set_data(cu::malloc_async(out_pad.nbytes(), encoder));
    encoder.add_temporary(out_pad);

    // Run GEMM with padded dimensions.
    dispatch_sm120_fp4(x, w_pad, s_pad, out_pad, group_size, encoder);

    // Extract first N columns from each row of padded output.
    cudaMemcpy2DAsync(
        out.data<void>(),           // dst
        N * elem_size,              // dst pitch (row stride in bytes)
        out_pad.data<void>(),       // src
        N_padded * elem_size,       // src pitch
        N * elem_size,              // width to copy per row
        M,                          // number of rows
        cudaMemcpyDeviceToDevice,
        stream);
    return;
  }

  // Normal aligned path — no padding needed.
  dispatch_sm120_fp4(x, w, scales, out, group_size, encoder);
}

// ============================================================================
// Public API: SM120 native FP8 quantized matmul (MXFP8).
//
// Uses mx_float8_t<float_e4m3_t> with ue8m0 scale factors, SFVecSize=32.
// TileShape=128x128x128 (same K as FP4, per CUTLASS example 79c).
// Activations are quantized on-the-fly to FP8 with per-block scale factors.
// ============================================================================

// Helper: dispatch to the correct execute_sm120_fp8_gemm variant.
// No small-tile variant for FP8 — CUTLASS TMA scale factor layout requires BM >= 128.
static void dispatch_sm120_fp8(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    cu::CommandEncoder& encoder) {
  const char* tag = "[qmm_fp8_sm120]";

  if (out.dtype() == bfloat16) {
    void* kernel_ptr = get_configured_kernel_mxfp8_bf16_pp();
    execute_sm120_fp8_gemm<MxFP8_BF16_Gemm_PP, __nv_bfloat16>(
        kernel_ptr, x, w, scales, out, group_size, encoder);
  } else if (out.dtype() == float16) {
    void* kernel_ptr = get_configured_kernel_mxfp8_fp16_pp();
    execute_sm120_fp8_gemm<MxFP8_FP16_Gemm_PP, __half>(
        kernel_ptr, x, w, scales, out, group_size, encoder);
  } else {
    throw std::runtime_error(
        fmt::format("{} Unsupported dtype for SM120 MXFP8 GEMM.", tag));
  }
}

void cute_qmm_fp8_sm120(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    cu::CommandEncoder& encoder) {
  const char* tag = "[qmm_fp8_sm120]";
  int M = out.shape(-2);
  int N = out.shape(-1);
  int K = x.shape(-1);

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_output_array(out);

  // N-padding for TMA alignment (same pattern as FP4).
  int N_padded = (N + 127) / 128 * 128;
  bool needs_n_pad = (N_padded != N);

  if (needs_n_pad) {
    auto& stream = encoder.stream();
    size_t elem_size = size_of(out.dtype());

    // FP8 weights: (N, K) — 1 byte per element.
    int w_cols = w.shape(-1);
    array w_pad({N_padded, w_cols}, w.dtype(), nullptr, {});
    w_pad.set_data(cu::malloc_async(w_pad.nbytes(), encoder));
    encoder.add_temporary(w_pad);
    cudaMemsetAsync(w_pad.data<void>(), 0, w_pad.nbytes(), stream);
    cudaMemcpyAsync(
        w_pad.data<void>(), w.data<void>(), w.nbytes(),
        cudaMemcpyDeviceToDevice, stream);

    int s_cols = scales.shape(-1);
    array s_pad({N_padded, s_cols}, scales.dtype(), nullptr, {});
    s_pad.set_data(cu::malloc_async(s_pad.nbytes(), encoder));
    encoder.add_temporary(s_pad);
    cudaMemsetAsync(s_pad.data<void>(), 0, s_pad.nbytes(), stream);
    cudaMemcpyAsync(
        s_pad.data<void>(), scales.data<void>(), scales.nbytes(),
        cudaMemcpyDeviceToDevice, stream);

    array out_pad({M, N_padded}, out.dtype(), nullptr, {});
    out_pad.set_data(cu::malloc_async(out_pad.nbytes(), encoder));
    encoder.add_temporary(out_pad);

    dispatch_sm120_fp8(x, w_pad, s_pad, out_pad, group_size, encoder);

    cudaMemcpy2DAsync(
        out.data<void>(), N * elem_size,
        out_pad.data<void>(), N_padded * elem_size,
        N * elem_size, M,
        cudaMemcpyDeviceToDevice, stream);
    return;
  }

  dispatch_sm120_fp8(x, w, scales, out, group_size, encoder);
}

void clear_sm120_sf_cache() {
  std::lock_guard<std::mutex> lock(sfb_cache_mutex);
  for (auto& [key, entry] : sfb_cache) {
    cudaFree(entry.device_ptr);
  }
  sfb_cache.clear();
}

} // namespace mlx::core

#else // No SM120 support

namespace mlx::core {

void cute_qmm_fp4_sm120(
    const array&,
    const array&,
    const array&,
    array&,
    int,
    int,
    cu::CommandEncoder&) {
  throw std::runtime_error(
      "[qmm_fp4_sm120] SM120 block-scaled GEMM requires CUDA 13.0+ "
      "compiled with SM120/SM121 target.");
}

void cute_qmm_fp8_sm120(
    const array&,
    const array&,
    const array&,
    array&,
    int,
    cu::CommandEncoder&) {
  throw std::runtime_error(
      "[qmm_fp8_sm120] SM120 block-scaled GEMM requires CUDA 13.0+ "
      "compiled with SM120/SM121 target.");
}

void clear_sm120_sf_cache() {
  // No-op: SM120 not supported on this build.
}

} // namespace mlx::core

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED
