// Copyright © 2026 pnivek
// E003-grouped: CUTLASS SM120 grouped block-scaled GEMM for MoE.
//
// Replaces the per-expert GEMM loop in gather_qmm_sm120_gpu with a single
// CUTLASS GemmUniversalMode::kGrouped launch. All experts processed in parallel.
//
// Design:
// - Small host sync (E+1 ints) for quantization kernel scheduling
// - Per-expert activation quantization (E small kernel launches, stream-pipelined)
// - Single grouped SFB cache: reformat all E experts' SFBs at once
// - PDL (Programmatic Dependent Launch) for zero-sync arg→GEMM chaining
// - N-padding to next 128 multiple (required by CUTLASS TMA alignment)

#include "mlx/backend/cuda/quantized/gather_qmm.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/primitives.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

#include "cutlass/arch/config.h"

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
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/kernel_hardware_info.h"

// NOTE: Do NOT "using namespace cute;" — cute::Shape conflicts with mlx::core::Shape.

namespace mlx::core {

// ============================================================================
// Local kernels (mirror of gather_qmm.cu — needed since those are not in a header)
// ============================================================================

template <typename T>
__device__ void local_vectorized_copy(const T* src, T* dst, size_t n) {
  constexpr int VEC = 16 / sizeof(T);
  bool aligned = (reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
                 (reinterpret_cast<uintptr_t>(dst) % 16 == 0) && (n % VEC == 0);
  if (aligned) {
    size_t v = n / VEC;
    auto* sv = reinterpret_cast<const uint4*>(src);
    auto* dv = reinterpret_cast<uint4*>(dst);
    for (size_t i = threadIdx.x; i < v; i += blockDim.x)
      dv[i] = sv[i];
  } else {
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
      dst[i] = src[i];
  }
}

template <typename T>
__global__ void local_gather_rows_kernel(
    const T* src, T* dst, const uint32_t* indices,
    int num_items, int M, int row_elems) {
  int idx = blockIdx.x;
  if (idx >= num_items) return;
  uint32_t src_batch = indices[idx];
  size_t n = static_cast<size_t>(M) * row_elems;
  local_vectorized_copy(src + src_batch * n, dst + idx * n, n);
}

__global__ void local_zero_u32(uint32_t* buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = 0;
}

__global__ void local_count_experts(const uint32_t* ids, int B, uint32_t* counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B) atomicAdd(&counts[ids[i]], 1);
}

// Scatter with (optional) column truncation: src has src_cols per row, dst has dst_cols.
template <typename T>
__global__ void scatter_rows_truncated(
    const T* src, T* dst, const uint32_t* perm,
    int num_items, int M, int src_cols, int dst_cols) {
  int idx = blockIdx.x;
  if (idx >= num_items) return;
  uint32_t dst_batch = perm[idx];
  for (int m = 0; m < M; m++) {
    const T* sr = src + (size_t)idx * M * src_cols + (size_t)m * src_cols;
    T* dr = dst + (size_t)dst_batch * M * dst_cols + (size_t)m * dst_cols;
    for (int n = threadIdx.x; n < dst_cols; n += blockDim.x)
      dr[n] = sr[n];
  }
}

// ============================================================================
// Device+host helper: construct a packed CUTLASS stride from logical dims.
// Mirrors qmm_sm120.cu's make_packed_stride but with __host__ __device__ so
// it can be called inside __global__ kernels.
// ============================================================================
template <typename Stride>
__host__ __device__ __forceinline__ Stride make_grouped_packed_stride(int dim0, int dim1) {
  Stride stride{};
  using Elem0 = cute::tuple_element_t<0, Stride>;
  if constexpr (cute::is_static_v<Elem0>) {
    // ColumnMajor-like: first element is Int<1>, set second = dim0 (rows).
    cute::get<1>(stride) = dim0;
  } else {
    // RowMajor-like: second element is Int<1>, set first = dim1 (cols).
    cute::get<0>(stride) = dim1;
  }
  // Batch stride (index 2) stays 0 for non-batched.
  return stride;
}

// ============================================================================
// Device helper: safe pointer offset for sub-byte types.
// ============================================================================
template <typename T>
__device__ __forceinline__ const T* safe_ptr_off(const T* ptr, size_t elem_off) {
  constexpr int adj =
      (cutlass::sizeof_bits<T>::value < 8) ? (8 / cutlass::sizeof_bits<T>::value) : 1;
  return ptr + elem_off / adj;
}
template <typename T>
__device__ __forceinline__ T* safe_ptr_off(T* ptr, size_t elem_off) {
  constexpr int adj =
      (cutlass::sizeof_bits<T>::value < 8) ? (8 / cutlass::sizeof_bits<T>::value) : 1;
  return ptr + elem_off / adj;
}

// ============================================================================
// Kernel: compute_grouped_gemm_args (device-side per-group argument setup).
// Uses GDC barriers to chain with the CUTLASS grouped GEMM kernel via PDL.
// ============================================================================
template <
    int ScaleGranularity,
    typename ScaleConfig,
    typename ElementA,    // raw FP4 element (float_e2m1_t)
    typename ElementB,
    typename SFType,      // scale factor type (ue4m3 for NVFP4, ue8m0 for MXFP4)
    typename ElementD,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideD,
    typename LayoutSFA,
    typename LayoutSFB>
__global__ void grouped_gemm_setup_args(
    const ElementA* A,        // [B_total, K/2] quantized activations
    const ElementB* B,        // [E, N_padded, K/2] padded weights
    const SFType* SFA,        // grouped SFA buffer (pre-quantized, zero-init'd)
    const SFType* SFB,        // grouped SFB buffer [E, sfb_per_expert_elems]
    ElementD* D,        // [B_total, N_padded] output
    int* m_indptr,      // [E+1] token offsets per expert
    int N_padded,
    int K,
    int E,
    size_t sfb_per_expert_elems,
    ProblemShape* problem_sizes,
    const ElementA** A_ptr,
    const ElementB** B_ptr,
    const SFType** SFA_ptr,
    const SFType** SFB_ptr,
    ElementD** D_ptr,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideD* stride_D,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E) return;

  // GDC: wait for upstream kernels, then signal GEMM to start.
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  constexpr size_t align_mn = 128;
  constexpr size_t align_k = static_cast<size_t>(ScaleGranularity) * 4;

  size_t sf_n = static_cast<size_t>(N_padded);  // already aligned
  size_t swizzled_k =
      (static_cast<size_t>(K) + align_k - 1) / align_k * align_k;  // K is aligned
  size_t sf_k = swizzled_k / static_cast<size_t>(ScaleGranularity);

  int m_off = m_indptr[i];
  int m_cnt = m_indptr[i + 1] - m_off;

  // SFA row offset for expert i.
  size_t sf_m_off =
      (static_cast<size_t>(m_off) + static_cast<size_t>(i) * (align_mn - 1)) /
      align_mn * align_mn;

  problem_sizes[i] = ProblemShape(m_cnt, N_padded, K);

  stride_A[i] = make_grouped_packed_stride<StrideA>(m_cnt, K);
  stride_B[i] = make_grouped_packed_stride<StrideB>(N_padded, K);
  stride_D[i] = make_grouped_packed_stride<StrideD>(m_cnt, N_padded);

  A_ptr[i] = safe_ptr_off(A, static_cast<size_t>(m_off) * static_cast<size_t>(K));
  B_ptr[i] = safe_ptr_off(
      B, static_cast<size_t>(i) * static_cast<size_t>(N_padded) * static_cast<size_t>(K));
  D_ptr[i] = safe_ptr_off(
      D, static_cast<size_t>(m_off) * static_cast<size_t>(N_padded));

  auto sf_shape = cute::make_shape(m_cnt, (int)sf_n, (int)swizzled_k, 1);
  layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(sf_shape);
  SFA_ptr[i] = safe_ptr_off(SFA, sf_m_off * sf_k);

  layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(sf_shape);
  SFB_ptr[i] = safe_ptr_off(SFB, static_cast<size_t>(i) * sfb_per_expert_elems);
}

// ============================================================================
// Kernel: reformat_all_sfb — reformat all E experts' SFBs in one launch.
// src: [E, N_src, num_groups] row-major  (N_src may be < N_padded → zero-fill)
// dst: [E, sfb_per_expert_elems] CUTLASS interleaved
// ============================================================================
template <typename SFType, typename LayoutSFB>
__global__ void reformat_all_sfb_kernel(
    const SFType* src,
    SFType* dst,
    LayoutSFB layout_per_expert,
    int E, int N_src, int N_padded, int num_groups_per_row,
    int sf_vec_size,
    size_t sfb_per_expert_elems) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(E) * N_padded * num_groups_per_row;
  if (idx >= total) return;

  size_t per_expert_elems_mn = static_cast<size_t>(N_padded) * num_groups_per_row;
  int e = static_cast<int>(idx / per_expert_elems_mn);
  size_t rem = idx % per_expert_elems_mn;
  int row = static_cast<int>(rem / num_groups_per_row);
  int g   = static_cast<int>(rem % num_groups_per_row);

  SFType val = (row < N_src)
      ? src[static_cast<size_t>(e) * N_src * num_groups_per_row + row * num_groups_per_row + g]
      : SFType{};

  auto off = layout_per_expert(row, g * sf_vec_size, 0);
  dst[static_cast<size_t>(e) * sfb_per_expert_elems + off] = val;
}

// ============================================================================
// Activation quantization (FP4) — copied from qmm_sm120.cu.
// We need these in this translation unit so we can call them with per-expert
// layout objects. The kernels are identical; the prefix "grouped_" avoids ODR.
// ============================================================================
template <int SF_VEC_SIZE, int EPT, typename SFType, typename LayoutSF>
__global__ void grouped_quant_fp4_fp16(
    const __half* __restrict__ in,
    uint8_t* __restrict__ out,
    SFType* __restrict__ sf,
    LayoutSF layout,
    int M, int K) {
  constexpr int TPG = SF_VEC_SIZE / EPT;
  constexpr int GPW = 32 / TPG;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int wid = gid / 32, lane = gid % 32;
  int sg = lane / TPG, ll = lane % TPG;
  int ng = K / SF_VEC_SIZE, tg = M * ng;
  int grp = wid * GPW + sg;
  bool v = grp < tg;
  int si = v ? grp : tg - 1;
  int m = si / ng, g = si % ng;
  int ba = m * K + g * SF_VEC_SIZE + ll * EPT;
  uint2 ld = *reinterpret_cast<const uint2*>(&in[ba]);
  __half2 h01 = *reinterpret_cast<__half2*>(&ld.x);
  __half2 h23 = *reinterpret_cast<__half2*>(&ld.y);
  float v0 = __half2float(__low2half(h01)), v1 = __half2float(__high2half(h01));
  float v2 = __half2float(__low2half(h23)), v3 = __half2float(__high2half(h23));
  float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
  unsigned mask = (TPG == 32) ? 0xffffffffu
      : (((1u << TPG) - 1) << (sg * TPG));
  #pragma unroll
  for (int off = TPG / 2; off > 0; off >>= 1)
    am = fmaxf(am, __shfl_xor_sync(mask, am, off));
  float sc = (am > 0.f) ? (am / 6.f) : 1.f;
  float isc = 1.f / sc;
  if (v && ll == 0) sf[layout(m, g * SF_VEC_SIZE, 0)] = static_cast<SFType>(sc);
  if (v) {
    int ob = m * (K / 2) + g * (SF_VEC_SIZE / 2) + ll * (EPT / 2);
    out[ob]   = __nv_cvt_float2_to_fp4x2(make_float2(v0*isc, v1*isc), __NV_E2M1, cudaRoundNearest);
    out[ob+1] = __nv_cvt_float2_to_fp4x2(make_float2(v2*isc, v3*isc), __NV_E2M1, cudaRoundNearest);
  }
}

template <int SF_VEC_SIZE, int EPT, typename SFType, typename LayoutSF>
__global__ void grouped_quant_fp4_bf16(
    const __nv_bfloat16* __restrict__ in,
    uint8_t* __restrict__ out,
    SFType* __restrict__ sf,
    LayoutSF layout,
    int M, int K) {
  constexpr int TPG = SF_VEC_SIZE / EPT;
  constexpr int GPW = 32 / TPG;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int wid = gid / 32, lane = gid % 32;
  int sg = lane / TPG, ll = lane % TPG;
  int ng = K / SF_VEC_SIZE, tg = M * ng;
  int grp = wid * GPW + sg;
  bool v = grp < tg;
  int si = v ? grp : tg - 1;
  int m = si / ng, g = si % ng;
  int ba = m * K + g * SF_VEC_SIZE + ll * EPT;
  uint2 ld = *reinterpret_cast<const uint2*>(&in[ba]);
  __nv_bfloat162 b01 = *reinterpret_cast<__nv_bfloat162*>(&ld.x);
  __nv_bfloat162 b23 = *reinterpret_cast<__nv_bfloat162*>(&ld.y);
  float v0 = __bfloat162float(__low2bfloat16(b01)), v1 = __bfloat162float(__high2bfloat16(b01));
  float v2 = __bfloat162float(__low2bfloat16(b23)), v3 = __bfloat162float(__high2bfloat16(b23));
  float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
  unsigned mask = (TPG == 32) ? 0xffffffffu
      : (((1u << TPG) - 1) << (sg * TPG));
  #pragma unroll
  for (int off = TPG / 2; off > 0; off >>= 1)
    am = fmaxf(am, __shfl_xor_sync(mask, am, off));
  float sc = (am > 0.f) ? (am / 6.f) : 1.f;
  float isc = 1.f / sc;
  if (v && ll == 0) sf[layout(m, g * SF_VEC_SIZE, 0)] = static_cast<SFType>(sc);
  if (v) {
    int ob = m * (K / 2) + g * (SF_VEC_SIZE / 2) + ll * (EPT / 2);
    out[ob]   = __nv_cvt_float2_to_fp4x2(make_float2(v0*isc, v1*isc), __NV_E2M1, cudaRoundNearest);
    out[ob+1] = __nv_cvt_float2_to_fp4x2(make_float2(v2*isc, v3*isc), __NV_E2M1, cudaRoundNearest);
  }
}

// ============================================================================
// Grouped SFB cache. Key: (raw_scales_ptr, E, N_padded, K, group_size).
// ============================================================================
struct GroupedSFBKey {
  const void* ptr; int E, N_padded, K, gs;
  bool operator==(const GroupedSFBKey& o) const {
    return ptr==o.ptr && E==o.E && N_padded==o.N_padded && K==o.K && gs==o.gs;
  }
};
struct GroupedSFBHash {
  size_t operator()(const GroupedSFBKey& k) const {
    size_t h = std::hash<const void*>{}(k.ptr);
    h ^= std::hash<int>{}(k.E) + 0x9e3779b9 + (h<<6) + (h>>2);
    h ^= std::hash<int>{}(k.N_padded) + 0x9e3779b9 + (h<<6) + (h>>2);
    h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h<<6) + (h>>2);
    h ^= std::hash<int>{}(k.gs) + 0x9e3779b9 + (h<<6) + (h>>2);
    return h;
  }
};
static std::unordered_map<GroupedSFBKey, void*, GroupedSFBHash> g_grouped_sfb_cache;
static std::mutex g_grouped_sfb_mutex;

// ============================================================================
// Grouped GEMM type struct.
// Mirrors Sm120BlockScaledGemm but uses GroupProblemShape for kGrouped mode.
// ============================================================================
template <
    typename ElementAMain_,  // mainloop A type (nv_float4_t<> or tuple<>)
    typename ElementBMain_,  // mainloop B type
    typename ElementOut_,    // output type (half_t or bfloat16_t)
    typename SFType_,        // scale factor element type
    int AlignA_, int AlignB_>
struct Sm120GroupedGemm {
  using ElementAMain = ElementAMain_;
  using ElementBMain = ElementBMain_;
  using ElementOut   = ElementOut_;
  using SFType       = SFType_;
  static constexpr int AlignA = AlignA_;
  static constexpr int AlignB = AlignB_;

  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementOut>::value;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OpClass,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          void, LayoutDTag*, AlignD,
          ElementOut, LayoutDTag*, AlignD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OpClass,
          ElementAMain, LayoutATag*, AlignA,
          ElementBMain, LayoutBTag*, AlignB,
          ElementAccumulator,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Extract types from the built kernel.
  using StrideA     = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB     = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD     = typename Gemm::GemmKernel::InternalStrideD;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using LayoutSFA   = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB   = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  // ElementA from Gemm::ElementA is the raw (not mainloop-wrapped) type.
  using ElementA    = typename Gemm::ElementA;
  using ElementB    = typename Gemm::ElementB;
};

// ============================================================================
// Concrete type aliases (non-template so device_kernel<> has correct linkage).
// ============================================================================
using NvFP4G_FP16 = Sm120GroupedGemm<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::half_t, cutlass::float_ue4m3_t, 32, 32>;

using NvFP4G_BF16 = Sm120GroupedGemm<
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::bfloat16_t, cutlass::float_ue4m3_t, 32, 32>;

using MxFP4G_FP16 = Sm120GroupedGemm<
    cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>,
    cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>,
    cutlass::half_t, cutlass::float_ue8m0_t, 32, 32>;

using MxFP4G_BF16 = Sm120GroupedGemm<
    cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>,
    cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>,
    cutlass::bfloat16_t, cutlass::float_ue8m0_t, 32, 32>;

// ============================================================================
// Non-template smem configurators.
// Called once per variant before gemm.initialize() to ensure the CUTLASS
// device_kernel<> has >48KB smem attribute set. With
// --device-entity-has-hidden-visibility=false, device_kernel<> has external
// linkage and cudaFuncSetAttribute() resolves correctly.
// ============================================================================
static void configure_smem_nvfp4_fp16() {
  static std::once_flag f;
  std::call_once(f, [] {
    using K = NvFP4G_FP16::GemmKernel;
    if (K::SharedStorageSize >= (48 << 10))
      cudaFuncSetAttribute((void*)cutlass::device_kernel<K>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, K::SharedStorageSize);
  });
}
static void configure_smem_nvfp4_bf16() {
  static std::once_flag f;
  std::call_once(f, [] {
    using K = NvFP4G_BF16::GemmKernel;
    if (K::SharedStorageSize >= (48 << 10))
      cudaFuncSetAttribute((void*)cutlass::device_kernel<K>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, K::SharedStorageSize);
  });
}
static void configure_smem_mxfp4_fp16() {
  static std::once_flag f;
  std::call_once(f, [] {
    using K = MxFP4G_FP16::GemmKernel;
    if (K::SharedStorageSize >= (48 << 10))
      cudaFuncSetAttribute((void*)cutlass::device_kernel<K>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, K::SharedStorageSize);
  });
}
static void configure_smem_mxfp4_bf16() {
  static std::once_flag f;
  std::call_once(f, [] {
    using K = MxFP4G_BF16::GemmKernel;
    if (K::SharedStorageSize >= (48 << 10))
      cudaFuncSetAttribute((void*)cutlass::device_kernel<K>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, K::SharedStorageSize);
  });
}

// ============================================================================
// get_or_reformat_grouped_sfb<GemmType>
// Reformats all E experts' SFBs into a single contiguous buffer (cached).
// ============================================================================
template <typename GemmType>
static void* get_or_reformat_grouped_sfb(
    const void* raw_scales,  // [E, N_src, K/gs] MLX row-major
    int E, int N_src, int N_padded, int K, int group_size,
    cudaStream_t stream) {
  using ScaleConfig = typename GemmType::ScaleConfig;
  using LayoutSFB   = typename GemmType::LayoutSFB;
  using SFType      = typename GemmType::SFType;

  GroupedSFBKey key{raw_scales, E, N_padded, K, group_size};
  {
    std::lock_guard<std::mutex> lk(g_grouped_sfb_mutex);
    auto it = g_grouped_sfb_cache.find(key);
    if (it != g_grouped_sfb_cache.end())
      return it->second;
  }

  int sf_vec_size = ScaleConfig::SFVecSize;
  int num_groups_per_row = K / sf_vec_size;
  // Layout with M=1, N_padded, K (M-independent for SFB).
  auto shape1 = cute::make_shape(1, N_padded, K, 1);
  LayoutSFB layout = ScaleConfig::tile_atom_to_shape_SFB(shape1);
  size_t sfb_per_expert = static_cast<size_t>(cute::cosize(layout));
  size_t total_bytes = static_cast<size_t>(E) * sfb_per_expert * sizeof(SFType);

  void* dst = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dst, total_bytes));

  constexpr int kT = 256;
  size_t tot = static_cast<size_t>(E) * N_padded * num_groups_per_row;
  int blk = static_cast<int>((tot + kT - 1) / kT);
  reformat_all_sfb_kernel<<<blk, kT, 0, stream>>>(
      reinterpret_cast<const SFType*>(raw_scales),
      reinterpret_cast<SFType*>(dst),
      layout, E, N_src, N_padded, num_groups_per_row, sf_vec_size, sfb_per_expert);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  {
    std::lock_guard<std::mutex> lk(g_grouped_sfb_mutex);
    auto it = g_grouped_sfb_cache.find(key);
    if (it != g_grouped_sfb_cache.end()) {
      cudaFree(dst);
      return it->second;
    }
    g_grouped_sfb_cache[key] = dst;
  }
  return dst;
}

// ============================================================================
// execute_grouped_fp4_gemm<GemmType, InputType>
// Quantizes activations per expert, builds grouped GEMM args via PDL, runs GEMM.
// ============================================================================
template <typename GemmType, typename InputType>
void execute_grouped_fp4_gemm(
    void (*configure_smem)(),
    const array& x_gathered,              // [B_total, K] gathered activations (sorted)
    const array& w_for_gemm,             // [E, N_padded, K/2] padded weights
    const void* grouped_sfb,             // pre-cached grouped SFB
    array& out_sorted,                   // [B_total, N_padded] output buffer
    const std::vector<int>& host_m_indptr,  // [E+1]
    int* d_m_indptr,                     // [E+1] on device
    int B_total, int E, int N_padded, int K,
    cu::CommandEncoder& enc) {
  using ScaleConfig = typename GemmType::ScaleConfig;
  using LayoutSFA   = typename GemmType::LayoutSFA;
  using LayoutSFB   = typename GemmType::LayoutSFB;
  using SFType      = typename GemmType::SFType;
  using Gemm        = typename GemmType::Gemm;
  using GemmKernel  = typename GemmType::GemmKernel;
  using ProblemShape = typename GemmType::ProblemShape;
  using StrideA     = typename GemmType::StrideA;
  using StrideB     = typename GemmType::StrideB;
  using StrideD     = typename GemmType::StrideD;
  using ElementA    = typename GemmType::ElementA;
  using ElementB    = typename GemmType::ElementB;
  using ElementD    = typename Gemm::EpilogueOutputOp::ElementOutput;
  static constexpr int SF_VEC_SIZE = ScaleConfig::SFVecSize;
  static constexpr int EPT = 4;  // ELEMS_PER_THREAD

  auto& stream = enc.stream();

  // ── Allocate quantized activation buffer [B_total, K/2] ─────────────────
  size_t xq_bytes = static_cast<size_t>(B_total) * K / 2;
  auto xq_alloc = cu::malloc_async(xq_bytes, enc);
  array xq_buf(xq_alloc, {B_total, K / 2}, uint8);
  enc.add_temporary(xq_buf);
  uint8_t* xq_base = gpu_ptr<uint8_t>(xq_buf);

  // ── Allocate grouped SFA buffer (zero-initialized) ──────────────────────
  // Upper bound on SFA rows: (B_total + E*127)/128*128 + 128 safety margin.
  constexpr int align_mn = 128;
  int sf_k = K / SF_VEC_SIZE;
  size_t max_sf_m =
      (static_cast<size_t>(B_total) +
       static_cast<size_t>(E) * (align_mn - 1) + align_mn - 1) /
          align_mn * align_mn + align_mn;
  size_t sfa_bytes = max_sf_m * sf_k * sizeof(SFType);
  auto sfa_alloc = cu::malloc_async(sfa_bytes, enc);
  array sfa_buf(sfa_alloc, {(int)sfa_bytes}, uint8);
  enc.add_temporary(sfa_buf);
  // MUST zero-init: TMA loads full 128-row tiles even when M < 128.
  cudaMemsetAsync(sfa_buf.data<void>(), 0, sfa_bytes, stream);
  SFType* sfa_base = reinterpret_cast<SFType*>(sfa_buf.data<void>());

  // ── Quantize activations per expert ─────────────────────────────────────
  constexpr int TPG = SF_VEC_SIZE / EPT;
  constexpr int GPW = 32 / TPG;
  constexpr int kT = 256;
  const InputType* x_base =
      reinterpret_cast<const InputType*>(x_gathered.data<void>());

  for (int e = 0; e < E; e++) {
    int m_off = host_m_indptr[e];
    int cnt_e = host_m_indptr[e + 1] - m_off;
    if (cnt_e == 0)
      continue;

    // SFA row offset for expert e.
    size_t sf_m_off =
        (static_cast<size_t>(m_off) + static_cast<size_t>(e) * (align_mn - 1)) /
        align_mn * align_mn;

    // Per-expert SFA layout.
    auto layout_sfa_e =
        ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(cnt_e, N_padded, K, 1));

    int n_grps = cnt_e * sf_k;
    int n_warps = (n_grps + GPW - 1) / GPW;
    int n_threads = n_warps * 32;
    int blocks = (n_threads + kT - 1) / kT;

    const InputType* x_e = x_base + static_cast<size_t>(m_off) * K;
    uint8_t* xq_e = xq_base + static_cast<size_t>(m_off) * K / 2;
    SFType* sfa_e = sfa_base + sf_m_off * sf_k;

    if constexpr (std::is_same_v<InputType, __half>) {
      grouped_quant_fp4_fp16<SF_VEC_SIZE, EPT>
          <<<blocks, kT, 0, stream>>>(x_e, xq_e, sfa_e, layout_sfa_e, cnt_e, K);
    } else {
      grouped_quant_fp4_bf16<SF_VEC_SIZE, EPT>
          <<<blocks, kT, 0, stream>>>(x_e, xq_e, sfa_e, layout_sfa_e, cnt_e, K);
    }
  }

  // ── Build per-group GEMM argument arrays ────────────────────────────────
  // Allocate all arrays in one encoder-managed buffer.
  using UProbShape = typename ProblemShape::UnderlyingProblemShape;
  size_t n = static_cast<size_t>(E);
  size_t arg_bytes =
      (n * sizeof(UProbShape) + 15) / 16 * 16 +
      (n * sizeof(const ElementA*) + 15) / 16 * 16 * 2 +  // A_ptr, B_ptr
      (n * sizeof(ElementD*) + 15) / 16 * 16 +
      (n * sizeof(const SFType*) + 15) / 16 * 16 * 2 +    // SFA_ptr, SFB_ptr
      (n * sizeof(StrideA) + 15) / 16 * 16 +
      (n * sizeof(StrideB) + 15) / 16 * 16 +
      (n * sizeof(StrideD) + 15) / 16 * 16 +
      (n * sizeof(LayoutSFA) + 15) / 16 * 16 +
      (n * sizeof(LayoutSFB) + 15) / 16 * 16 +
      256;  // alignment headroom

  auto arg_alloc = cu::malloc_async(arg_bytes, enc);
  array arg_buf(arg_alloc, {(int)arg_bytes}, uint8);
  enc.add_temporary(arg_buf);
  char* p = reinterpret_cast<char*>(arg_buf.data<void>());
  auto adv = [&](size_t sz) {
    char* r = p;
    p += (sz + 15) / 16 * 16;
    return r;
  };

  auto* d_prob  = reinterpret_cast<UProbShape*>(adv(n * sizeof(UProbShape)));
  auto* d_Aptr  = reinterpret_cast<const ElementA**>(adv(n * sizeof(ElementA*)));
  auto* d_Bptr  = reinterpret_cast<const ElementB**>(adv(n * sizeof(ElementB*)));
  auto* d_Dptr  = reinterpret_cast<ElementD**>(adv(n * sizeof(ElementD*)));
  auto* d_SFAptr = reinterpret_cast<const SFType**>(adv(n * sizeof(SFType*)));
  auto* d_SFBptr = reinterpret_cast<const SFType**>(adv(n * sizeof(SFType*)));
  auto* d_strA  = reinterpret_cast<StrideA*>(adv(n * sizeof(StrideA)));
  auto* d_strB  = reinterpret_cast<StrideB*>(adv(n * sizeof(StrideB)));
  auto* d_strD  = reinterpret_cast<StrideD*>(adv(n * sizeof(StrideD)));
  auto* d_layA  = reinterpret_cast<LayoutSFA*>(adv(n * sizeof(LayoutSFA)));
  auto* d_layB  = reinterpret_cast<LayoutSFB*>(adv(n * sizeof(LayoutSFB)));

  // Compute per-expert SFB size for the args kernel.
  auto sfb_layout_1 = GemmType::ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, N_padded, K, 1));
  size_t sfb_per_expert = static_cast<size_t>(cute::cosize(sfb_layout_1));

  // ── Launch compute args via PDL ──────────────────────────────────────────
  auto args_kernel = grouped_gemm_setup_args<
      ScaleConfig::SFVecSize, ScaleConfig,
      ElementA, ElementB, SFType, ElementD,
      UProbShape, StrideA, StrideB, StrideD, LayoutSFA, LayoutSFB>;

  int n_t = std::min(E, 1024);
  int n_b = (E + n_t - 1) / n_t;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = n_b;
  cfg.blockDim = n_t;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute pdl_attr[1];
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = true;
  cfg.numAttrs = 1;
  cfg.attrs = pdl_attr;

  CHECK_CUDA_ERROR(cudaLaunchKernelEx(
      &cfg, args_kernel,
      reinterpret_cast<const ElementA*>(xq_buf.data<void>()),
      reinterpret_cast<const ElementB*>(w_for_gemm.data<void>()),
      sfa_base,
      reinterpret_cast<SFType*>(const_cast<void*>(grouped_sfb)),
      reinterpret_cast<ElementD*>(const_cast<void*>(out_sorted.data<void>())),
      d_m_indptr, N_padded, K, E, sfb_per_expert,
      d_prob, d_Aptr, d_Bptr, d_SFAptr, d_SFBptr, d_Dptr,
      d_strA, d_strB, d_strD, d_layA, d_layB));

  // ── Launch CUTLASS grouped GEMM ──────────────────────────────────────────
  configure_smem();  // ensure smem attribute set for this variant

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {E, d_prob, nullptr},
      {d_Aptr, d_strA, d_Bptr, d_strB, d_SFAptr, d_layA, d_SFBptr, d_layB},
      {{1.0f, 0.0f}, nullptr, nullptr, d_Dptr, d_strD},
      hw_info};

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = nullptr;
  if (ws_size > 0) {
    auto ws_alloc = cu::malloc_async(ws_size, enc);
    array ws_buf(ws_alloc, {(int)ws_size}, uint8);
    enc.add_temporary(ws_buf);
    ws_ptr = ws_buf.data<void>();
  }

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "[grouped_gemm] can_implement failed: " +
        std::to_string(static_cast<int>(status)));
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "[grouped_gemm] initialize failed: " +
        std::to_string(static_cast<int>(status)));
  }
  // PDL disabled: CUTLASS grouped GEMM does not yet support launch_with_pdl.
  status = gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/false);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "[grouped_gemm] run failed: " + std::to_string(static_cast<int>(status)));
  }
}

// ============================================================================
// gather_qmm_grouped_gpu — public entry point
// ============================================================================
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
    const Stream& s) {
  int M = x.shape(-2);
  int K = x.shape(-1);
  int N = out.shape(-1);
  int B = static_cast<int>(lhs_indices.size());
  int E = w.shape(0);

  if (B == 0)
    return;

  int N_padded = (N + 127) / 128 * 128;
  bool needs_n_pad = (N_padded != N);

  array lhs_flat = ensure_row_contiguous(lhs_indices, enc, s);
  array rhs_flat = ensure_row_contiguous(rhs_indices, enc, s);

  enc.commit();
  enc.begin_direct_launch();

  // ── Phase 1: On-device counting sort by expert ──────────────────────────
  auto sorted = sort_gather_indices(lhs_flat, rhs_flat, B, E, enc, s);

  // ── Phase 2: Gather activations in sorted order ─────────────────────────
  int B_total = B * M;
  int xsz = size_of(x.dtype());
  array x_gathered({B_total, K}, x.dtype(), nullptr, {});
  x_gathered.set_data(cu::malloc_async(
      static_cast<size_t>(B_total) * K * xsz, enc));
  enc.add_temporary(x_gathered);

  enc.set_input_array(x);
  enc.set_input_array(sorted.sorted_lhs);
  enc.set_output_array(x_gathered);
  if (xsz == 2) {
    enc.add_kernel_node(
        local_gather_rows_kernel<__half>, dim3(B), dim3(256),
        gpu_ptr<__half>(x), gpu_ptr<__half>(x_gathered),
        gpu_ptr<uint32_t>(sorted.sorted_lhs), B, M, K);
  } else {
    enc.add_kernel_node(
        local_gather_rows_kernel<float>, dim3(B), dim3(256),
        gpu_ptr<float>(x), gpu_ptr<float>(x_gathered),
        gpu_ptr<uint32_t>(sorted.sorted_lhs), B, M, K);
  }

  // ── Phase 3: Build expert counts → m_indptr (host sync for quant loop) ──
  array d_cnt_buf({E}, uint32, nullptr, {});
  d_cnt_buf.set_data(cu::malloc_async(E * sizeof(uint32_t), enc));
  enc.add_temporary(d_cnt_buf);
  uint32_t* d_cnts = gpu_ptr<uint32_t>(d_cnt_buf);

  enc.set_output_array(d_cnt_buf);
  enc.add_kernel_node(local_zero_u32, dim3((E+255)/256), dim3(256), d_cnts, E);

  enc.set_input_array(sorted.sorted_rhs);
  enc.set_output_array(d_cnt_buf);
  enc.add_kernel_node(
      local_count_experts, dim3((B+255)/256), dim3(256),
      gpu_ptr<uint32_t>(sorted.sorted_rhs), B, d_cnts);

  // Alloc device m_indptr (int[E+1]).
  array d_indptr_buf({E + 1}, uint32, nullptr, {});
  d_indptr_buf.set_data(cu::malloc_async((E + 1) * sizeof(int), enc));
  enc.add_temporary(d_indptr_buf);
  int* d_m_indptr = reinterpret_cast<int*>(gpu_ptr<uint32_t>(d_indptr_buf));

  // Host sync: copy expert counts (E * 4 bytes — tiny).
  uint32_t* h_counts = nullptr;
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_counts, E * sizeof(uint32_t), cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      h_counts, d_cnts, E * sizeof(uint32_t), cudaMemcpyDeviceToHost, enc.stream()));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(enc.stream()));

  std::vector<int> host_m_indptr(E + 1);
  host_m_indptr[0] = 0;
  for (int e = 0; e < E; e++)
    host_m_indptr[e + 1] = host_m_indptr[e] + static_cast<int>(h_counts[e]);
  CHECK_CUDA_ERROR(cudaFreeHost(h_counts));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      d_m_indptr, host_m_indptr.data(), (E + 1) * sizeof(int),
      cudaMemcpyHostToDevice, enc.stream()));

  // ── Phase 4: Pad weights to N_padded if needed ──────────────────────────
  array w_for_gemm = w;
  if (needs_n_pad) {
    int w_cols = w.shape(-1);  // K/2 bytes for packed FP4
    array w_pad({E, N_padded, w_cols}, w.dtype(), nullptr, {});
    w_pad.set_data(cu::malloc_async(
        static_cast<size_t>(E) * N_padded * w_cols * size_of(w.dtype()), enc));
    enc.add_temporary(w_pad);
    cudaMemsetAsync(w_pad.data<void>(), 0, w_pad.nbytes(), enc.stream());
    // Copy E expert slabs, each N rows → N_padded rows (zero-padded).
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
        w_pad.data<uint8_t>(),
        static_cast<size_t>(N_padded) * w_cols,   // dst pitch per slab
        w.data<uint8_t>(),
        static_cast<size_t>(N) * w_cols,           // src pitch per slab
        static_cast<size_t>(N) * w_cols,           // copy width
        static_cast<size_t>(E),                    // copy height
        cudaMemcpyDefault, enc.stream()));
    w_for_gemm = w_pad;
  }

  // ── Phase 5: Get grouped SFB (cached) ───────────────────────────────────
  void* grouped_sfb = nullptr;
  if (mode == QuantizationMode::Nvfp4) {
    if (out.dtype() == float16)
      grouped_sfb = get_or_reformat_grouped_sfb<NvFP4G_FP16>(
          scales.data<void>(), E, N, N_padded, K, group_size, enc.stream());
    else
      grouped_sfb = get_or_reformat_grouped_sfb<NvFP4G_BF16>(
          scales.data<void>(), E, N, N_padded, K, group_size, enc.stream());
  } else {
    if (out.dtype() == float16)
      grouped_sfb = get_or_reformat_grouped_sfb<MxFP4G_FP16>(
          scales.data<void>(), E, N, N_padded, K, group_size, enc.stream());
    else
      grouped_sfb = get_or_reformat_grouped_sfb<MxFP4G_BF16>(
          scales.data<void>(), E, N, N_padded, K, group_size, enc.stream());
  }

  // ── Phase 6: Allocate sorted output buffer ──────────────────────────────
  int out_esz = size_of(out.dtype());
  array out_sorted({B_total, N_padded}, out.dtype(), nullptr, {});
  out_sorted.set_data(cu::malloc_async(
      static_cast<size_t>(B_total) * N_padded * out_esz, enc));
  enc.add_temporary(out_sorted);

  // ── Phase 7: Quantize + grouped GEMM ────────────────────────────────────
  bool is_bf16 = (x.dtype() == bfloat16);
  bool is_fp16_out = (out.dtype() == float16);

  if (mode == QuantizationMode::Nvfp4) {
    if (is_fp16_out) {
      if (!is_bf16)
        execute_grouped_fp4_gemm<NvFP4G_FP16, __half>(
            configure_smem_nvfp4_fp16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
      else
        execute_grouped_fp4_gemm<NvFP4G_FP16, __nv_bfloat16>(
            configure_smem_nvfp4_fp16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
    } else {
      if (!is_bf16)
        execute_grouped_fp4_gemm<NvFP4G_BF16, __half>(
            configure_smem_nvfp4_bf16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
      else
        execute_grouped_fp4_gemm<NvFP4G_BF16, __nv_bfloat16>(
            configure_smem_nvfp4_bf16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
    }
  } else {
    // Mxfp4
    if (is_fp16_out) {
      if (!is_bf16)
        execute_grouped_fp4_gemm<MxFP4G_FP16, __half>(
            configure_smem_mxfp4_fp16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
      else
        execute_grouped_fp4_gemm<MxFP4G_FP16, __nv_bfloat16>(
            configure_smem_mxfp4_fp16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
    } else {
      if (!is_bf16)
        execute_grouped_fp4_gemm<MxFP4G_BF16, __half>(
            configure_smem_mxfp4_bf16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
      else
        execute_grouped_fp4_gemm<MxFP4G_BF16, __nv_bfloat16>(
            configure_smem_mxfp4_bf16, x_gathered, w_for_gemm, grouped_sfb,
            out_sorted, host_m_indptr, d_m_indptr, B_total, E, N_padded, K, enc);
    }
  }

  // ── Phase 8: Scatter output back to original order ──────────────────────
  enc.set_input_array(out_sorted);
  enc.set_input_array(sorted.sorted_perm);
  enc.set_output_array(out);

  if (!needs_n_pad) {
    // N == N_padded: standard scatter (no column truncation).
    scatter_gather_output(out_sorted, sorted.sorted_perm, out, B, M, N, enc);
  } else {
    // N_padded > N: scatter with column truncation.
    if (out_esz == 2) {
      enc.add_kernel_node(
          scatter_rows_truncated<__half>, dim3(B), dim3(256),
          gpu_ptr<__half>(out_sorted), gpu_ptr<__half>(out),
          gpu_ptr<uint32_t>(sorted.sorted_perm), B, M, N_padded, N);
    } else {
      enc.add_kernel_node(
          scatter_rows_truncated<float>, dim3(B), dim3(256),
          gpu_ptr<float>(out_sorted), gpu_ptr<float>(out),
          gpu_ptr<uint32_t>(sorted.sorted_perm), B, M, N_padded, N);
    }
  }

  enc.end_direct_launch();
}

#else  // !SM120

void gather_qmm_grouped_gpu(
    const array&, const array&, const array&,
    const array&, const array&,
    array&, int, int, QuantizationMode,
    cu::CommandEncoder&, const Stream&) {
  throw std::runtime_error("[gather_qmm_grouped] SM120 not supported on this build.");
}

#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED || SM121

} // namespace mlx::core
