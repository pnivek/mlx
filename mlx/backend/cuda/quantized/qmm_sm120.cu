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
// For MXFP8: Uses m16n8k32 MMA with ue8m0 scale factors, SFVecSize=32.
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

// Must include CUTLASS arch config BEFORE the guard check so the SM120/SM121
// macros are defined when compiling for the right target.
#include "cutlass/arch/config.h"

// SM120 block-scaled GEMM requires CUTLASS SM120 support.
// Guard everything so the file compiles cleanly on older toolkits.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

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
    typename TileShape>
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
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

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

// ============================================================================
// Activation quantization kernel.
//
// Quantizes fp16/bf16 activations to FP4 (e2m1) with per-block scale factors
// in CUTLASS interleaved layout. Each thread handles one block of sf_vec_size
// elements.
// ============================================================================

// Quantize a float to 4-bit e2m1 encoding (1 sign + 2 exp + 1 mantissa).
// Representable magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
__device__ __forceinline__ uint8_t quantize_float_to_e2m1(float v) {
  float av = fabsf(v);
  uint8_t bits;
  // Round to nearest representable value using midpoints.
  if (av < 0.25f)
    bits = 0b000; // 0
  else if (av < 0.75f)
    bits = 0b001; // 0.5
  else if (av < 1.25f)
    bits = 0b010; // 1.0
  else if (av < 1.75f)
    bits = 0b011; // 1.5
  else if (av < 2.5f)
    bits = 0b100; // 2.0
  else if (av < 3.5f)
    bits = 0b101; // 3.0
  else if (av < 5.0f)
    bits = 0b110; // 4.0
  else
    bits = 0b111; // 6.0
  // Sign bit in bit 3.
  if (v < 0.0f) bits |= 0b1000;
  return bits;
}

template <typename SFType, typename LayoutSF>
__global__ void quantize_activation_fp4_kernel(
    const __half* __restrict__ input, // (M, K) row-major
    uint8_t* __restrict__ output,     // (M, K/2) packed FP4
    SFType* __restrict__ sf_out,      // CUTLASS-format scale factors
    LayoutSF layout_sfa,
    int M,
    int K,
    int sf_vec_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_groups = K / sf_vec_size;
  int total = M * num_groups;
  if (idx >= total) return;

  int m = idx / num_groups;
  int g = idx % num_groups;
  const __half* block_start = input + m * K + g * sf_vec_size;

  // Compute max absolute value in block.
  float amax = 0.0f;
  for (int i = 0; i < sf_vec_size; i++) {
    amax = fmaxf(amax, fabsf(__half2float(block_start[i])));
  }

  // Scale: amax / max_representable_e2m1 (6.0).
  float scale = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  // Store scale factor in CUTLASS interleaved layout.
  auto sf_offset = layout_sfa(m, g * sf_vec_size, 0);
  sf_out[sf_offset] = static_cast<SFType>(scale);

  // Quantize and pack FP4 values (2 per byte, lower nibble first).
  int pack_base = m * (K / 2) + g * (sf_vec_size / 2);
  for (int i = 0; i < sf_vec_size; i += 2) {
    float v0 = __half2float(block_start[i]) * inv_scale;
    float v1 = __half2float(block_start[i + 1]) * inv_scale;
    uint8_t q0 = quantize_float_to_e2m1(v0);
    uint8_t q1 = quantize_float_to_e2m1(v1);
    output[pack_base + i / 2] = q0 | (q1 << 4);
  }
}

// BF16 variant for activation quantization.
template <typename SFType, typename LayoutSF>
__global__ void quantize_activation_fp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    SFType* __restrict__ sf_out,
    LayoutSF layout_sfa,
    int M,
    int K,
    int sf_vec_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_groups = K / sf_vec_size;
  int total = M * num_groups;
  if (idx >= total) return;

  int m = idx / num_groups;
  int g = idx % num_groups;
  const __nv_bfloat16* block_start = input + m * K + g * sf_vec_size;

  float amax = 0.0f;
  for (int i = 0; i < sf_vec_size; i++) {
    amax = fmaxf(amax, fabsf(__bfloat162float(block_start[i])));
  }

  float scale = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  auto sf_offset = layout_sfa(m, g * sf_vec_size, 0);
  sf_out[sf_offset] = static_cast<SFType>(scale);

  int pack_base = m * (K / 2) + g * (sf_vec_size / 2);
  for (int i = 0; i < sf_vec_size; i += 2) {
    float v0 = __bfloat162float(block_start[i]) * inv_scale;
    float v1 = __bfloat162float(block_start[i + 1]) * inv_scale;
    uint8_t q0 = quantize_float_to_e2m1(v0);
    uint8_t q1 = quantize_float_to_e2m1(v1);
    output[pack_base + i / 2] = q0 | (q1 << 4);
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
  using LayoutSFB = typename GemmType::LayoutSFB;
  using ElementA = typename GemmType::ElementA;
  using SFType = typename ElementA::ScaleFactorType;

  int M = out.shape(-2);
  int N = out.shape(-1);
  int K = x.shape(-1);
  int sf_vec_size = BlkConfig::SFVecSize;
  int num_act_groups = M * (K / sf_vec_size);
  int num_wt_groups = N * (K / sf_vec_size);

  auto problem_shape = cute::make_shape(M, N, K, 1);
  LayoutSFA layout_SFA = BlkConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = BlkConfig::tile_atom_to_shape_SFB(problem_shape);

  int sfa_size = cute::cosize(layout_SFA);
  int sfb_size = cute::cosize(layout_SFB);

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

  // 3. Reformatted weight scale factors in CUTLASS layout.
  size_t sfb_bytes = static_cast<size_t>(sfb_size) * sizeof(SFType);
  auto sfb_alloc = cu::malloc_async(sfb_bytes, encoder);
  array sfb_buf(sfb_alloc, {static_cast<int>(sfb_bytes)}, uint8);
  encoder.add_temporary(sfb_buf);

  // NOTE: No weight transpose needed! CUTLASS ColumnMajor B with
  // TagToStrideB<ColumnMajor> = Stride<int64_t, _1, int64_t> gives
  // stride_B = (K, 1, 0), meaning B(n,k) = ptr[n*K + k] — K contiguous.
  // MLX stores weights as (N, K/2) row-major (K contiguous), which matches.

  auto& stream = encoder.stream();
  constexpr int kThreads = 256;

  // Step 1: Quantize activations to FP4.
  {
    int blocks = (num_act_groups + kThreads - 1) / kThreads;
    if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
      quantize_activation_fp4_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data<void>()),
          reinterpret_cast<uint8_t*>(x_q_buf.data<void>()),
          reinterpret_cast<SFType*>(sfa_buf.data<void>()),
          layout_SFA, M, K, sf_vec_size);
    } else {
      quantize_activation_fp4_kernel<<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const __half*>(x.data<void>()),
          reinterpret_cast<uint8_t*>(x_q_buf.data<void>()),
          reinterpret_cast<SFType*>(sfa_buf.data<void>()),
          layout_SFA, M, K, sf_vec_size);
    }
  }

  // Step 2: Reformat weight scale factors from row-major to CUTLASS layout.
  {
    int wt_groups_per_row = K / group_size;
    int blocks = (num_wt_groups + kThreads - 1) / kThreads;
    reformat_sf_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const SFType*>(scales.data<void>()),
        reinterpret_cast<SFType*>(sfb_buf.data<void>()),
        layout_SFB, N, wt_groups_per_row, sf_vec_size);
  }

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
      sfb_buf.data<void>(),
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

// Helper: dispatch to the correct execute_sm120_fp4_gemm variant based on
// group_size and output dtype.
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
      void* kernel_ptr = get_configured_kernel_nvfp4_bf16();
      execute_sm120_fp4_gemm<NvFP4_BF16_Gemm, __nv_bfloat16>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else if (out.dtype() == float16) {
      void* kernel_ptr = get_configured_kernel_nvfp4_fp16();
      execute_sm120_fp4_gemm<NvFP4_FP16_Gemm, __half>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else {
      throw std::runtime_error(
          fmt::format("{} Unsupported dtype for SM120 NVFP4 GEMM.", tag));
    }
  } else if (group_size == 32) {
    // MXFP4: mx_float4_t with ue8m0 scale factors, SFVecSize=32.
    if (out.dtype() == bfloat16) {
      void* kernel_ptr = get_configured_kernel_mxfp4_bf16();
      execute_sm120_fp4_gemm<MxFP4_BF16_Gemm, __nv_bfloat16>(
          kernel_ptr, x, w, scales, out, group_size, encoder);
    } else if (out.dtype() == float16) {
      void* kernel_ptr = get_configured_kernel_mxfp4_fp16();
      execute_sm120_fp4_gemm<MxFP4_FP16_Gemm, __half>(
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
// ============================================================================
void cute_qmm_fp8_sm120(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    cu::CommandEncoder& encoder) {
  // TODO: Implement MXFP8 path.
  // Similar to FP4 but with mx_float8_t<float_e4m3_t>, SFVecSize=32,
  // TileShape=128x128x64, and FP8 activation quantization.
  throw std::runtime_error(
      "[qmm_fp8_sm120] MXFP8 SM120 native GEMM not yet implemented.");
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

} // namespace mlx::core

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED
