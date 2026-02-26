// Copyright © 2025 Apple Inc.

// GatherQMM CUDA implementation.
// Strategy: group indices by expert, dequantize each expert's weights once,
// then run one cuBLAS GEMM per expert group.

#include "mlx/backend/cuda/quantized/gather_qmm.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/primitives.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <unordered_map>
#include <vector>

namespace mlx::core {

// Gather kernel: for each index i in [0, num_items), copy M*row_elems elements
// from src[gather_indices[i] * M * row_elems] to dst[i * M * row_elems].
template <typename T>
__global__ void gather_rows_kernel(
    const T* src,
    T* dst,
    const uint32_t* gather_indices,
    int num_items,
    int M,
    int row_elems) {
  int idx = blockIdx.x;
  if (idx >= num_items)
    return;

  uint32_t src_batch = gather_indices[idx];
  size_t total_elems = static_cast<size_t>(M) * row_elems;
  const T* src_ptr = src + static_cast<size_t>(src_batch) * total_elems;
  T* dst_ptr = dst + static_cast<size_t>(idx) * total_elems;

  for (size_t i = threadIdx.x; i < total_elems; i += blockDim.x) {
    dst_ptr[i] = src_ptr[i];
  }
}

// Scatter kernel: for each index i in [0, num_items), copy M*row_elems
// elements from src[i * M * row_elems] to
// dst[scatter_indices[i] * M * row_elems].
template <typename T>
__global__ void scatter_rows_kernel(
    const T* src,
    T* dst,
    const int* scatter_indices,
    int num_items,
    int M,
    int row_elems) {
  int idx = blockIdx.x;
  if (idx >= num_items)
    return;

  int dst_batch = scatter_indices[idx];
  size_t total_elems = static_cast<size_t>(M) * row_elems;
  const T* src_ptr = src + static_cast<size_t>(idx) * total_elems;
  T* dst_ptr = dst + static_cast<size_t>(dst_batch) * total_elems;

  for (size_t i = threadIdx.x; i < total_elems; i += blockDim.x) {
    dst_ptr[i] = src_ptr[i];
  }
}

// Bookkeeping for one expert group after host-side grouping.
struct ExpertGroup {
  uint32_t expert_id;
  int num_items;
  int offset; // offset into flat index buffers
};

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
    const Stream& s) {
  int M = x.shape(-2);
  int K = x.shape(-1);
  int N = out.shape(-1);
  int B = lhs_indices.size();

  if (B == 0)
    return;

  // --- Phase 1: Synchronize and read indices on host ---
  // Make indices contiguous (they may be broadcast/strided views).
  array lhs_flat = ensure_row_contiguous(lhs_indices, enc, s);
  array rhs_flat = ensure_row_contiguous(rhs_indices, enc, s);

  // Declare indices as inputs so the encoder ensures they're computed,
  // then commit + sync so we can safely read them on the host.
  enc.set_input_array(lhs_flat);
  enc.set_input_array(rhs_flat);
  enc.synchronize();

  // Read contiguous indices to host.
  std::vector<uint32_t> lhs_host(B), rhs_host(B);
  CHECK_CUDA_ERROR(cudaMemcpy(
      lhs_host.data(),
      lhs_flat.data<uint32_t>(),
      B * sizeof(uint32_t),
      cudaMemcpyDefault));
  CHECK_CUDA_ERROR(cudaMemcpy(
      rhs_host.data(),
      rhs_flat.data<uint32_t>(),
      B * sizeof(uint32_t),
      cudaMemcpyDefault));

  // Group by expert so we dequantize each expert's weights only once.
  std::unordered_map<uint32_t, std::vector<std::pair<int, uint32_t>>> groups;
  for (int i = 0; i < B; i++) {
    groups[rhs_host[i]].push_back({i, lhs_host[i]});
  }

  // --- Phase 2: Build flat index buffers and upload in one shot ---
  std::vector<uint32_t> all_gather(B);
  std::vector<int> all_scatter(B);
  std::vector<ExpertGroup> expert_groups;
  int offset = 0;
  for (auto& [expert_id, pairs] : groups) {
    ExpertGroup eg;
    eg.expert_id = expert_id;
    eg.num_items = static_cast<int>(pairs.size());
    eg.offset = offset;
    for (int i = 0; i < eg.num_items; i++) {
      all_gather[offset + i] = pairs[i].second; // lhs_index
      all_scatter[offset + i] = pairs[i].first; // out_index
    }
    offset += eg.num_items;
    expert_groups.push_back(eg);
  }

  // Allocate device buffers for indices.
  array d_gather_buf({B}, uint32, nullptr, {});
  d_gather_buf.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(d_gather_buf);

  array d_scatter_buf({B}, int32, nullptr, {});
  d_scatter_buf.set_data(cu::malloc_async(B * sizeof(int), enc));
  enc.add_temporary(d_scatter_buf);

  // Upload all indices. Synchronous cudaMemcpy is safe here because we just
  // called enc.synchronize() and haven't started a new graph yet.
  CHECK_CUDA_ERROR(cudaMemcpy(
      gpu_ptr<uint32_t>(d_gather_buf),
      all_gather.data(),
      B * sizeof(uint32_t),
      cudaMemcpyDefault));
  CHECK_CUDA_ERROR(cudaMemcpy(
      gpu_ptr<int>(d_scatter_buf),
      all_scatter.data(),
      B * sizeof(int),
      cudaMemcpyDefault));

  // --- Phase 3: Per-expert dequant + gather + GEMM + scatter ---
  size_t w_expert_stride = w.strides()[0];
  size_t scales_expert_stride = scales.strides()[0];
  size_t biases_expert_stride = biases ? biases->strides()[0] : 0;

  int out_elem_size = size_of(out.dtype());
  int x_elem_size = size_of(x.dtype());

  for (auto& eg : expert_groups) {
    uint32_t expert_id = eg.expert_id;
    int num_items = eg.num_items;
    int batch_m = num_items * M;

    // Pointers into the flat index buffers for this expert group.
    const uint32_t* gather_ptr =
        gpu_ptr<uint32_t>(d_gather_buf) + eg.offset;
    const int* scatter_ptr = gpu_ptr<int>(d_scatter_buf) + eg.offset;

    // --- Expert weight slice views (zero-copy) ---
    auto make_expert_slice =
        [&](const array& arr, size_t expert_stride) -> array {
      Shape slice_shape(arr.shape().begin() + 1, arr.shape().end());
      Strides slice_strides(arr.strides().begin() + 1, arr.strides().end());
      array slice(slice_shape, arr.dtype(), nullptr, {});
      slice.copy_shared_buffer(
          arr,
          slice_strides,
          arr.flags(),
          arr.data_size(),
          static_cast<int64_t>(expert_id) *
              static_cast<int64_t>(expert_stride));
      return slice;
    };

    array w_expert = make_expert_slice(w, w_expert_stride);
    array scales_expert = make_expert_slice(scales, scales_expert_stride);
    std::optional<array> biases_expert = std::nullopt;
    if (biases) {
      biases_expert = make_expert_slice(*biases, biases_expert_stride);
    }

    // --- Dequantize expert weights → w_fp (N_expert, K) ---
    // (dequantize functions internally call set_input_array/set_output_array)
    int N_expert = w_expert.shape(0);
    array w_dequant({N_expert, K}, out.dtype(), nullptr, {});
    w_dequant.set_data(cu::malloc_async(
        static_cast<size_t>(N_expert) * K * out_elem_size, enc));
    enc.add_temporary(w_dequant);

    if (mode == QuantizationMode::Affine) {
      affine_dequantize(
          w_expert,
          scales_expert,
          *biases_expert,
          w_dequant,
          group_size,
          bits,
          enc,
          s);
    } else {
      fp_dequantize(
          w_expert, scales_expert, w_dequant, group_size, bits, enc, s);
    }

    // --- Gather x rows → x_gathered (batch_m, K) ---
    array x_gathered({batch_m, K}, x.dtype(), nullptr, {});
    x_gathered.set_data(cu::malloc_async(
        static_cast<size_t>(batch_m) * K * x_elem_size, enc));
    enc.add_temporary(x_gathered);

    // Declare data dependencies for the gather kernel.
    enc.set_input_array(x);
    enc.set_input_array(d_gather_buf);
    enc.set_output_array(x_gathered);

    dim3 grid(num_items);
    dim3 block(256);

    if (x_elem_size == 2) {
      auto src_p = gpu_ptr<__half>(x);
      auto dst_p = gpu_ptr<__half>(x_gathered);
      enc.add_kernel_node(
          gather_rows_kernel<__half>,
          grid, block, 0,
          src_p, dst_p, gather_ptr, num_items, M, K);
    } else {
      auto src_p = gpu_ptr<float>(x);
      auto dst_p = gpu_ptr<float>(x_gathered);
      enc.add_kernel_node(
          gather_rows_kernel<float>,
          grid, block, 0,
          src_p, dst_p, gather_ptr, num_items, M, K);
    }

    // --- cuBLAS GEMM: x_gathered @ w_dequant.T → out_gathered ---
    // (CublasGemm::run internally calls set_input_array/set_output_array)
    array out_gathered({batch_m, N}, out.dtype(), nullptr, {});
    out_gathered.set_data(cu::malloc_async(
        static_cast<size_t>(batch_m) * N * out_elem_size, enc));
    enc.add_temporary(out_gathered);

    // w_dequant is (N_expert, K) row-major.
    // For transpose=true: out = x @ w.T
    //   cuBLAS: C(batch_m, N) = A(batch_m, K) * B(N, K)^T
    CublasGemm gemm(
        enc.device(),
        out.dtype(),
        /* a_transposed */ false,
        /* a_rows */ static_cast<uint64_t>(batch_m),
        /* a_cols */ static_cast<uint64_t>(K),
        /* lda */ static_cast<int64_t>(K),
        /* b_transposed */ true,
        /* b_rows */ static_cast<uint64_t>(K),
        /* b_cols */ static_cast<uint64_t>(N_expert),
        /* ldb */ static_cast<int64_t>(K),
        /* batch_count */ 1,
        /* a_batch_stride */ 0,
        /* b_batch_stride */ 0);

    gemm.run(
        enc,
        out_gathered,
        x_gathered,
        w_dequant,
        /* batch_shape */ {1},
        /* a_batch_strides */ {0},
        /* b_batch_strides */ {0},
        /* alpha */ 1.0f);

    // --- Scatter results to output positions ---
    // Declare data dependencies for the scatter kernel.
    enc.set_input_array(out_gathered);
    enc.set_input_array(d_scatter_buf);
    enc.set_output_array(out);

    if (out_elem_size == 2) {
      auto src_p = gpu_ptr<__half>(out_gathered);
      auto dst_p = gpu_ptr<__half>(out);
      enc.add_kernel_node(
          scatter_rows_kernel<__half>,
          grid, block, 0,
          src_p, dst_p, scatter_ptr, num_items, M, N);
    } else {
      auto src_p = gpu_ptr<float>(out_gathered);
      auto dst_p = gpu_ptr<float>(out);
      enc.add_kernel_node(
          scatter_rows_kernel<float>,
          grid, block, 0,
          src_p, dst_p, scatter_ptr, num_items, M, N);
    }
  }
}

} // namespace mlx::core
