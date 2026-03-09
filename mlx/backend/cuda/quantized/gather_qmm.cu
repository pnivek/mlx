// Copyright © 2025 Apple Inc.

// GatherQMM CUDA implementation.
//
// Two execution paths:
// 1. QMV decode (M<=16, FP modes): on-device sorted gather + QMV kernel.
// 2. Prefill (M>16 or affine): host sync to read indices, per-expert
//    dequant + cuBLAS GEMM loop.

#include "mlx/backend/cuda/quantized/gather_qmm.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/primitives.h"

#include <cuda/cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <unordered_map>
#include <vector>

namespace mlx::core {

// ============================================================================
// Shared helpers
// ============================================================================

// Vectorized copy helper: use uint4 (16-byte) loads when aligned.
template <typename T>
__device__ void vectorized_copy(const T* src_ptr, T* dst_ptr, size_t total_elems) {
  constexpr int VEC_ELEMS = 16 / sizeof(T);
  bool aligned = (reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0) &&
                 (reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0) &&
                 (total_elems % VEC_ELEMS == 0);

  if (aligned) {
    size_t vec_count = total_elems / VEC_ELEMS;
    auto* src_vec = reinterpret_cast<const uint4*>(src_ptr);
    auto* dst_vec = reinterpret_cast<uint4*>(dst_ptr);
    for (size_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (size_t i = threadIdx.x; i < total_elems; i += blockDim.x) {
      dst_ptr[i] = src_ptr[i];
    }
  }
}

// Gather kernel: copy M*row_elems elements per item using indirect index.
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

  vectorized_copy(src_ptr, dst_ptr, total_elems);
}

// Scatter kernel: copy M*row_elems elements per item to indirect position.
template <typename T>
__global__ void scatter_rows_kernel(
    const T* src,
    T* dst,
    const uint32_t* scatter_indices,
    int num_items,
    int M,
    int row_elems) {
  int idx = blockIdx.x;
  if (idx >= num_items)
    return;

  uint32_t dst_batch = scatter_indices[idx];
  size_t total_elems = static_cast<size_t>(M) * row_elems;
  const T* src_ptr = src + static_cast<size_t>(idx) * total_elems;
  T* dst_ptr = dst + static_cast<size_t>(dst_batch) * total_elems;

  vectorized_copy(src_ptr, dst_ptr, total_elems);
}

// ============================================================================
// Fused path: on-device counting sort + grouped GEMM (no host sync)
// ============================================================================

// Zero a uint32 buffer. Used to initialize expert_counts before histogram.
__global__ void zero_uint32_kernel(uint32_t* buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    buf[i] = 0;
  }
}

// Step 1: Count items per expert using global atomics.
__global__ void count_experts_kernel(
    const uint32_t* rhs_indices,
    int B,
    uint32_t* expert_counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B) {
    atomicAdd(&expert_counts[rhs_indices[i]], 1);
  }
}

// Step 2: Exclusive prefix sum on expert_counts → expert_offsets.
// Also initializes write_pos = expert_offsets for the scatter step.
// Single block, E ≤ 1024 threads.
__global__ void prefix_sum_kernel(
    const uint32_t* expert_counts,
    int E,
    uint32_t* expert_offsets,
    uint32_t* expert_write_pos) {
  if (threadIdx.x == 0) {
    uint32_t sum = 0;
    for (int e = 0; e < E; e++) {
      expert_offsets[e] = sum;
      expert_write_pos[e] = sum;
      sum += expert_counts[e];
    }
  }
}

// Step 3: Scatter items to sorted positions by expert.
// Uses atomic increment on write_pos to get unique sorted position.
__global__ void scatter_sort_kernel(
    const uint32_t* rhs_indices,
    const uint32_t* lhs_indices,
    int B,
    uint32_t* expert_write_pos,
    uint32_t* sorted_lhs,
    uint32_t* sorted_rhs,
    uint32_t* sorted_perm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B) {
    uint32_t expert = rhs_indices[i];
    uint32_t pos = atomicAdd(&expert_write_pos[expert], 1);
    sorted_lhs[pos] = lhs_indices[i];
    sorted_rhs[pos] = expert;
    sorted_perm[pos] = static_cast<uint32_t>(i);
  }
}


SortedGatherIndices sort_gather_indices(
    const array& lhs_indices,
    const array& rhs_indices,
    int B,
    int E,
    cu::CommandEncoder& enc,
    const Stream& s) {
  // Caller must ensure indices are row-contiguous.
  const array& lhs_flat = lhs_indices;
  const array& rhs_flat = rhs_indices;

  // Allocate sort buffer: [counts(E)|offsets(E)|write_pos(E)]
  size_t internal_size = static_cast<size_t>(E) * sizeof(uint32_t) * 3;
  array sort_buf(
      cu::malloc_async(internal_size, enc),
      {static_cast<int>(internal_size)},
      uint8);
  enc.add_temporary(sort_buf);

  uint32_t* sort_base = gpu_ptr<uint32_t>(sort_buf);
  uint32_t* d_counts = sort_base;
  uint32_t* d_offsets = sort_base + E;
  uint32_t* d_write_pos = sort_base + 2 * E;

  // Allocate output arrays.
  array sorted_lhs({B}, uint32, nullptr, {});
  sorted_lhs.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(sorted_lhs);

  array sorted_rhs({B}, uint32, nullptr, {});
  sorted_rhs.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(sorted_rhs);

  array sorted_perm({B}, uint32, nullptr, {});
  sorted_perm.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(sorted_perm);

  // Zero counts.
  enc.set_output_array(sort_buf);
  {
    int threads = 256;
    int blocks = (E + threads - 1) / threads;
    enc.add_kernel_node(
        zero_uint32_kernel,
        dim3(blocks), dim3(threads),
        d_counts, E);
  }

  // Count items per expert.
  enc.set_input_array(rhs_flat);
  enc.set_output_array(sort_buf);
  {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    enc.add_kernel_node(
        count_experts_kernel,
        dim3(blocks), dim3(threads),
        gpu_ptr<uint32_t>(rhs_flat), B, d_counts);
  }

  // Prefix sum.
  enc.set_input_array(sort_buf);
  enc.set_output_array(sort_buf);
  enc.add_kernel_node(
      prefix_sum_kernel,
      dim3(1), dim3(1),
      d_counts, E, d_offsets, d_write_pos);

  // Scatter to sorted positions.
  enc.set_input_array(rhs_flat);
  enc.set_input_array(lhs_flat);
  enc.set_output_array(sorted_lhs);
  enc.set_output_array(sorted_rhs);
  enc.set_output_array(sorted_perm);
  {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    enc.add_kernel_node(
        scatter_sort_kernel,
        dim3(blocks), dim3(threads),
        gpu_ptr<uint32_t>(rhs_flat),
        gpu_ptr<uint32_t>(lhs_flat),
        B, d_write_pos,
        gpu_ptr<uint32_t>(sorted_lhs),
        gpu_ptr<uint32_t>(sorted_rhs),
        gpu_ptr<uint32_t>(sorted_perm));
  }

  return {std::move(sorted_lhs), std::move(sorted_rhs), std::move(sorted_perm)};
}

void scatter_gather_output(
    const array& src,
    const array& sorted_perm,
    array& dst,
    int B,
    int M,
    int N,
    cu::CommandEncoder& enc) {
  int elem_size = size_of(dst.dtype());

  enc.set_input_array(src);
  enc.set_input_array(sorted_perm);
  enc.set_output_array(dst);

  if (elem_size == 2) {
    enc.add_kernel_node(
        scatter_rows_kernel<__half>,
        dim3(B), dim3(256),
        gpu_ptr<__half>(src),
        gpu_ptr<__half>(dst),
        gpu_ptr<uint32_t>(sorted_perm),
        B, M, N);
  } else {
    enc.add_kernel_node(
        scatter_rows_kernel<float>,
        dim3(B), dim3(256),
        gpu_ptr<float>(src),
        gpu_ptr<float>(dst),
        gpu_ptr<uint32_t>(sorted_perm),
        B, M, N);
  }
}

// ============================================================================
// Host-sync path: per-expert dequant + cuBLAS GEMM loop
// ============================================================================

struct ExpertGroup {
  uint32_t expert_id;
  int num_items;
  int offset;
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
  array lhs_flat = ensure_row_contiguous(lhs_indices, enc, s);
  array rhs_flat = ensure_row_contiguous(rhs_indices, enc, s);

  enc.set_input_array(lhs_flat);
  enc.set_input_array(rhs_flat);
  enc.synchronize();

  // Use direct kernel launches (bypass CUDA graph) to work around
  // cudaGraphExecUpdate issues on SM121/CUDA 13.0. The per-expert loop
  // creates a variable number of graph nodes depending on which experts
  // are active, causing graph topology mismatches on update.
  enc.begin_direct_launch();

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

  // Group by expert.
  std::unordered_map<uint32_t, std::vector<std::pair<int, uint32_t>>> groups;
  for (int i = 0; i < B; i++) {
    groups[rhs_host[i]].push_back({i, lhs_host[i]});
  }

  // --- Phase 2: Build flat index buffers and upload ---
  std::vector<uint32_t> all_gather(B);
  std::vector<uint32_t> all_scatter(B);
  std::vector<ExpertGroup> expert_groups;
  int offset = 0;
  for (auto& [expert_id, pairs] : groups) {
    ExpertGroup eg;
    eg.expert_id = expert_id;
    eg.num_items = static_cast<int>(pairs.size());
    eg.offset = offset;
    for (int i = 0; i < eg.num_items; i++) {
      all_gather[offset + i] = pairs[i].second;
      all_scatter[offset + i] = pairs[i].first;
    }
    offset += eg.num_items;
    expert_groups.push_back(eg);
  }

  array d_gather_buf({B}, uint32, nullptr, {});
  d_gather_buf.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(d_gather_buf);

  array d_scatter_buf({B}, uint32, nullptr, {});
  d_scatter_buf.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(d_scatter_buf);

  CHECK_CUDA_ERROR(cudaMemcpy(
      gpu_ptr<uint32_t>(d_gather_buf),
      all_gather.data(),
      B * sizeof(uint32_t),
      cudaMemcpyDefault));
  CHECK_CUDA_ERROR(cudaMemcpy(
      gpu_ptr<uint32_t>(d_scatter_buf),
      all_scatter.data(),
      B * sizeof(uint32_t),
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

    const uint32_t* gather_ptr =
        gpu_ptr<uint32_t>(d_gather_buf) + eg.offset;
    const uint32_t* scatter_ptr = gpu_ptr<uint32_t>(d_scatter_buf) + eg.offset;

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

    int N_expert = w_expert.shape(0);
    array w_dequant({N_expert, K}, out.dtype(), nullptr, {});
    w_dequant.set_data(cu::malloc_async(
        static_cast<size_t>(N_expert) * K * out_elem_size, enc));
    enc.add_temporary(w_dequant);

    if (mode == QuantizationMode::Affine) {
      affine_dequantize(
          w_expert, scales_expert, *biases_expert,
          w_dequant, group_size, bits, enc, s);
    } else {
      fp_dequantize(
          w_expert, scales_expert, w_dequant, group_size, bits, std::nullopt, enc, s);
    }

    array x_gathered({batch_m, K}, x.dtype(), nullptr, {});
    x_gathered.set_data(cu::malloc_async(
        static_cast<size_t>(batch_m) * K * x_elem_size, enc));
    enc.add_temporary(x_gathered);

    enc.set_input_array(x);
    enc.set_input_array(d_gather_buf);
    enc.set_output_array(x_gathered);

    dim3 grid(num_items);
    dim3 block(256);

    if (x_elem_size == 2) {
      enc.add_kernel_node(
          gather_rows_kernel<__half>,
          grid, block,
          gpu_ptr<__half>(x), gpu_ptr<__half>(x_gathered),
          gather_ptr, num_items, M, K);
    } else {
      enc.add_kernel_node(
          gather_rows_kernel<float>,
          grid, block,
          gpu_ptr<float>(x), gpu_ptr<float>(x_gathered),
          gather_ptr, num_items, M, K);
    }

    array out_gathered({batch_m, N}, out.dtype(), nullptr, {});
    out_gathered.set_data(cu::malloc_async(
        static_cast<size_t>(batch_m) * N * out_elem_size, enc));
    enc.add_temporary(out_gathered);

    CublasGemm gemm(
        enc.device(),
        out.dtype(),
        false,
        static_cast<uint64_t>(batch_m),
        static_cast<uint64_t>(K),
        static_cast<int64_t>(K),
        true,
        static_cast<uint64_t>(K),
        static_cast<uint64_t>(N_expert),
        static_cast<int64_t>(K),
        1, 0, 0);

    gemm.run(enc, out_gathered, x_gathered, w_dequant,
             {1}, {0}, {0}, 1.0f);

    enc.set_input_array(out_gathered);
    enc.set_input_array(d_scatter_buf);
    enc.set_output_array(out);

    if (out_elem_size == 2) {
      enc.add_kernel_node(
          scatter_rows_kernel<__half>,
          grid, block,
          gpu_ptr<__half>(out_gathered), gpu_ptr<__half>(out),
          scatter_ptr, num_items, M, N);
    } else {
      enc.add_kernel_node(
          scatter_rows_kernel<float>,
          grid, block,
          gpu_ptr<float>(out_gathered), gpu_ptr<float>(out),
          scatter_ptr, num_items, M, N);
    }
  }

  enc.end_direct_launch();
}

} // namespace mlx::core
