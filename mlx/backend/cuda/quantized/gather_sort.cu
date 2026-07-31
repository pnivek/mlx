// Copyright © 2025 Apple Inc.
// On-device counting sort + output scatter for the SM120 grouped MoE GEMM.
// Extracted from the pre-rebase gather_qmm.cu.

#include "mlx/backend/cuda/quantized/gather_qmm_sm120.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype_utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mlx::core {

namespace {

// Vectorized copy helper: use uint4 (16-byte) loads when aligned.
template <typename T>
__device__ void
vectorized_copy(const T* src_ptr, T* dst_ptr, size_t total_elems) {
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

// Step 2: Exclusive prefix sum on expert_counts -> expert_offsets.
// Also initializes write_pos = expert_offsets for the scatter step.
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

} // namespace

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

  // Sort buffer: [counts(E)|offsets(E)|write_pos(E)]
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

  array sorted_lhs({B}, uint32, nullptr, {});
  sorted_lhs.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(sorted_lhs);

  array sorted_rhs({B}, uint32, nullptr, {});
  sorted_rhs.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(sorted_rhs);

  array sorted_perm({B}, uint32, nullptr, {});
  sorted_perm.set_data(cu::malloc_async(B * sizeof(uint32_t), enc));
  enc.add_temporary(sorted_perm);

  enc.set_output_array(sort_buf);
  {
    int threads = 256;
    int blocks = (E + threads - 1) / threads;
    enc.add_kernel_node(
        zero_uint32_kernel, dim3(blocks), dim3(threads), d_counts, E);
  }

  enc.set_input_array(rhs_flat);
  enc.set_output_array(sort_buf);
  {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    enc.add_kernel_node(
        count_experts_kernel,
        dim3(blocks),
        dim3(threads),
        gpu_ptr<uint32_t>(rhs_flat),
        B,
        d_counts);
  }

  enc.set_input_array(sort_buf);
  enc.set_output_array(sort_buf);
  enc.add_kernel_node(
      prefix_sum_kernel, dim3(1), dim3(1), d_counts, E, d_offsets, d_write_pos);

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
        dim3(blocks),
        dim3(threads),
        gpu_ptr<uint32_t>(rhs_flat),
        gpu_ptr<uint32_t>(lhs_flat),
        B,
        d_write_pos,
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
        dim3(B),
        dim3(256),
        gpu_ptr<__half>(src),
        gpu_ptr<__half>(dst),
        gpu_ptr<uint32_t>(sorted_perm),
        B,
        M,
        N);
  } else {
    enc.add_kernel_node(
        scatter_rows_kernel<float>,
        dim3(B),
        dim3(256),
        gpu_ptr<float>(src),
        gpu_ptr<float>(dst),
        gpu_ptr<uint32_t>(sorted_perm),
        B,
        M,
        N);
  }
}

} // namespace mlx::core
