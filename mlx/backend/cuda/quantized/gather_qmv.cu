// Copyright © 2025 Apple Inc.

// Fused GatherQMV: quantized vector-matrix multiply with on-device index
// lookup for MoE expert routing. Eliminates the host-side synchronize +
// group-by-expert overhead of the non-fused GatherQMM path.
//
// For each output index i in [0, B):
//   out[i] = x[lhs_indices[i]] @ dequant(w[rhs_indices[i]]).T
//
// Grid: (M, ceil(N/rows_per_block), B)
// Each threadblock independently reads its expert_id from rhs_indices[blockIdx.z]
// and its x_batch from lhs_indices[blockIdx.z], then runs the standard QMV.

#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/gather_qmv.h"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

static constexpr int gather_rows_per_block = 8;

// On-device index lookup: use lhs_indices/rhs_indices to compute pointers.
// x is indexed by lhs_indices[blockIdx.z], w/scales by rhs_indices[blockIdx.z].
template <typename T>
__device__ void gather_adjust_offsets(
    const T*& x,
    const uint32_t*& w,
    const uint8_t*& scales,
    T*& y,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int x_stride, // M * K elements per x batch
    int w_stride, // elements per expert in w (N * K_packed)
    int s_stride, // elements per expert in scales
    int y_stride) { // M * N elements per output batch
  uint32_t batch_idx = cg::this_grid().block_index().z;
  uint32_t x_idx = lhs_indices[batch_idx];
  uint32_t w_idx = rhs_indices[batch_idx];
  x += x_idx * x_stride;
  w += w_idx * w_stride;
  scales += w_idx * s_stride;
  y += batch_idx * y_stride;
}

// The QMV implementation is identical to fp_qmv_impl in qmv.cu.
// We duplicate it here to keep the gather kernel self-contained and avoid
// cross-file template instantiation issues.
template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__device__ void gather_qmv_impl(
    const uint32_t* mat,
    const uint8_t* scales_,
    const T* vec,
    T* out,
    int rows,
    int cols) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int vals_per_item = bits == 8 ? 4 : 8;
  constexpr int nv_per_thread = vals_per_item * n_per_thread;
  auto g_idx = block.group_index();
  auto t_idx = block.thread_index();
  int row = g_idx.y * rows_per_block + t_idx.y;

  vec += g_idx.x * cols;
  out += g_idx.x * rows;

  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scales = (ScaleType*)(scales_);
  auto packed_cols = cols / vals_per_item;

  if (row < rows) {
    constexpr int scales_per_step = std::max(nv_per_thread / group_size, 1);
    constexpr int scale_step = (WARP_SIZE * nv_per_thread) / group_size;
    constexpr int n_per_step = n_per_thread / scales_per_step;
    scales += row * (cols / group_size) +
        (warp.thread_rank() * nv_per_thread) / group_size;
    float sum = 0.0f;
    for (int col = n_per_thread * warp.thread_rank(); col < packed_cols;
         col += (WARP_SIZE * n_per_thread)) {
      auto local_vec =
          unsafe_load_vector<nv_per_thread>(vec + vals_per_item * col, 0);
      auto local_mat =
          unsafe_load_vector<n_per_thread>(mat + row * packed_cols + col, 0);
#pragma unroll
      for (int i = 0; i < scales_per_step; ++i) {
        float2 local_sum = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < n_per_step; ++j) {
          int k = n_per_step * i + j;
          if constexpr (bits == 8) {
            auto v = dequant_fp8(local_mat[k]);
            local_sum.x +=
                v.x * static_cast<float>(local_vec[vals_per_item * k]);
            local_sum.x +=
                v.y * static_cast<float>(local_vec[vals_per_item * k + 1]);
            local_sum.y +=
                v.z * static_cast<float>(local_vec[vals_per_item * k + 2]);
            local_sum.y +=
                v.w * static_cast<float>(local_vec[vals_per_item * k + 3]);
          } else {
            auto v = dequant_fp4(local_mat[k]);
            local_sum.x +=
                v.x * static_cast<float>(local_vec[vals_per_item * k]);
            local_sum.y +=
                v.y * static_cast<float>(local_vec[vals_per_item * k + 1]);
            local_sum.x +=
                v.z * static_cast<float>(local_vec[vals_per_item * k + 2]);
            local_sum.y +=
                v.w * static_cast<float>(local_vec[vals_per_item * k + 3]);

            v = dequant_fp4(local_mat[k] >> 16);
            local_sum.x +=
                v.x * static_cast<float>(local_vec[vals_per_item * k + 4]);
            local_sum.y +=
                v.y * static_cast<float>(local_vec[vals_per_item * k + 5]);
            local_sum.x +=
                v.z * static_cast<float>(local_vec[vals_per_item * k + 6]);
            local_sum.y +=
                v.w * static_cast<float>(local_vec[vals_per_item * k + 7]);
          }
        }
        sum += (local_sum.x + local_sum.y) * float(scales[i]);
      }
      scales += scale_step;
    }

    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0) {
      out[row] = static_cast<T>(sum);
    }
  }
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void fp_gather_qmv_kernel(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int rows, // N (output features per expert)
    int cols, // K (input features)
    int x_stride, // M * K
    int w_stride, // N * K_packed
    int s_stride, // scales elements per expert
    int y_stride) { // M * N
  gather_adjust_offsets<T>(
      vec, mat, scales, out,
      lhs_indices, rhs_indices,
      x_stride, w_stride, s_stride, y_stride);
  gather_qmv_impl<T, rows_per_block, n_per_thread, bits, group_size,
                   use_mx_scale>(mat, scales, vec, out, rows, cols);
}

template <typename F>
void dispatch_1_2_4(int n, F&& f) {
  switch (n) {
    case 1:
      f(std::integral_constant<int, 1>{});
      break;
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
  }
}

void fp_gather_qmv(
    const array& mat,
    const array& scales,
    const array& vec,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    int B,
    CommandEncoder& encoder) {
  encoder.set_input_array(mat);
  encoder.set_input_array(scales);
  encoder.set_input_array(vec);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  dispatch_float_types(out.dtype(), "gather_qmv", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      dim3 block_dims{WARP_SIZE, gather_rows_per_block};
      uint32_t blocks_y = (N + gather_rows_per_block - 1) / gather_rows_per_block;

      const uint32_t* mat_ptr = gpu_ptr<uint32_t>(mat);
      const T* vec_ptr = gpu_ptr<T>(vec);

      // Compute strides for expert indexing.
      // mat shape: (E, N, K_packed), vec shape: (..., M, K)
      // mat.strides()[0] = N * K_packed (elements per expert)
      int w_stride = static_cast<int>(mat.strides()[0]);
      int s_stride = static_cast<int>(scales.strides()[0]);
      int x_stride = M * K; // elements per x batch (assuming contiguous last 2 dims)
      int y_stride = M * N; // elements per output batch

      // Determine alignment for n_per_thread dispatch.
      int n = 1;
      if (K % 32 == 0 && cu::is_aligned<4>(mat_ptr) &&
          ((bits == 4 && cu::is_aligned<8>(vec_ptr)) ||
           cu::is_aligned<4>(vec_ptr))) {
        n = 4;
      } else if (
          cu::is_aligned<2>(mat_ptr) &&
          ((bits == 4 && cu::is_aligned<4>(vec_ptr)) ||
           cu::is_aligned<2>(vec_ptr))) {
        n = 2;
      }

      dispatch_1_2_4(n, [&](auto n_val) {
        auto kernel =
            fp_gather_qmv_kernel<T, gather_rows_per_block, n_val.value, 4, 32, true>;
        if (bits == 8) {
          kernel =
              fp_gather_qmv_kernel<T, gather_rows_per_block, n_val.value, 8, 32, true>;
        } else if (group_size == 16) {
          kernel =
              fp_gather_qmv_kernel<T, gather_rows_per_block, n_val.value, 4, 16, false>;
        }
        encoder.add_kernel_node(
            kernel,
            {static_cast<uint32_t>(M), blocks_y, static_cast<uint32_t>(B)},
            block_dims,
            0,
            mat_ptr,
            gpu_ptr<uint8_t>(scales),
            vec_ptr,
            gpu_ptr<T>(out),
            gpu_ptr<uint32_t>(lhs_indices),
            gpu_ptr<uint32_t>(rhs_indices),
            N,
            K,
            x_stride,
            w_stride,
            s_stride,
            y_stride);
      });
    }
  });
}

} // namespace mlx::core::cu
