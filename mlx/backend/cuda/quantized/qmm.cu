// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm.h"
#include "mlx/dtype_utils.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

// clang-format off

namespace cute {

template <typename A, typename B>
struct F32FMA {
  using C = float;
  using D = float;
  using DRegisters = D[1];
  using ARegisters = A[1];
  using BRegisters = B[1];
  using CRegisters = C[1];
  CUTE_HOST_DEVICE static void fma(D& d, const A& a, const B& b, const C& c) {
    d = float(a) * float(b) + c;
  }
};

template <typename A, typename B>
struct MMA_Traits<F32FMA<A,B>> {
  using ValTypeD = float;
  using ValTypeA = A;
  using ValTypeB = B;
  using ValTypeC = float;
  using Shape_MNK = Shape<_1,_1,_1>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape<_1,_1>>;
  using BLayout = Layout<Shape<_1,_1>>;
  using CLayout = Layout<Shape<_1,_1>>;
};

} // namespace cute

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cute_gemm {

using namespace cute;

template <int PackFactor, typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant,
          typename AStride, typename ASmemLayout, typename TiledCopyA,
          typename BStride, typename BSmemLayout, typename TiledCopyB,
          typename SLayout, typename CStride, typename TiledMma>
__global__ void qmm_impl(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    const Quant* B,   BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    const Element* S, const Element* Z, SLayout S_layout,
    Element* C, CStride dC, TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // A and C tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A), select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C), select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,l_coord); // (M,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord

  // Shared memory buffers.
  __shared__ Element smemA[cosize_v<ASmemLayout>];
  __shared__ Element smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

  // Partition the copying of A tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);   // (ACPY,ACPY_M,ACPY_K)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Accumulators.
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // Predicates for m bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)),
                                  Stride<_1,_0>{});                       // (ACPY_M,ACPY_K)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (BLK_M,BLK_N)
  Tensor tAcA = thr_copy_a.partition_S(cA);                               // (ACPY,ACPY_M,ACPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);                                  // (MMA,MMA_M,MMA_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }

  if constexpr (PackFactor == 1) {
    // ===== INT8 path (original CuTe-based pipeline, unchanged) =====
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
    CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));

    Tensor mB_nkl = make_tensor(make_gmem_ptr(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
    Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout);                      // (N,(gs,K/gs),L)
    Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout);                      // (N,(gs,K/gs),L)

    Tensor mB = mB_nkl(_,_,l_coord); // (N,K)
    Tensor mS = mS_nkl(_,_,l_coord); // (N,(gs,K/gs))
    Tensor mZ = mZ_nkl(_,_,l_coord); // (N,(gs,K/gs))

    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
    Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
    Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

    ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
    Tensor tBgB = thr_copy_b.partition_S(gB);       // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);       // (BCPY,BCPY_N,BCPY_K)
    Tensor tBrB = make_fragment_like(tBsB);         // (BCPY,BCPY_N,BCPY_K)
    Tensor tBrBq = make_fragment_like<Quant>(tBsB); // (BCPY,BCPY_N,BCPY_K)
    Tensor tBgS = thr_copy_b.partition_S(gS);       // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBgZ = thr_copy_b.partition_S(gZ);       // (BCPY,BCPY_N,BCPY_K,k)

    // Copy gmem to rmem for k_tile=0.
    copy_if(copy_a, tApA, tAgA(_,_,_,0), tArA);
    copy(copy_b, tBgB(_,_,_,0), tBrBq);

    auto K_TILE_MAX = size<3>(tAgA);

    // Main loop.
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
      __syncthreads();

      // Dequantize B and then copy A/B to smem.
      Tensor scale = tBgS(_,_,_,k_tile);
      Tensor zero_point = tBgZ(_,_,_,k_tile);
      for (int i = 0; i < size(tBrB); ++i) {
        tBrB(i) = Element(float(static_cast<int>(Quant(tBrBq(i)))) * float(scale(i)) + float(zero_point(i)));
      }
      copy(tArA, tAsA);
      copy(tBrB, tBsB);
      __syncthreads();

      // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors.
      int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
      copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tArA);
      copy(copy_b, tBgB(_,_,_,k_tile_next), tBrBq);

      // Compute gemm on mma-partitioned smem.
      gemm(mma, tCsA, tCsB, tCrC);
    }
  } else {
    // ===== INT4 path: Optimized with shared memory staging + prefetch =====
    //
    // Optimization strategy (3 fixes):
    //
    // 1. SHARED MEMORY STAGING for packed B:
    //    Instead of each thread independently loading from global memory,
    //    all 128 threads cooperatively load packed bytes into a shared memory
    //    buffer using coalesced uint32_t loads. Then each thread reads from
    //    shared memory (bank-conflict-free) for nibble extraction.
    //
    // 2. SCALE/BIAS SHARED MEMORY STAGING:
    //    Cooperative load of scale/bias into smem eliminates L2 contention
    //    from 128 threads independently hitting the same cache lines.
    //
    // 3. PREFETCH PIPELINE:
    //    Load next tile's packed bytes into smem staging buffer AFTER gemm's
    //    data is in smem, overlapping the global memory loads with compute.
    //    (Unlike INT8, we can't prefetch into registers since we need smem
    //    staging for coalescing. Instead, we use double-buffered smem.)

    constexpr int BN = decltype(size<1>(cta_tiler))::value;  // 128
    constexpr int BK = decltype(size<2>(cta_tiler))::value;  // 64
    constexpr int BKP = BK / PackFactor;                     // 32

    constexpr int GROUP_SIZE = decltype(get<0>(get<1>(shape(S_layout))))::value;

    int K_val = size<2>(shape_MNKL);
    int N_val = size<1>(shape_MNKL);
    int Kp = K_val / PackFactor;
    int groups_per_row = K_val / GROUP_SIZE;

    const uint8_t* B_batch = reinterpret_cast<const uint8_t*>(B) + l_coord * N_val * Kp;
    int b_n_start = n_coord * BN;

    const Element* S_batch = S + l_coord * N_val * groups_per_row;
    const Element* Z_batch = Z + l_coord * N_val * groups_per_row;

    // ---- Shared memory for scale/bias staging ----
    constexpr int GROUPS_PER_TILE = BK / GROUP_SIZE;
    constexpr int TOTAL_GROUPS = BN * GROUPS_PER_TILE;
    __shared__ Element smemS[TOTAL_GROUPS];
    __shared__ Element smemZ[TOTAL_GROUPS];

    // ---- Shared memory for packed B staging ----
    // BN * BKP bytes = 128 * 32 = 4096 bytes. Store as uint32_t for alignment.
    constexpr int TOTAL_PACKED_U32 = (BN * BKP) / 4;  // 1024
    __shared__ uint32_t smemBpacked[TOTAL_PACKED_U32];

    auto K_TILE_MAX = size<3>(tAgA);

    // ---- Helper: coalesced load of packed B tile into shared memory ----
    // 128 threads cooperatively load TOTAL_PACKED_U32 uint32_t values.
    // Each thread loads TOTAL_PACKED_U32/128 values with stride-128 pattern
    // for perfect coalescing (consecutive threads → consecutive addresses).
    //
    // Memory layout: B is row-major, N rows of Kp bytes each.
    // Tile covers rows [b_n_start, b_n_start+BN) and columns [k_tile*BKP, (k_tile+1)*BKP).
    // We load into smemBpacked with the SAME row-major layout: row n has BKP bytes
    // at smemBpacked[n * BKP/4 .. (n+1) * BKP/4 - 1].
    //
    // For coalescing: consecutive threads should access consecutive global addresses.
    // B rows have stride Kp (which may be >> BKP), so we load row-by-row.
    // With BKP/4 = 8 uint32_t per row and 128 threads, we process 128/8 = 16 rows
    // per iteration. BN/16 = 8 iterations total.
    auto load_packed_tile_to_smem = [&](int k_tile) {
      int b_kp_start = k_tile * BKP;
      constexpr int BKP_U32 = BKP / 4;  // 8 uint32_t per row

      // Thread mapping: thread_idx = n_local_offset * BKP_U32 + kp_u32
      // where n_local_offset = thread_idx / BKP_U32, kp_u32 = thread_idx % BKP_U32
      // This gives 128/8 = 16 rows per iteration, and BN/16 = 8 iterations.
      constexpr int ROWS_PER_ITER = 128 / BKP_U32;  // 16
      constexpr int NUM_ITERS = BN / ROWS_PER_ITER;  // 8

      int kp_u32 = thread_idx % BKP_U32;
      int n_base = thread_idx / BKP_U32;

      CUTE_UNROLL
      for (int iter = 0; iter < NUM_ITERS; ++iter) {
        int n_local = n_base + iter * ROWS_PER_ITER;
        int n_global = b_n_start + n_local;

        // Global memory load: 16 threads read consecutive uint32_t within same row
        // → coalesced 64-byte transaction (16 threads * 4 bytes).
        // With 8 such groups (for 8 different kp_u32 values... wait, no.)
        // Actually: threads 0..7 have kp_u32=0..7 and n_base=0 (first 8 threads)
        // threads 8..15 have kp_u32=0..7 and n_base=1
        // So 8 consecutive threads (same n_base) read the full row → 32-byte transaction.
        // Then next 8 threads read next row → another 32-byte transaction.
        // That's still efficient: 128-byte sector utilization.
        uint32_t val = *reinterpret_cast<const uint32_t*>(
            B_batch + n_global * Kp + b_kp_start + kp_u32 * 4);

        smemBpacked[n_local * BKP_U32 + kp_u32] = val;
      }
    };

    // ---- Prefetch A for k_tile=0 ----
    copy_if(copy_a, tApA, tAgA(_,_,_,0), tArA);

    // ---- Main loop ----
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
      __syncthreads();

      // Phase 1: Cooperative load of packed B and scales into shared memory.
      load_packed_tile_to_smem(k_tile);

      // Load scales for this tile.
      {
        int k_global_base = k_tile * BK;
        int g_base = k_global_base / GROUP_SIZE;
        for (int idx = thread_idx; idx < TOTAL_GROUPS; idx += blockDim.x) {
          int n_local = idx / GROUPS_PER_TILE;
          int g_local = idx % GROUPS_PER_TILE;
          int n_global = b_n_start + n_local;
          int g_global = g_base + g_local;
          smemS[idx] = S_batch[n_global * groups_per_row + g_global];
          smemZ[idx] = Z_batch[n_global * groups_per_row + g_global];
        }
      }

      __syncthreads();

      // Phase 2: Each thread reads packed bytes from smem, extracts nibbles,
      // dequantizes using scales from smem, and writes to sB.
      // This replaces the scattered global memory reads with fast smem reads.
      //
      // Work distribution: same as load but now each thread processes its
      // assigned elements and writes to sB.
      constexpr int BKP_U32 = BKP / 4;
      constexpr int ROWS_PER_ITER = 128 / BKP_U32;
      constexpr int NUM_ITERS = BN / ROWS_PER_ITER;

      int kp_u32 = thread_idx % BKP_U32;
      int n_base = thread_idx / BKP_U32;

      CUTE_UNROLL
      for (int iter = 0; iter < NUM_ITERS; ++iter) {
        int n_local = n_base + iter * ROWS_PER_ITER;
        uint32_t packed4 = smemBpacked[n_local * BKP_U32 + kp_u32];

        CUTE_UNROLL
        for (int b = 0; b < 4; ++b) {
          uint8_t packed_byte = (packed4 >> (b * 8)) & 0xFF;
          int kp_local = kp_u32 * 4 + b;  // packed byte index in tile row

          int k_in_tile_lo = kp_local * 2;       // unpacked k index [0..BK-1]
          int k_in_tile_hi = kp_local * 2 + 1;

          int g_lo = k_in_tile_lo / GROUP_SIZE;
          int g_hi = k_in_tile_hi / GROUP_SIZE;

          float s = float(smemS[n_local * GROUPS_PER_TILE + g_lo]);
          float z = float(smemZ[n_local * GROUPS_PER_TILE + g_lo]);

          // Low nibble (bits 0-3).
          sB(n_local, k_in_tile_lo) = Element(float(packed_byte & 0xF) * s + z);

          // High nibble (bits 4-7).
          if (g_hi != g_lo) {
            s = float(smemS[n_local * GROUPS_PER_TILE + g_hi]);
            z = float(smemZ[n_local * GROUPS_PER_TILE + g_hi]);
          }
          sB(n_local, k_in_tile_hi) = Element(float(packed_byte >> 4) * s + z);
        }
      }

      // Copy A from registers to shared memory.
      copy(tArA, tAsA);
      __syncthreads();

      // Prefetch next A tile into registers (overlaps with gemm).
      int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
      copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tArA);

      // Compute gemm on mma-partitioned smem.
      gemm(mma, tCsA, tCsB, tCrC);
    }
  }

  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if (get<0>(tCcC(i)) < m_max_coord) {
      tCgC(i) = Element(tCrC(i));
    }
  }
}

template <typename Element, typename GroupSize, typename F>
inline auto dispatch_swizzle(F&& f) {
  if constexpr (sizeof(Element) == 4) {
    if constexpr (GroupSize::value <= 32) {
      f(Swizzle<3,2,3>{});
    } else {
      f(Swizzle<3,3,3>{});
    }
  } else {
    if constexpr (GroupSize::value <= 32) {
      f(Swizzle<2,3,3>{});
    } else {
      f(Swizzle<3,3,3>{});
    }
  }
}

template <typename Element, typename F>
inline auto dispatch_mma(bool is_sm80, F&& f) {
  if (is_sm80) {
    if constexpr (std::is_same_v<Element, float>) {
      f(make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                       Layout<Shape<_1,_4,_1>>{},
                       Tile<_16,_32,_8>{}));
      return;
    } else if constexpr (std::is_same_v<Element, cute::half_t>) {
      f(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                       Layout<Shape<_1,_4,_1>>{},
                       Tile<_16,_32,_16>{}));
      return;
    } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
      f(make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                       Layout<Shape<_1,_4,_1>>{},
                       Tile<_16,_32,_16>{}));
      return;
    }
  }
  f(make_tiled_mma(F32FMA<Element, Element>{},
                   Layout<Shape<_16,_8,_1>>{}));
}

template <int PackFactor, typename GroupSize, typename Element, typename Quant, typename F>
void qmm(
    int m, int n, int k, int l,
    GroupSize group_size,
    const Element* A,
    const Quant* B,
    const Element* S,
    const Element* Z,
    Element* C,
    bool is_sm80,
    F&& launch_kernel) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M,N,K,L)

  // Define TN strides (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k); // (dM,dK,dL)
  // For INT4 (PackFactor=2): B has k/2 bytes per row.
  // For INT8 (PackFactor=1): B has k bytes per row.
  int kp = k / PackFactor;
  auto dB = make_stride(kp, Int<1>{}, n * kp); // (dN,dKp,dL)
  auto dC = make_stride(n, Int<1>{}, m * n); // (dM,dN,dL)

  // Define layout of scales (mixed).
  auto S_layout = make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, make_stride(Int<0>{}, Int<1>{}), n * k / group_size));

  // Define CTA tile sizes (static).
  auto bM = Int<16>{};
  auto bN = Int<128>{};
  auto bK = Int<max(64,group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                     Layout<Shape< _1,_8>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, Quant>{},
                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<32/sizeof_bits<Quant>::value>>>{});

  // Define the smem layouts (static).
  dispatch_swizzle<Element, GroupSize>([&](auto swizzle) {
    auto swizzle_atom = composition(swizzle,
                                    Layout<Shape<_8,GroupSize>,
                                           Stride<GroupSize,_1>>{});
    auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK));
    auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK));

    // Create tiled MMA.
    dispatch_mma<Element>(is_sm80, [&](auto mma) {
      // Launch kernel.
      auto* kernel = &qmm_impl<
          PackFactor,
          decltype(prob_shape), decltype(cta_tiler),
          Element, Quant,
          decltype(dA), decltype(sA_layout), decltype(copy_a),
          decltype(dB), decltype(sB_layout), decltype(copy_b),
          decltype(S_layout), decltype(dC), decltype(mma)>;
      dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
      dim3 block_dims(size(mma));
      void* args[] = {
        &prob_shape, &cta_tiler,
        &A, &dA, &sA_layout, &copy_a,
        &B, &dB, &sB_layout, &copy_b,
        &S, &Z, &S_layout,
        &C, &dC, &mma};
      launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, 0, args);
    });
  });
}

} // namespace cute_gemm

// clang-format on

namespace mlx::core {

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format(
            "[{0}] Unsupported dtype: {1}.", tag, dtype_to_string(dtype)));
  }
}

template <typename F>
inline void dispatch_quant_types(int bits, const char* tag, F&& f) {
  // Both INT4 and INT8 use uint8_t as storage type.
  // INT4 nibble extraction is handled in the kernel via PackFactor.
  if (bits == 4 || bits == 8) {
    f.template operator()<uint8_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("[{0}] {1}-bit quantization is not supported.", tag, bits));
  }
}

template <typename F>
inline void dispatch_pack_factor(int bits, const char* tag, F&& f) {
  if (bits == 4) {
    f(cute::Int<2>{});
  } else if (bits == 8) {
    f(cute::Int<1>{});
  } else {
    throw std::invalid_argument(
        fmt::format("[{0}] {1}-bit quantization is not supported.", tag, bits));
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 16) {
    f(cute::Int<16>{});
  } else if (group_size == 32) {
    f(cute::Int<32>{});
  } else if (group_size == 64) {
    f(cute::Int<64>{});
  } else {
    throw std::invalid_argument(
        fmt::format("[{0}] Group size {1} is not supported.", tag, group_size));
  }
}

void cute_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder) {
  const char* tag = "[quantized_matmul]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);
  if (n % 128 != 0) {
    throw std::runtime_error(
        fmt::format("[{0}] N must be multiples of 128.", tag));
  }
  if (k % 64 != 0) {
    throw std::runtime_error(
        fmt::format("[{0}] K must be multiples of 64.", tag));
  }
  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types(bits, tag, [&]<typename Quant>() {
      dispatch_groups(group_size, tag, [&](auto group_size) {
        dispatch_pack_factor(bits, tag, [&](auto pack_factor_tag) {
          constexpr int PackFactor = decltype(pack_factor_tag)::value;
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          encoder.set_input_array(biases);
          encoder.set_output_array(out);
          cute_gemm::qmm<PackFactor>(
              m,
              n,
              k,
              l,
              group_size,
              gpu_ptr<Element>(x),
              gpu_ptr<Quant>(w),
              gpu_ptr<Element>(scales),
              gpu_ptr<Element>(biases),
              gpu_ptr<Element>(out),
              encoder.device().compute_capability_major() >= 8,
              [&](auto* kernel,
                  dim3 num_blocks,
                  dim3 block_dims,
                  uint32_t smem_bytes,
                  void** args) {
                encoder.add_kernel_node(
                    kernel, num_blocks, block_dims, smem_bytes, args);
              });
        });
      });
    });
  });
}

} // namespace mlx::core
