// Varlen FlashAttention + Fused RoPE Backward CUDA/CuTe kernel for SM75 (Turing)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "mha_common.cuh"
#include "mha_ops.cuh"

using namespace cute;

// =================================================================================================
// 1. Device Kernel & Helpers
// =================================================================================================

// Apply forward RoPE in registers (32x8 thread layout, 8 half elements per step).
template <int TILE_K, typename FragFP16>
__device__ __forceinline__ void apply_rope_fwd_32x8(
    FragFP16& rTile,                              // ((1,8), 2, 1) [FP16] - register tile to rotate
    const cutlass::half_t* __restrict__ cos_ptr,  // (L_max, 64) [FP16] - RoPE cos table
    const cutlass::half_t* __restrict__ sin_ptr,  // (L_max, 64) [FP16] - RoPE sin table
    int tile_start_pos,                           // scalar [int] - sequence start position of tile
    int offset,                                   // scalar [int] - sequence RoPE offset
    int valid_rows,                               // scalar [int] - remaining valid rows in sequence
    int row_m0_g2r,                               // scalar [int] - row index for step m=0
    int row_m1_g2r,                               // scalar [int] - row index for step m=1
    int col_vec_g2r,                              // scalar [int] - 8-column vector group index (0..7)
    bool is_second_half_g2r                       // scalar [bool] - true for upper columns [32..63]
) {
    CUTE_UNROLL
    for (int m = 0; m < size<1>(rTile); ++m) {
        const int row_in_tile = (m == 0) ? row_m0_g2r : row_m1_g2r;
        const bool valid_row = (row_in_tile < valid_rows);
        const int rope_pos = valid_row ? (offset + tile_start_pos + row_in_tile) : 0;
        const __half2* cos_h2 = reinterpret_cast<const __half2*>(
            cos_ptr + rope_pos * TILE_K + col_vec_g2r * 8);
        const __half2* sin_h2 = reinterpret_cast<const __half2*>(
            sin_ptr + rope_pos * TILE_K + col_vec_g2r * 8);
        apply_rope_fwd_8elem_shfl(&rTile(0, m, 0), cos_h2, sin_h2, is_second_half_g2r);
        if (!valid_row) {
            clear(rTile(_, m, 0));
        }
    }
}

// Apply backward RoPE in-place on FP32 register fragment (16x16 thread layout, 4 floats per step).
template <int TILE_K, typename FragF32>
__device__ __forceinline__ void apply_rope_bwd_f32_16x16(
    FragF32& rF32,                                // ((1,4), 4, 1) [FP32] - gradient before RoPE (in-place)
    const cutlass::half_t* __restrict__ cos_ptr,  // (L_max, 64) [FP16] - RoPE cos table
    const cutlass::half_t* __restrict__ sin_ptr,  // (L_max, 64) [FP16] - RoPE sin table
    int tile_start_pos,                           // scalar [int] - sequence start position of tile
    int offset,                                   // scalar [int] - sequence RoPE offset
    int valid_rows,                               // scalar [int] - remaining valid rows in sequence
    int row_in_step_f32,                          // scalar [int] - row within 16-row group (tid >> 4)
    int col_vec_f32,                              // scalar [int] - 4-column group index (tid & 15)
    bool is_second_half_f32                       // scalar [bool] - true for upper columns [32..63]
) {
    CUTE_UNROLL
    for (int m = 0; m < size<1>(rF32); ++m) {
        const int row_in_tile = m * 16 + row_in_step_f32;
        const bool valid_row = (row_in_tile < valid_rows);
        const int rope_pos = valid_row ? (offset + tile_start_pos + row_in_tile) : 0;
        const __half2* cos_h2 = reinterpret_cast<const __half2*>(
            cos_ptr + rope_pos * TILE_K + col_vec_f32 * 4);
        const __half2* sin_h2 = reinterpret_cast<const __half2*>(
            sin_ptr + rope_pos * TILE_K + col_vec_f32 * 4);

        float c_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (valid_row) {
            const __half2 c01 = cos_h2[0];
            const __half2 c23 = cos_h2[1];
            const __half2 s01 = sin_h2[0];
            const __half2 s23 = sin_h2[1];
            c_arr[0] = __low2float(c01);
            c_arr[1] = __high2float(c01);
            c_arr[2] = __low2float(c23);
            c_arr[3] = __high2float(c23);
            s_arr[0] = __low2float(s01);
            s_arr[1] = __high2float(s01);
            s_arr[2] = __low2float(s23);
            s_arr[3] = __high2float(s23);
        }

        // All warp lanes must execute __shfl_xor_sync unconditionally before valid_row guard.
        CUTE_UNROLL
        for (int v = 0; v < size<0>(rF32); ++v) {
            const float g_val = rF32(v, m, 0);
            const float partner_g = __shfl_xor_sync(0xffffffff, g_val, 8);
            if (valid_row) {
                rF32(v, m, 0) = is_second_half_f32
                    ? fmaf(g_val, c_arr[v], -partner_g * s_arr[v])
                    : fmaf(g_val, c_arr[v], partner_g * s_arr[v]);
            }
        }
    }
}


template <
    int TILE_M = 64,
    int TILE_N = 64,
    int TILE_K = 64,
    int NUM_THREADS = 256,
    bool HAS_ROPE = true,
    bool IS_CAUSAL = false
>
__global__ void __launch_bounds__(NUM_THREADS, 1)
kernel_fused_mha_varlen_bwd(
    const cutlass::half_t* __restrict__ dO_ptr,   // (H_q, Total_Lq, 64) [FP16] - output gradient dO
    const cutlass::half_t* __restrict__ Q_ptr,    // (H_q, Total_Lq, 64) [FP16] - unrotated query Q
    const cutlass::half_t* __restrict__ K_ptr,    // (H_kv, Total_Lk, 64) [FP16] - unrotated key K
    const cutlass::half_t* __restrict__ V_ptr,    // (H_kv, Total_Lk, 64) [FP16] - value V
    const cutlass::half_t* __restrict__ O_ptr,    // (H_q, Total_Lq, 64) [FP16] - forward output O
    const float* __restrict__ LSE_ptr,            // (H_q, Total_Lq)     [FP32] - forward LSE
    float* __restrict__ dQ_fp32_ptr,              // (H_q, Total_Lq, 64) [FP32] - atomic dQ accumulator buffer
    cutlass::half_t* __restrict__ dK_ptr,         // (H_kv, Total_Lk, 64) [FP16] - output gradient dK
    cutlass::half_t* __restrict__ dV_ptr,         // (H_kv, Total_Lk, 64) [FP16] - output gradient dV
    const int32_t* __restrict__ cu_seqlens_q_ptr, // (B + 1,) [Int32] - cumulative Q lengths
    const int32_t* __restrict__ cu_seqlens_k_ptr, // (B + 1,) [Int32] - cumulative K/V lengths
    const int32_t* __restrict__ offsets_ptr,      // (B,) [Int32] - per-sequence RoPE offset (or nullptr)
    const cutlass::half_t* __restrict__ cos_ptr,  // (L_max, 64) [FP16] - RoPE cos table (or nullptr)
    const cutlass::half_t* __restrict__ sin_ptr,  // (L_max, 64) [FP16] - RoPE sin table (or nullptr)
    int H_q,                                      // scalar [int] - number of Q heads
    int H_kv,                                     // scalar [int] - number of K/V heads
    int Total_Lq,                                 // scalar [int] - total packed Q tokens
    int Total_Lk,                                 // scalar [int] - total packed K/V tokens
    float sm_scale                                // scalar [float] - 1 / sqrt(64) softmax scale
) {
    // 1. Coordinates, SMEM & tensor partitioning
    const int tid = static_cast<int>(threadIdx.x);
    const int block_k_idx = static_cast<int>(blockIdx.x);
    const int batch_idx = static_cast<int>(blockIdx.y);
    const int h_kv = static_cast<int>(blockIdx.z);

    // Read varlen sequence bounds from GMEM
    const int q_start_seq = cu_seqlens_q_ptr[batch_idx];
    const int q_end_seq = cu_seqlens_q_ptr[batch_idx + 1];
    const int lq_i = q_end_seq - q_start_seq;

    const int k_start_seq = cu_seqlens_k_ptr[batch_idx];
    const int k_end_seq = cu_seqlens_k_ptr[batch_idx + 1];
    const int lk_i = k_end_seq - k_start_seq;

    const int k_start_col = block_k_idx * TILE_N;
    if (k_start_col >= lk_i) {
        return; // Entire K tile is out of bounds
    }

    const int offset = (HAS_ROPE && offsets_ptr != nullptr) ? offsets_ptr[batch_idx] : 0;
    const int num_groups = H_q / H_kv;
    const int valid_k_rows = lk_i - k_start_col;

    // Dynamic shared memory layout (57,600 B <= 64 KB)
    extern __shared__ uint8_t smem_raw[];
    cutlass::half_t* sQ_ptr = reinterpret_cast<cutlass::half_t*>(smem_raw);
    cutlass::half_t* sK_ptr = sQ_ptr + TILE_M * TILE_K;                // [8 KB .. 16 KB)
    cutlass::half_t* sV_ptr = sK_ptr + TILE_N * TILE_K;                // [16 KB .. 24 KB)
    cutlass::half_t* sDO_ptr = sV_ptr + TILE_N * TILE_K;               // [24 KB .. 32 KB)
    cutlass::half_t* sP_ptr = sDO_ptr + TILE_M * TILE_K;               // [32 KB .. 40 KB)
    cutlass::half_t* sStaging_ptr = sP_ptr + TILE_M * TILE_N;          // [40 KB .. 48 KB)
    cutlass::half_t* sdS_ptr = sStaging_ptr + TILE_M * TILE_N;         // [48 KB .. 56 KB)
    float* sStaging_fp32_ptr = reinterpret_cast<float*>(sP_ptr);       // Reuse [32 KB .. 48 KB) (16 KB)
    float* sD_ptr = reinterpret_cast<float*>(sdS_ptr + TILE_M * TILE_N); // [56 KB .. 56.25 KB)

    // CuTe layouts, copy atoms, and TiledMMA types
    using Element = cutlass::half_t;
    using SmemLayoutFP16 = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_N>>, Stride<Int<TILE_N>, _1>>{}));
    using SmemLayoutMN = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_N>>, Stride<_1, Int<TILE_M>>>{}));
    using SmemLayoutF32 = decltype(composition(
        Swizzle<3, 2, 4>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_N>>, Stride<Int<TILE_N>, _1>>{}));

    using TiledCopyG2R = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        make_layout(Shape<_32, _8>{}, LayoutRight{}),
        make_layout(Shape<_1, _8>{}, LayoutRight{})));
    using TiledCopyF32 = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        make_layout(Shape<_16, _16>{}, LayoutRight{}),
        make_layout(Shape<_1, _4>{}, LayoutRight{})));
    using TiledCopyFP16_16x16 = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint2>, Element>{},
        make_layout(Shape<_16, _16>{}, LayoutRight{}),
        make_layout(Shape<_1, _4>{}, LayoutRight{})));

    using MMA_Op = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMA = decltype(make_tiled_mma(MMA_Op{}, Layout<Shape<_4, _2, _1>>{}));

    using S2RCopyAtomA_KMajor = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomA_MNMajor = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
    using S2RCopyAtomB_KMajor = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomB_MNMajor = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    // Swizzled K-major and MN-major SMEM views
    Tensor sQ = make_tensor(make_smem_ptr(sQ_ptr), SmemLayoutFP16{});
    Tensor sQ_mn = make_tensor(make_smem_ptr(sQ_ptr), SmemLayoutMN{});
    Tensor sK = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutFP16{});
    Tensor sK_mn = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutMN{});
    Tensor sV = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutFP16{});
    Tensor sDO = make_tensor(make_smem_ptr(sDO_ptr), SmemLayoutFP16{});
    Tensor sDO_mn = make_tensor(make_smem_ptr(sDO_ptr), SmemLayoutMN{});
    Tensor sP = make_tensor(make_smem_ptr(sP_ptr), SmemLayoutFP16{});
    Tensor sP_mn = make_tensor(make_smem_ptr(sP_ptr), SmemLayoutMN{});
    Tensor sStaging = make_tensor(make_smem_ptr(sStaging_ptr), SmemLayoutFP16{});
    Tensor sdS = make_tensor(make_smem_ptr(sdS_ptr), SmemLayoutFP16{});
    Tensor sdS_mn = make_tensor(make_smem_ptr(sdS_ptr), SmemLayoutMN{});
    Tensor sStaging_fp32 = make_tensor(make_smem_ptr(sStaging_fp32_ptr), SmemLayoutF32{});

    // GMEM tensors for K, V, dK, dV of this CTA
    const int64_t kv_tile_offset = static_cast<int64_t>(h_kv) * Total_Lk * TILE_K +
                                   static_cast<int64_t>(k_start_seq + k_start_col) * TILE_K;
    Tensor gK = make_tensor(make_gmem_ptr(K_ptr + kv_tile_offset),
                            make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                            make_stride(Int<TILE_K>{}, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(V_ptr + kv_tile_offset),
                            make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                            make_stride(Int<TILE_K>{}, _1{}));
    Tensor gDK = make_tensor(make_gmem_ptr(dK_ptr + kv_tile_offset),
                             make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                             make_stride(Int<TILE_K>{}, _1{}));
    Tensor gDV = make_tensor(make_gmem_ptr(dV_ptr + kv_tile_offset),
                             make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                             make_stride(Int<TILE_K>{}, _1{}));

    // Thread slices, fragments, and boundary predicates
    TiledCopyG2R tiled_copy_g2r;
    auto thr_copy_g2r = tiled_copy_g2r.get_thread_slice(tid);
    Tensor tKgK = thr_copy_g2r.partition_S(gK);
    Tensor tKsK = thr_copy_g2r.partition_D(sK);
    Tensor tVgV = thr_copy_g2r.partition_S(gV);
    Tensor tVsV = thr_copy_g2r.partition_D(sV);
    Tensor tDKgDK = thr_copy_g2r.partition_D(gDK);
    Tensor tDVgDV = thr_copy_g2r.partition_D(gDV);
    Tensor tQsQ = thr_copy_g2r.partition_D(sQ);
    Tensor tDOsDO = thr_copy_g2r.partition_D(sDO);
    Tensor tStaging_S = thr_copy_g2r.partition_S(sStaging);

    Tensor rK = make_fragment_like(tKsK);
    Tensor rV = make_fragment_like(tVsV);
    Tensor rQ = make_fragment_like(tQsQ);
    Tensor rDO = make_fragment_like(tDOsDO);

    Tensor cTile_g2r = thr_copy_g2r.partition_D(
        make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_N>{})));
    const int row_m0_g2r = get<0>(cTile_g2r(0, 0, 0));
    const int row_m1_g2r = get<0>(cTile_g2r(0, 1, 0));
    const int col_vec_g2r = tid & 7;
    const bool is_second_half_g2r = (col_vec_g2r >= 4);

    Tensor tKpK = make_tensor<bool>(make_shape(size<1>(tKgK), size<2>(tKgK)));
    tKpK(0, 0) = (row_m0_g2r < valid_k_rows);
    tKpK(1, 0) = (row_m1_g2r < valid_k_rows);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)));

    // FP32 staging (16x16 threads x 4 floats = uint4) for dQ/dK RoPE backward
    TiledCopyF32 tiled_copy_f32;
    auto thr_copy_f32 = tiled_copy_f32.get_thread_slice(tid);
    Tensor tStagingF32_S = thr_copy_f32.partition_S(sStaging_fp32);
    Tensor rF32 = make_fragment_like(tStagingF32_S);
    Tensor rF32_fp16 = make_tensor_like<Element>(rF32);
    const int row_in_step_f32 = tid >> 4;
    const int col_vec_f32 = tid & 15;
    const bool is_second_half_f32 = (col_vec_f32 >= 8);

    TiledCopyFP16_16x16 tiled_copy_fp16_16x16;
    auto thr_copy_fp16_16x16 = tiled_copy_fp16_16x16.get_thread_slice(tid);
    Tensor tStaging_D_16x16 = thr_copy_fp16_16x16.partition_D(sStaging);

    // MMA, accumulators, and S2R partitions
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCrC = thr_mma.partition_fragment_C(sP); // Reused across GEMMs 1, 3, 5
    Tensor tCrDK = make_tensor_like(tCrC);
    Tensor tCrDV = make_tensor_like(tCrC);
    Tensor rC_fp16 = make_tensor_like<Element>(tCrC);

    Tensor tCrA = thr_mma.partition_fragment_A(sQ); // Shared operand fragment across all 5 GEMMs
    Tensor tCrB = thr_mma.partition_fragment_B(sK); // Shared operand fragment across all 5 GEMMs

    auto s2r_copy_A_k = make_tiled_copy_A(S2RCopyAtomA_KMajor{}, tiled_mma);
    auto s2r_thr_A_k = s2r_copy_A_k.get_thread_slice(tid);
    Tensor tArA_k = s2r_thr_A_k.retile_D(tCrA);

    auto s2r_copy_B_k = make_tiled_copy_B(S2RCopyAtomB_KMajor{}, tiled_mma);
    auto s2r_thr_B_k = s2r_copy_B_k.get_thread_slice(tid);
    Tensor tBrB_k = s2r_thr_B_k.retile_D(tCrB);

    auto s2r_copy_A_mn = make_tiled_copy_A(S2RCopyAtomA_MNMajor{}, tiled_mma);
    auto s2r_thr_A_mn = s2r_copy_A_mn.get_thread_slice(tid);
    Tensor tArA_mn = s2r_thr_A_mn.retile_D(tCrA);

    auto s2r_copy_B_mn = make_tiled_copy_B(S2RCopyAtomB_MNMajor{}, tiled_mma);
    auto s2r_thr_B_mn = s2r_copy_B_mn.get_thread_slice(tid);
    Tensor tBrB_mn = s2r_thr_B_mn.retile_D(tCrB);

    Tensor tAsQ = s2r_thr_A_k.partition_S(sQ);
    Tensor tAsDO = s2r_thr_A_k.partition_S(sDO);
    Tensor tAsDS = s2r_thr_A_k.partition_S(sdS);
    Tensor tAsP_mn = s2r_thr_A_mn.partition_S(sP_mn);
    Tensor tAsDS_mn = s2r_thr_A_mn.partition_S(sdS_mn);

    Tensor tBsK = s2r_thr_B_k.partition_S(sK);
    Tensor tBsV = s2r_thr_B_k.partition_S(sV);
    Tensor tBsDO_mn = s2r_thr_B_mn.partition_S(sDO_mn);
    Tensor tBsQ_mn = s2r_thr_B_mn.partition_S(sQ_mn);
    Tensor tBsK_mn = s2r_thr_B_mn.partition_S(sK_mn);

    Tensor tCsP_C = thr_mma.partition_C(sP);
    Tensor tCsDS_C = thr_mma.partition_C(sdS);
    Tensor tCsStaging_C = thr_mma.partition_C(sStaging);
    Tensor tCsStagingF32_C = thr_mma.partition_C(sStaging_fp32);

    Tensor cC = thr_mma.partition_C(
        make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_N>{})));
    const int mma_row_a = get<0>(cC(0, 0, 0));
    const int mma_row_b = get<0>(cC(2, 0, 0));

    // 2. Prologue: load K, V and apply forward RoPE to K
    clear(tCrDK);
    clear(tCrDV);
    clear(rK);
    clear(rV);

    cute::copy_if(tiled_copy_g2r, tKpK, tKgK, rK);
    cute::copy_if(tiled_copy_g2r, tKpK, tVgV, rV);

    if constexpr (HAS_ROPE) {
        apply_rope_fwd_32x8<TILE_K>(
            rK, cos_ptr, sin_ptr, k_start_col, offset, valid_k_rows,
            row_m0_g2r, row_m1_g2r, col_vec_g2r, is_second_half_g2r);
    }

    cute::copy(rK, tKsK);
    cute::copy(rV, tVsV);
    __syncthreads();

    // 3. Main loop over GQA heads and Q blocks
    const int num_q_blocks = (lq_i + TILE_M - 1) / TILE_M;
    const int start_q_block = IS_CAUSAL ? block_k_idx : 0;

    for (int g = 0; g < num_groups; ++g) {
        const int h_q = h_kv * num_groups + g;
        const int64_t q_head_base = static_cast<int64_t>(h_q) * Total_Lq * TILE_K +
                                    static_cast<int64_t>(q_start_seq) * TILE_K;
        const float* lse_base = LSE_ptr + static_cast<int64_t>(h_q) * Total_Lq + q_start_seq;

        for (int block_q_idx = start_q_block; block_q_idx < num_q_blocks; ++block_q_idx) {
            const int q_start_row = block_q_idx * TILE_M;
            const int valid_q_rows = lq_i - q_start_row;

            const int64_t q_tile_offset = q_head_base + static_cast<int64_t>(q_start_row) * TILE_K;
            Tensor gQ = make_tensor(make_gmem_ptr(Q_ptr + q_tile_offset),
                                    make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                    make_stride(Int<TILE_K>{}, _1{}));
            Tensor gDO = make_tensor(make_gmem_ptr(dO_ptr + q_tile_offset),
                                     make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                     make_stride(Int<TILE_K>{}, _1{}));
            Tensor gO = make_tensor(make_gmem_ptr(O_ptr + q_tile_offset),
                                    make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                    make_stride(Int<TILE_K>{}, _1{}));
            Tensor gDQ_f32 = make_tensor(make_gmem_ptr(dQ_fp32_ptr + q_tile_offset),
                                         make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                         make_stride(Int<TILE_K>{}, _1{}));

            Tensor tQgQ = thr_copy_g2r.partition_S(gQ);
            Tensor tDOgDO = thr_copy_g2r.partition_S(gDO);
            Tensor tOgO = thr_copy_g2r.partition_S(gO);
            Tensor tDQgDQ_f32 = thr_copy_f32.partition_D(gDQ_f32);

            tQpQ(0, 0) = (row_m0_g2r < valid_q_rows);
            tQpQ(1, 0) = (row_m1_g2r < valid_q_rows);

            // 3.1 Load Q (+ RoPE), dO, O and compute D = rowsum(dO * O)
            clear(rQ);
            clear(rDO);
            clear(rV); // Temporarily reuse rV for O (unused until Epilogue)

            cute::copy_if(tiled_copy_g2r, tQpQ, tQgQ, rQ);
            cute::copy_if(tiled_copy_g2r, tQpQ, tDOgDO, rDO);
            cute::copy_if(tiled_copy_g2r, tQpQ, tOgO, rV);

            if constexpr (HAS_ROPE) {
                apply_rope_fwd_32x8<TILE_K>(
                    rQ, cos_ptr, sin_ptr, q_start_row, offset, valid_q_rows,
                    row_m0_g2r, row_m1_g2r, col_vec_g2r, is_second_half_g2r);
            }

            // Compute D_i = sum_d (dO[i, d] * O[i, d]) via 8-lane reduction
            CUTE_UNROLL
            for (int m = 0; m < size<1>(rDO); ++m) {
                float partial_d = 0.0f;
                CUTE_UNROLL
                for (int v = 0; v < size<0>(rDO); ++v) {
                    partial_d = fmaf(half_to_float(rDO(v, m, 0)),
                                     half_to_float(rV(v, m, 0)),
                                     partial_d);
                }
                partial_d = octet_reduce_sum_f32(partial_d);
                if (col_vec_g2r == 0) {
                    const int row_in_tile = (m == 0) ? row_m0_g2r : row_m1_g2r;
                    sD_ptr[row_in_tile] = partial_d;
                }
            }

            cute::copy(rQ, tQsQ);
            cute::copy(rDO, tDOsDO);
            __syncthreads();

            // 3.2 GEMM 1: S = Q @ K^T -> P = exp(scale * S - LSE) -> sP
            clear(tCrC);
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_k, tAsQ(_, _, k_step), tArA_k(_, _, k_step));
                cute::copy(s2r_copy_B_k, tBsK(_, _, k_step), tBrB_k(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrC);
            }

            const int global_row_a = q_start_row + mma_row_a;
            const int global_row_b = q_start_row + mma_row_b;
            const bool valid_qa = (global_row_a < lq_i);
            const bool valid_qb = (global_row_b < lq_i);
            const float lse_a = valid_qa ? lse_base[global_row_a] : 0.0f;
            const float lse_b = valid_qb ? lse_base[global_row_b] : 0.0f;

            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(tCrC); ++ni) {
                const int col_0 = get<1>(cC(0, 0, ni));
                const int col_1 = get<1>(cC(1, 0, ni));
                const int global_col_0 = k_start_col + col_0;
                const int global_col_1 = k_start_col + col_1;

                const bool mask_00 = (!valid_qa) || (global_col_0 >= lk_i) ||
                                     (IS_CAUSAL && (global_col_0 > global_row_a));
                const bool mask_01 = (!valid_qa) || (global_col_1 >= lk_i) ||
                                     (IS_CAUSAL && (global_col_1 > global_row_a));
                const bool mask_10 = (!valid_qb) || (global_col_0 >= lk_i) ||
                                     (IS_CAUSAL && (global_col_0 > global_row_b));
                const bool mask_11 = (!valid_qb) || (global_col_1 >= lk_i) ||
                                     (IS_CAUSAL && (global_col_1 > global_row_b));

                const float p_00 = mask_00 ? 0.0f : expf(fmaf(tCrC(0, 0, ni), sm_scale, -lse_a));
                const float p_01 = mask_01 ? 0.0f : expf(fmaf(tCrC(1, 0, ni), sm_scale, -lse_a));
                const float p_10 = mask_10 ? 0.0f : expf(fmaf(tCrC(2, 0, ni), sm_scale, -lse_b));
                const float p_11 = mask_11 ? 0.0f : expf(fmaf(tCrC(3, 0, ni), sm_scale, -lse_b));

                rC_fp16(0, 0, ni) = float_to_out<Element>(p_00);
                rC_fp16(1, 0, ni) = float_to_out<Element>(p_01);
                rC_fp16(2, 0, ni) = float_to_out<Element>(p_10);
                rC_fp16(3, 0, ni) = float_to_out<Element>(p_11);
            }

            cute::copy(rC_fp16, tCsP_C);
            __syncthreads();

            // 3.3 GEMM 2: dV += P^T @ dO (FP32 accumulate in tCrDV)
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_mn, tAsP_mn(_, _, k_step), tArA_mn(_, _, k_step));
                cute::copy(s2r_copy_B_mn, tBsDO_mn(_, _, k_step), tBrB_mn(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrDV);
            }

            // 3.4 GEMM 3: dP = dO @ V^T -> dS_unscaled = P * (dP - D) -> sdS
            clear(tCrC);
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_k, tAsDO(_, _, k_step), tArA_k(_, _, k_step));
                cute::copy(s2r_copy_B_k, tBsV(_, _, k_step), tBrB_k(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrC);
            }

            // Load P from sP into registers to compute dS (defer sm_scale for FP16 dynamic range)
            cute::copy(tCsP_C, rC_fp16);
            const float D_a = sD_ptr[mma_row_a];
            const float D_b = sD_ptr[mma_row_b];

            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(tCrC); ++ni) {
                const float dS_00 = half_to_float(rC_fp16(0, 0, ni)) * (tCrC(0, 0, ni) - D_a);
                const float dS_01 = half_to_float(rC_fp16(1, 0, ni)) * (tCrC(1, 0, ni) - D_a);
                const float dS_10 = half_to_float(rC_fp16(2, 0, ni)) * (tCrC(2, 0, ni) - D_b);
                const float dS_11 = half_to_float(rC_fp16(3, 0, ni)) * (tCrC(3, 0, ni) - D_b);

                rC_fp16(0, 0, ni) = float_to_out<Element>(dS_00);
                rC_fp16(1, 0, ni) = float_to_out<Element>(dS_01);
                rC_fp16(2, 0, ni) = float_to_out<Element>(dS_10);
                rC_fp16(3, 0, ni) = float_to_out<Element>(dS_11);
            }

            cute::copy(rC_fp16, tCsDS_C);
            __syncthreads();

            // 3.5 GEMM 4: dK += dS^T @ Q (FP32 accumulate in tCrDK)
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_mn, tAsDS_mn(_, _, k_step), tArA_mn(_, _, k_step));
                cute::copy(s2r_copy_B_mn, tBsQ_mn(_, _, k_step), tBrB_mn(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrDK);
            }

            // --- 3.6 GEMM 5: dQ_local = dS @ K -> Scale -> Stage FP32 -> RoPE bwd -> atomicAdd ---
            clear(tCrC);
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_k, tAsDS(_, _, k_step), tArA_k(_, _, k_step));
                cute::copy(s2r_copy_B_mn, tBsK_mn(_, _, k_step), tBrB_mn(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrC);
            }

            CUTE_UNROLL
            for (int i = 0; i < size(tCrC); ++i) {
                tCrC(i) *= sm_scale;
            }

            // Store dQ_local FP32 into sStaging_fp32 ([32KB..48KB), disjoint from sdS [48KB..56KB))
            cute::copy(tCrC, tCsStagingF32_C);
            __syncthreads();

            cute::copy(tiled_copy_f32, tStagingF32_S, rF32);

            if constexpr (HAS_ROPE) {
                apply_rope_bwd_f32_16x16<TILE_K>(
                    rF32, cos_ptr, sin_ptr, q_start_row, offset, valid_q_rows,
                    row_in_step_f32, col_vec_f32, is_second_half_f32);
            }

            // Atomic accumulation across K blocks into FP32 dQ buffer
            CUTE_UNROLL
            for (int m = 0; m < size<1>(rF32); ++m) {
                const int row_q = m * 16 + row_in_step_f32;
                if (row_q < valid_q_rows) {
                    CUTE_UNROLL
                    for (int v = 0; v < size<0>(rF32); ++v) {
                        atomicAdd(&tDQgDQ_f32(v, m, 0), rF32(v, m, 0));
                    }
                }
            }
            __syncthreads(); // Protect SMEM before next Q block or Epilogue
        }
    }

    // 4. Epilogue: store dV and dK
    // 4.1 Store dV: FP32 -> FP16 -> sStaging -> 128-bit coalesced GMEM
    CUTE_UNROLL
    for (int i = 0; i < size(tCrDV); ++i) {
        rC_fp16(i) = float_to_out<Element>(tCrDV(i));
    }
    cute::copy(rC_fp16, tCsStaging_C);
    __syncthreads();

    cute::copy(tiled_copy_g2r, tStaging_S, rV);
    cute::copy_if(tiled_copy_g2r, tKpK, rV, tDVgDV);
    __syncthreads(); // Wait for dV reads from sStaging before reusing SMEM for dK

    // 4.2 Store dK: apply sm_scale -> RoPE backward (if HAS_ROPE) -> sStaging -> 128-bit GMEM
    CUTE_UNROLL
    for (int i = 0; i < size(tCrDK); ++i) {
        tCrDK(i) *= sm_scale;
    }

    if constexpr (HAS_ROPE) {
        cute::copy(tCrDK, tCsStagingF32_C);
        __syncthreads();

        cute::copy(tiled_copy_f32, tStagingF32_S, rF32);
        __syncthreads(); // Wait for all sStaging_fp32 reads before writing sStaging ([40KB..48KB))

        apply_rope_bwd_f32_16x16<TILE_K>(
            rF32, cos_ptr, sin_ptr, k_start_col, offset, valid_k_rows,
            row_in_step_f32, col_vec_f32, is_second_half_f32);

        CUTE_UNROLL
        for (int i = 0; i < size(rF32); ++i) {
            rF32_fp16(i) = float_to_out<Element>(rF32(i));
        }

        cute::copy(tiled_copy_fp16_16x16, rF32_fp16, tStaging_D_16x16);
        __syncthreads();
    } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrDK); ++i) {
            rC_fp16(i) = float_to_out<Element>(tCrDK(i));
        }
        cute::copy(rC_fp16, tCsStaging_C);
        __syncthreads();
    }

    cute::copy(tiled_copy_g2r, tStaging_S, rK);
    cute::copy_if(tiled_copy_g2r, tKpK, rK, tDKgDK);
}

// =================================================================================================
// 2. Host Function
// =================================================================================================


std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_mha_varlen_bwd(
    const at::Tensor& do_grad,                        // (H_q, Total_Lq, 64) [FP16] - output gradient dO
    const at::Tensor& q,                              // (H_q, Total_Lq, 64) [FP16] - unrotated query Q
    const at::Tensor& k,                              // (H_kv, Total_Lk, 64) [FP16] - unrotated key K
    const at::Tensor& v,                              // (H_kv, Total_Lk, 64) [FP16] - value V
    const at::Tensor& o,                              // (H_q, Total_Lq, 64) [FP16] - forward output O
    const at::Tensor& lse,                            // (H_q, Total_Lq)     [FP32] - forward LSE
    const at::Tensor& cu_seqlens_q,                   // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                   // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets,         // (B,) [Int32] (optional)
    int64_t max_seqlen_k,                             // scalar [int64] - max K sequence length
    const std::optional<at::Tensor>& cos,             // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,             // (L_max, 64) [FP16] (optional)
    bool is_causal                                    // scalar [bool] - causal mask flag
) {
    // 1. Torch Checks
    TORCH_CHECK(do_grad.is_cuda(), "do_grad must be a CUDA tensor");
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(o.is_cuda(), "o must be a CUDA tensor");
    TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be a CUDA tensor");

    TORCH_CHECK(do_grad.device() == q.device() && k.device() == q.device() &&
                v.device() == q.device() && o.device() == q.device() &&
                lse.device() == q.device() && cu_seqlens_q.device() == q.device() &&
                cu_seqlens_k.device() == q.device(),
                "All input tensors must reside on the same CUDA device");

    TORCH_CHECK(do_grad.is_contiguous() && q.is_contiguous() && k.is_contiguous() &&
                v.is_contiguous() && o.is_contiguous() && lse.is_contiguous() &&
                cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "All input tensors must be contiguous");

    TORCH_CHECK(do_grad.scalar_type() == at::kHalf, "do_grad must be float16");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "v must be float16");
    TORCH_CHECK(o.scalar_type() == at::kHalf, "o must be float16");
    TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be float32");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_k must be int32");

    TORCH_CHECK(q.dim() == 3 && do_grad.dim() == 3 && o.dim() == 3 &&
                k.dim() == 3 && v.dim() == 3,
                "do_grad, q, k, v, o must be 3D tensors");
    TORCH_CHECK(q.size(2) == 64 && k.size(2) == 64, "Head dimension must be 64");
    TORCH_CHECK(do_grad.sizes() == q.sizes(), "do_grad shape must match q shape");
    TORCH_CHECK(o.sizes() == q.sizes(), "o shape must match q shape");
    TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k shape");

    const int64_t H_q = q.size(0);
    const int64_t Total_Lq = q.size(1);
    const int64_t H_kv = k.size(0);
    const int64_t Total_Lk = k.size(1);

    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0),
                "H_q must be a positive multiple of H_kv");
    TORCH_CHECK(lse.dim() == 2 && lse.size(0) == H_q && lse.size(1) == Total_Lq,
                "lse must have shape (H_q, Total_Lq)");

    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1 &&
                cu_seqlens_q.size(0) >= 1 && cu_seqlens_q.size(0) == cu_seqlens_k.size(0),
                "cu_seqlens_q and cu_seqlens_k must be 1D tensors of identical length >= 1");
    TORCH_CHECK(max_seqlen_k >= 0, "max_seqlen_k must be non-negative");

    const int64_t B = cu_seqlens_q.size(0) - 1;

    if (offsets.has_value()) {
        TORCH_CHECK(offsets->is_cuda() && offsets->device() == q.device() &&
                    offsets->is_contiguous() && offsets->scalar_type() == at::kInt &&
                    offsets->dim() == 1 && offsets->size(0) == B,
                    "offsets must be a contiguous CUDA int32 tensor of shape (B,)");
    }

    TORCH_CHECK(cos.has_value() == sin.has_value(),
                "cos and sin must either both be provided or both be omitted");
    const bool has_rope = cos.has_value();
    if (has_rope) {
        TORCH_CHECK(cos->is_cuda() && sin->is_cuda() &&
                    cos->device() == q.device() && sin->device() == q.device() &&
                    cos->is_contiguous() && sin->is_contiguous() &&
                    cos->scalar_type() == at::kHalf && sin->scalar_type() == at::kHalf &&
                    cos->dim() == 2 && sin->dim() == 2 &&
                    cos->size(1) == 64 && cos->sizes() == sin->sizes(),
                    "cos and sin must be contiguous CUDA float16 tensors of identical shape (L_max, 64)");
        if (Total_Lq > 0 || Total_Lk > 0) {
            TORCH_CHECK(cos->size(0) > 0, "cos/sin table must have L_max > 0 for non-empty sequences");
        }
    }

    // 2. Buffer allocation and kernel launch
    const c10::cuda::CUDAGuard device_guard(q.device());

    at::Tensor dk = at::zeros_like(k);
    at::Tensor dv = at::zeros_like(v);

    if (B == 0 || max_seqlen_k == 0 || Total_Lq == 0 || Total_Lk == 0) {
        return std::make_tuple(at::zeros_like(q), dk, dv);
    }

    at::Tensor dq_fp32 = at::zeros({H_q, Total_Lq, 64}, q.options().dtype(at::kFloat));

    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 64;
    constexpr int NUM_THREADS = 256;
    constexpr int smem_size =
        7 * TILE_M * TILE_N * static_cast<int>(sizeof(cutlass::half_t)) +
        TILE_M * static_cast<int>(sizeof(float));

    const int num_k_blocks = static_cast<int>((max_seqlen_k + TILE_N - 1) / TILE_N);
    const dim3 grid(static_cast<unsigned int>(num_k_blocks),
                    static_cast<unsigned int>(B),
                    static_cast<unsigned int>(H_kv));
    const dim3 block(NUM_THREADS);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(TILE_K));

    const auto* do_ptr = reinterpret_cast<const cutlass::half_t*>(do_grad.data_ptr<at::Half>());
    const auto* q_ptr = reinterpret_cast<const cutlass::half_t*>(q.data_ptr<at::Half>());
    const auto* k_ptr = reinterpret_cast<const cutlass::half_t*>(k.data_ptr<at::Half>());
    const auto* v_ptr = reinterpret_cast<const cutlass::half_t*>(v.data_ptr<at::Half>());
    const auto* o_ptr = reinterpret_cast<const cutlass::half_t*>(o.data_ptr<at::Half>());
    const float* lse_ptr = lse.data_ptr<float>();
    float* dq_fp32_ptr = dq_fp32.data_ptr<float>();
    auto* dk_ptr = reinterpret_cast<cutlass::half_t*>(dk.data_ptr<at::Half>());
    auto* dv_ptr = reinterpret_cast<cutlass::half_t*>(dv.data_ptr<at::Half>());
    const int32_t* cu_q_ptr = cu_seqlens_q.data_ptr<int32_t>();
    const int32_t* cu_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
    const int32_t* offsets_ptr = offsets.has_value() ? offsets->data_ptr<int32_t>() : nullptr;
    const cutlass::half_t* cos_ptr = has_rope
        ? reinterpret_cast<const cutlass::half_t*>(cos->data_ptr<at::Half>())
        : nullptr;
    const cutlass::half_t* sin_ptr = has_rope
        ? reinterpret_cast<const cutlass::half_t*>(sin->data_ptr<at::Half>())
        : nullptr;

    {
        auto launch_bwd = [&](auto has_rope_tag, auto is_causal_tag) {
            constexpr bool kHasRope = decltype(has_rope_tag)::value;
            constexpr bool kIsCausal = decltype(is_causal_tag)::value;
            auto kernel_ptr = kernel_fused_mha_varlen_bwd<
                TILE_M, TILE_N, TILE_K, NUM_THREADS, kHasRope, kIsCausal>;
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            kernel_ptr<<<grid, block, smem_size, stream>>>(
                do_ptr,
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                lse_ptr,
                dq_fp32_ptr,
                dk_ptr,
                dv_ptr,
                cu_q_ptr,
                cu_k_ptr,
                offsets_ptr,
                cos_ptr,
                sin_ptr,
                static_cast<int>(H_q),
                static_cast<int>(H_kv),
                static_cast<int>(Total_Lq),
                static_cast<int>(Total_Lk),
                sm_scale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        };

        if (has_rope) {
            if (is_causal) {
                launch_bwd(std::true_type{}, std::true_type{});
            } else {
                launch_bwd(std::true_type{}, std::false_type{});
            }
        } else {
            if (is_causal) {
                launch_bwd(std::false_type{}, std::true_type{});
            } else {
                launch_bwd(std::false_type{}, std::false_type{});
            }
        }
    }

    // 3. Cast dQ to FP16, release FP32 buffer, and return outputs
    at::Tensor dq = dq_fp32.to(at::kHalf);
    dq_fp32 = at::Tensor();

    return std::make_tuple(dq, dk, dv);
}
