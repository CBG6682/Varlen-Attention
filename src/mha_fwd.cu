#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
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
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "mha_common.cuh"
#include "mha_ops.cuh"

using namespace cute;

// Varlen FlashAttention + Fused RoPE Forward kernel (SM75 64x64x64).

template <
    typename Element,
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int NUM_THREADS,
    bool HAS_ROPE,
    bool IS_CAUSAL,
    typename SmemLayoutKMajor,
    typename SmemLayoutMNMajor,
    typename TiledCopyG2R,
    typename CopyAtomLSE,
    typename TiledMMAType,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB_KMajor,
    typename S2RCopyAtomB_MNMajor
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_mha_varlen_fwd(
    const Element* __restrict__ gQ_ptr,          // (Total_Lq, H_q, 64) [FP16]
    const Element* __restrict__ gK_ptr,          // (Total_Lk, H_kv, 64) [FP16]
    const Element* __restrict__ gV_ptr,          // (Total_Lk, H_kv, 64) [FP16]
    const int32_t* __restrict__ cu_seqlens_q,    // (B + 1,) [Int32]
    const int32_t* __restrict__ cu_seqlens_k,    // (B + 1,) [Int32]
    const Element* __restrict__ gCos_ptr,        // (L_max, 64) [FP16] (nullptr if !HAS_ROPE)
    const Element* __restrict__ gSin_ptr,        // (L_max, 64) [FP16] (nullptr if !HAS_ROPE)
    const int32_t* __restrict__ offsets_ptr,     // (B,) [Int32] (nullptr if offset == 0)
    Element*       __restrict__ gO_ptr,          // (Total_Lq, H_q, 64) [FP16]
    float*         __restrict__ gLSE_ptr,        // (H_q, Total_Lq) [FP32] (nullable)
    int64_t q_stride_l,                          // scalar: int64
    int64_t q_stride_h,                          // scalar: int64
    int64_t k_stride_l,                          // scalar: int64
    int64_t k_stride_h,                          // scalar: int64
    int64_t v_stride_l,                          // scalar: int64
    int64_t v_stride_h,                          // scalar: int64
    int64_t o_stride_l,                          // scalar: int64
    int64_t o_stride_h,                          // scalar: int64
    int64_t total_lq,                            // scalar: int64
    int64_t l_max,                               // scalar: int64
    int H_q,                                     // scalar: int
    int H_kv,                                    // scalar: int
    float scale                                  // scalar: float (1 / sqrt(64) = 0.125)
) {
    // 1. Varlen sequence bounds & coordinates
    const int block_q_idx = static_cast<int>(blockIdx.x);
    const int head_q_idx  = static_cast<int>(blockIdx.y);
    const int seq_idx     = static_cast<int>(blockIdx.z);

    const int rep         = H_q / H_kv;
    const int head_kv_idx = head_q_idx / rep;

    const int q_start    = cu_seqlens_q[seq_idx];
    const int q_end      = cu_seqlens_q[seq_idx + 1];
    const int lq_i       = q_end - q_start;

    const int k_start    = cu_seqlens_k[seq_idx];
    const int k_end      = cu_seqlens_k[seq_idx + 1];
    const int lk_i       = k_end - k_start;

    const int seq_offset = (offsets_ptr != nullptr) ? offsets_ptr[seq_idx] : 0;

    const int q_start_row = block_q_idx * TILE_M;
    if (q_start_row >= lq_i) return;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp_id      = tid / 32;
    const int lane_id      = tid % 32;
    const int quad_id      = lane_id / 4;
    const int lane_in_quad = lane_id % 4;

    const int row_a        = warp_id * 16 + quad_id;
    const int row_b        = warp_id * 16 + quad_id + 8;
    const int global_row_a = q_start_row + row_a;
    const int global_row_b = q_start_row + row_b;

    const int  col_vec_idx    = tid % (TILE_K / 8);
    const bool is_second_half = (col_vec_idx >= 4);

    // Offset base pointers to (seq_start, head) in (L, H, D) strided layout
    const Element* q_base_ptr   = gQ_ptr + static_cast<int64_t>(q_start) * q_stride_l + static_cast<int64_t>(head_q_idx)  * q_stride_h;
    const Element* k_base_ptr   = gK_ptr + static_cast<int64_t>(k_start) * k_stride_l + static_cast<int64_t>(head_kv_idx) * k_stride_h;
    const Element* v_base_ptr   = gV_ptr + static_cast<int64_t>(k_start) * v_stride_l + static_cast<int64_t>(head_kv_idx) * v_stride_h;
    Element*       o_base_ptr   = gO_ptr + static_cast<int64_t>(q_start) * o_stride_l + static_cast<int64_t>(head_q_idx)  * o_stride_h;
    float*         lse_head_ptr = (gLSE_ptr != nullptr)
        ? (gLSE_ptr + static_cast<int64_t>(head_q_idx) * total_lq + q_start)
        : nullptr;

    const Element* cos_seq_ptr  = HAS_ROPE ? (gCos_ptr + static_cast<int64_t>(seq_offset) * TILE_K) : nullptr;
    const Element* sin_seq_ptr  = HAS_ROPE ? (gSin_ptr + static_cast<int64_t>(seq_offset) * TILE_K) : nullptr;
    const int64_t  cos_rows     = HAS_ROPE ? (l_max - static_cast<int64_t>(seq_offset)) : 0;

    // 2. Tensor & SMEM partitioning (16 KB dynamic SMEM split into two 8 KB buffers)
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Element* sBuf0_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* sBuf1_ptr = reinterpret_cast<Element*>(smem_raw + TILE_M * TILE_K * sizeof(Element));

    Tensor sBuf0      = make_tensor(make_smem_ptr(sBuf0_ptr), SmemLayoutKMajor{});
    Tensor sBuf1      = make_tensor(make_smem_ptr(sBuf1_ptr), SmemLayoutKMajor{});
    Tensor sBuf1_V_mn = make_tensor(make_smem_ptr(sBuf1_ptr), SmemLayoutMNMajor{});

    // Strided THD GMEM tensors for current sequence and head
    Tensor mQ   = make_tensor(make_gmem_ptr(q_base_ptr),  make_shape(lq_i, Int<TILE_K>{}),     make_stride(q_stride_l,   Int<1>{}));
    Tensor mK   = make_tensor(make_gmem_ptr(k_base_ptr),  make_shape(lk_i, Int<TILE_K>{}),     make_stride(k_stride_l,   Int<1>{}));
    Tensor mV   = make_tensor(make_gmem_ptr(v_base_ptr),  make_shape(lk_i, Int<TILE_K>{}),     make_stride(v_stride_l,   Int<1>{}));
    Tensor mO   = make_tensor(make_gmem_ptr(o_base_ptr),  make_shape(lq_i, Int<TILE_K>{}),     make_stride(o_stride_l,   Int<1>{}));
    Tensor mCos = make_tensor(make_gmem_ptr(cos_seq_ptr), make_shape(cos_rows, Int<TILE_K>{}), make_stride(Int<TILE_K>{}, Int<1>{}));
    Tensor mSin = make_tensor(make_gmem_ptr(sin_seq_ptr), make_shape(cos_rows, Int<TILE_K>{}), make_stride(Int<TILE_K>{}, Int<1>{}));

    // Slice Q and O tiles for current block
    Tensor gQ = local_tile(mQ, make_tile(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(block_q_idx, 0));
    Tensor gO = local_tile(mO, make_tile(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(block_q_idx, 0));

    // 128-bit GMEM <-> SMEM TiledCopy partitions
    TiledCopyG2R tiled_copy_g2r;
    auto thr_copy_g2r = tiled_copy_g2r.get_slice(tid);

    auto tQgQ    = thr_copy_g2r.partition_S(gQ);
    auto tQsBuf0 = thr_copy_g2r.partition_D(sBuf0);
    auto tVsBuf1 = thr_copy_g2r.partition_D(sBuf1);
    auto tOsBuf1 = thr_copy_g2r.partition_S(sBuf1);
    auto tOgO    = thr_copy_g2r.partition_D(gO);

    // 16-byte aligned staging fragments for g2r -> RoPE -> r2s
    constexpr int FRAG_ELEMS = (TILE_M * TILE_K) / NUM_THREADS;
    auto frag_layout = make_layout(shape(tQgQ));
    alignas(16) Element buf_stage[FRAG_ELEMS];
    alignas(16) Element buf_cos[FRAG_ELEMS];
    alignas(16) Element buf_sin[FRAG_ELEMS];
    Tensor rStage = make_tensor(make_rmem_ptr(buf_stage), frag_layout);
    Tensor rCos   = make_tensor(make_rmem_ptr(buf_cos),   frag_layout);
    Tensor rSin   = make_tensor(make_rmem_ptr(buf_sin),   frag_layout);

    // Precompute boundary predication
    Tensor cTile = make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_K>{}));
    Tensor tQcQ  = thr_copy_g2r.partition_S(cTile);
    Tensor tQpQ  = make_tensor<bool>(shape(tQgQ));

    const bool q_in_bounds = (q_start_row + TILE_M <= lq_i);
    if (q_in_bounds) {
        cute::fill(tQpQ, true);
    } else {
        #pragma unroll
        for (int i = 0; i < size(tQpQ); ++i) {
            tQpQ(i) = (q_start_row + get<0>(tQcQ(i)) < lq_i);
        }
    }

    int num_k_blocks = (lk_i + TILE_N - 1) / TILE_N;
    if constexpr (IS_CAUSAL) {
        num_k_blocks = min(num_k_blocks, block_q_idx + 1);
    }

    const int  last_k_block_idx = (lk_i > 0) ? ((lk_i - 1) / TILE_N) : 0;
    const bool last_k_in_bounds = ((last_k_block_idx + 1) * TILE_N <= lk_i);
    Tensor tKpK = make_tensor<bool>(shape(tQgQ));
    if (last_k_in_bounds) {
        cute::fill(tKpK, true);
    } else {
        #pragma unroll
        for (int i = 0; i < size(tKpK); ++i) {
            tKpK(i) = (last_k_block_idx * TILE_N + get<0>(tQcQ(i)) < lk_i);
        }
    }

    // LSE output tensor and lane-leader predication
    Tensor tLgLSE = make_tensor(
        make_gmem_ptr((lse_head_ptr != nullptr) ? (lse_head_ptr + q_start_row + warp_id * 16 + quad_id) : nullptr),
        make_shape(Int<1>{}, Int<2>{}),
        make_stride(Int<1>{}, Int<8>{})
    );
    Tensor tLrLSE = make_tensor<float>(make_shape(Int<1>{}, Int<2>{}), make_stride(Int<1>{}, Int<1>{}));
    Tensor tLpLSE = make_tensor<bool>(make_shape(Int<1>{}, Int<2>{}), make_stride(Int<1>{}, Int<1>{}));
    tLpLSE(0, 0) = (gLSE_ptr != nullptr) && (lane_in_quad == 0) && (global_row_a < lq_i);
    tLpLSE(0, 1) = (gLSE_ptr != nullptr) && (lane_in_quad == 0) && (global_row_b < lq_i);

    // TiledMMA, S2R copy atoms, and shared operand fragments
    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto s2r_copy_a    = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b_km = make_tiled_copy_B(S2RCopyAtomB_KMajor{}, tiled_mma);
    auto s2r_copy_b_mn = make_tiled_copy_B(S2RCopyAtomB_MNMajor{}, tiled_mma);

    auto thr_s2r_a     = s2r_copy_a.get_slice(tid);
    auto thr_s2r_b_km  = s2r_copy_b_km.get_slice(tid);
    auto thr_s2r_b_mn  = s2r_copy_b_mn.get_slice(tid);

    auto tAsBuf0_s2r   = thr_s2r_a.partition_S(sBuf0);
    auto tBsBuf0_s2r   = thr_s2r_b_km.partition_S(sBuf0);
    auto tBsBuf1_s2r   = thr_s2r_b_mn.partition_S(sBuf1_V_mn);
    auto tCsBuf1_C     = thr_mma.partition_C(sBuf1);

    // tCrQ holds Q; (tCrA, tCrB) are shared between S = Q@K^T and O += P@V
    Tensor tCrQ = thr_mma.partition_fragment_A(sBuf0);
    Tensor tCrA = thr_mma.partition_fragment_A(sBuf0);
    Tensor tCrB = thr_mma.partition_fragment_B(sBuf0);

    auto tCrQ_s2r_view = thr_s2r_a.retile_D(tCrQ);
    auto tCrB_km_view  = thr_s2r_b_km.retile_D(tCrB);
    auto tCrB_mn_view  = thr_s2r_b_mn.retile_D(tCrB);

    // -------------------------------------------------------------------------------------
    // 3. Accumulators, Online Softmax state & RoPE/copy helpers
    // -------------------------------------------------------------------------------------
    Tensor tCrO = thr_mma.partition_fragment_C(sBuf1);
    Tensor tCrS = thr_mma.partition_fragment_C(sBuf1);
    cute::clear(tCrO);

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    constexpr float LOG2E   = 1.4426950408889634f;

    float m_old_a = NEG_INF;
    float m_old_b = NEG_INF;
    float l_old_a = 0.0f;
    float l_old_b = 0.0f;

    // Load 128-bit vector from GMEM to staging registers
    auto g2r_load = [&](auto const& tSrc, auto& rDst, auto const& pred, bool in_bounds) {
        if (in_bounds) {
            cute::copy(tiled_copy_g2r, tSrc, rDst);
        } else {
            cute::clear(rDst);
            cute::copy_if(tiled_copy_g2r, pred, tSrc, rDst);
        }
    };

    // Write 128-bit uint4 registers to swizzled SMEM
    auto r2s_store = [](auto const& rSrc, auto& tSmemD) {
        #pragma unroll
        for (int m = 0; m < size<1>(rSrc); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(rSrc); ++n) {
                *reinterpret_cast<uint4*>(&tSmemD(0, m, n)) = *reinterpret_cast<const uint4*>(&rSrc(0, m, n));
            }
        }
    };

    // Forward RoPE rotation in registers using __shfl_xor_sync(..., 4)
    auto apply_rope_reg = [&]() {
        #pragma unroll
        for (int m = 0; m < size<1>(rStage); ++m) {
            apply_rope_fwd_8elem_shfl(
                &rStage(0, m, 0),
                reinterpret_cast<const __half2*>(&rCos(0, m, 0)),
                reinterpret_cast<const __half2*>(&rSin(0, m, 0)),
                is_second_half
            );
        }
    };

    // -------------------------------------------------------------------------------------
    // 2. Prologue: load Q (+ RoPE) into tCrQ and prefetch K_0 into sBuf0
    // -------------------------------------------------------------------------------------
    g2r_load(tQgQ, rStage, tQpQ, q_in_bounds);
    if constexpr (HAS_ROPE) {
        Tensor gCos_q = local_tile(mCos, make_tile(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(block_q_idx, 0));
        Tensor gSin_q = local_tile(mSin, make_tile(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(block_q_idx, 0));
        g2r_load(thr_copy_g2r.partition_S(gCos_q), rCos, tQpQ, q_in_bounds);
        g2r_load(thr_copy_g2r.partition_S(gSin_q), rSin, tQpQ, q_in_bounds);
        apply_rope_reg();
    }
    r2s_store(rStage, tQsBuf0);
    __syncthreads();

    // Load rotated Q from sBuf0 into tCrQ via LDSM_N
    cute::copy(s2r_copy_a, tAsBuf0_s2r, tCrQ_s2r_view);
    __syncthreads();

    // Prefetch K_0 (+ RoPE) into sBuf0 for the first iteration
    if (num_k_blocks > 0) {
        const bool k0_in_bounds = (0 < last_k_block_idx) || last_k_in_bounds;
        Tensor gK_0 = local_tile(mK, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(0, 0));
        g2r_load(thr_copy_g2r.partition_S(gK_0), rStage, tKpK, k0_in_bounds);
        if constexpr (HAS_ROPE) {
            Tensor gCos_0 = local_tile(mCos, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(0, 0));
            Tensor gSin_0 = local_tile(mSin, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(0, 0));
            g2r_load(thr_copy_g2r.partition_S(gCos_0), rCos, tKpK, k0_in_bounds);
            g2r_load(thr_copy_g2r.partition_S(gSin_0), rSin, tKpK, k0_in_bounds);
            apply_rope_reg();
        }
        r2s_store(rStage, tQsBuf0);
        __syncthreads();
    }

    // -------------------------------------------------------------------------------------
    // 3. Main loop over K/V blocks of sequence seq_idx
    // -------------------------------------------------------------------------------------
    for (int k_tile = 0; k_tile < num_k_blocks; ++k_tile) {
        const int  k_start_col      = k_tile * TILE_N;
        const bool curr_k_in_bounds = (k_tile < last_k_block_idx) || last_k_in_bounds;

        // Compute S = Q @ K^T while prefetching V_curr
        Tensor gV_curr = local_tile(mV, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(k_tile, 0));
        g2r_load(thr_copy_g2r.partition_S(gV_curr), rStage, tKpK, curr_k_in_bounds);

        // Load K from sBuf0 into tCrB and compute S = Q @ K^T
        cute::copy(s2r_copy_b_km, tBsBuf0_s2r, tCrB_km_view);
        cute::clear(tCrS);
        cute::gemm(tiled_mma, tCrQ, tCrB, tCrS);

        // Write prefetched V_curr to sBuf1 and sync
        r2s_store(rStage, tVsBuf1);
        __syncthreads();

        // Scale, causal mask, Online Softmax & direct register P -> tCrA
        float m_new_a = m_old_a;
        float m_new_b = m_old_b;

        #pragma unroll
        for (int n = 0; n < size<2>(tCrS); ++n) {
            const int col_0 = k_start_col + n * 8 + lane_in_quad * 2;
            const int col_1 = col_0 + 1;

            const float s_a0 = (col_0 >= lk_i || (IS_CAUSAL && col_0 > global_row_a)) ? NEG_INF : (tCrS(0, 0, n) * scale);
            const float s_a1 = (col_1 >= lk_i || (IS_CAUSAL && col_1 > global_row_a)) ? NEG_INF : (tCrS(1, 0, n) * scale);
            const float s_b0 = (col_0 >= lk_i || (IS_CAUSAL && col_0 > global_row_b)) ? NEG_INF : (tCrS(2, 0, n) * scale);
            const float s_b1 = (col_1 >= lk_i || (IS_CAUSAL && col_1 > global_row_b)) ? NEG_INF : (tCrS(3, 0, n) * scale);

            tCrS(0, 0, n) = s_a0;
            tCrS(1, 0, n) = s_a1;
            tCrS(2, 0, n) = s_b0;
            tCrS(3, 0, n) = s_b1;

            m_new_a = fmaxf(m_new_a, fmaxf(s_a0, s_a1));
            m_new_b = fmaxf(m_new_b, fmaxf(s_b0, s_b1));
        }
        m_new_a = quad_reduce_max_f32(m_new_a);
        m_new_b = quad_reduce_max_f32(m_new_b);

        const float alpha_a = (m_old_a == NEG_INF || m_new_a == NEG_INF) ? 0.0f : exp2f((m_old_a - m_new_a) * LOG2E);
        const float alpha_b = (m_old_b == NEG_INF || m_new_b == NEG_INF) ? 0.0f : exp2f((m_old_b - m_new_b) * LOG2E);

        float row_sum_a = 0.0f;
        float row_sum_b = 0.0f;

        #pragma unroll
        for (int n = 0; n < size<2>(tCrS); ++n) {
            // Rescale accumulator tCrO by alpha
            tCrO(0, 0, n) *= alpha_a;
            tCrO(1, 0, n) *= alpha_a;
            tCrO(2, 0, n) *= alpha_b;
            tCrO(3, 0, n) *= alpha_b;

            const float s_a0 = tCrS(0, 0, n);
            const float s_a1 = tCrS(1, 0, n);
            const float s_b0 = tCrS(2, 0, n);
            const float s_b1 = tCrS(3, 0, n);

            const float p_a0 = (s_a0 == NEG_INF || m_new_a == NEG_INF) ? 0.0f : exp2f((s_a0 - m_new_a) * LOG2E);
            const float p_a1 = (s_a1 == NEG_INF || m_new_a == NEG_INF) ? 0.0f : exp2f((s_a1 - m_new_a) * LOG2E);
            const float p_b0 = (s_b0 == NEG_INF || m_new_b == NEG_INF) ? 0.0f : exp2f((s_b0 - m_new_b) * LOG2E);
            const float p_b1 = (s_b1 == NEG_INF || m_new_b == NEG_INF) ? 0.0f : exp2f((s_b1 - m_new_b) * LOG2E);

            row_sum_a += p_a0 + p_a1;
            row_sum_b += p_b0 + p_b1;

            // Direct register transfer tCrS -> tCrA (no SMEM, no sync)
            tCrA(0, 0, n) = float_to_out<Element>(p_a0);
            tCrA(1, 0, n) = float_to_out<Element>(p_a1);
            tCrA(2, 0, n) = float_to_out<Element>(p_b0);
            tCrA(3, 0, n) = float_to_out<Element>(p_b1);
        }

        row_sum_a = quad_reduce_sum_f32(row_sum_a);
        row_sum_b = quad_reduce_sum_f32(row_sum_b);

        m_old_a = m_new_a;
        m_old_b = m_new_b;
        l_old_a = fmaf(l_old_a, alpha_a, row_sum_a);
        l_old_b = fmaf(l_old_b, alpha_b, row_sum_b);

        // Compute O += P @ V while prefetching K_next
        const bool has_next = (k_tile + 1 < num_k_blocks);
        if (has_next) {
            const bool next_k_in_bounds = (k_tile + 1 < last_k_block_idx) || last_k_in_bounds;
            Tensor gK_next = local_tile(mK, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(k_tile + 1, 0));
            g2r_load(thr_copy_g2r.partition_S(gK_next), rStage, tKpK, next_k_in_bounds);
            if constexpr (HAS_ROPE) {
                Tensor gCos_next = local_tile(mCos, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(k_tile + 1, 0));
                Tensor gSin_next = local_tile(mSin, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(k_tile + 1, 0));
                g2r_load(thr_copy_g2r.partition_S(gCos_next), rCos, tKpK, next_k_in_bounds);
                g2r_load(thr_copy_g2r.partition_S(gSin_next), rSin, tKpK, next_k_in_bounds);
            }
        }

        // Load transposed V via LDSM_T from sBuf1 into tCrB and compute O += P @ V
        cute::copy(s2r_copy_b_mn, tBsBuf1_s2r, tCrB_mn_view);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrO);

        // Apply RoPE and write K_next to sBuf0
        if (has_next) {
            if constexpr (HAS_ROPE) {
                apply_rope_reg();
            }
            r2s_store(rStage, tQsBuf0);
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------------------
    // 4. Epilogue: normalize O, store LSE, stage O via sBuf1 -> gO
    // -------------------------------------------------------------------------------------
    const float inv_l_a = (l_old_a > 0.0f && m_old_a != NEG_INF) ? __fdividef(1.0f, l_old_a) : 0.0f;
    const float inv_l_b = (l_old_b > 0.0f && m_old_b != NEG_INF) ? __fdividef(1.0f, l_old_b) : 0.0f;

    #pragma unroll
    for (int n = 0; n < size<2>(tCrO); ++n) {
        tCrO(0, 0, n) *= inv_l_a;
        tCrO(1, 0, n) *= inv_l_a;
        tCrO(2, 0, n) *= inv_l_b;
        tCrO(3, 0, n) *= inv_l_b;
    }

    // Store FP32 LSE with lane-leader predication when backward buffer is requested
    if (gLSE_ptr != nullptr) {
        tLrLSE(0, 0) = (l_old_a > 0.0f && m_old_a != NEG_INF) ? (m_old_a + __logf(l_old_a)) : NEG_INF;
        tLrLSE(0, 1) = (l_old_b > 0.0f && m_old_b != NEG_INF) ? (m_old_b + __logf(l_old_b)) : NEG_INF;
        cute::copy_if(CopyAtomLSE{}, tLpLSE, tLrLSE, tLgLSE);
    }

    // Convert FP32 accumulator to FP16 -> R2S into swizzled sBuf1 -> 128-bit S2G
    Tensor rO_fp16 = make_tensor<Element>(shape(tCrO));
    #pragma unroll
    for (int i = 0; i < size(tCrO); ++i) {
        rO_fp16(i) = float_to_out<Element>(tCrO(i));
    }
    cute::copy(rO_fp16, tCsBuf1_C);
    __syncthreads();

    // S2G: vectorized 128-bit store from swizzled sBuf1 to gO
    if (q_in_bounds) {
        cute::copy(tiled_copy_g2r, tOsBuf1, tOgO);
    } else {
        cute::copy_if(tiled_copy_g2r, tQpQ, tOsBuf1, tOgO);
    }
}

// Host launcher writing directly into preallocated o (and optional lse)

void launch_fused_mha_varlen_fwd_into(
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& k,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // scalar [int64]
    at::Tensor& o,                                           // (Total_Lq, H_q, 64) [FP16]
    const std::optional<at::Tensor>& lse_opt,                // (H_q, Total_Lq) [FP32] (optional)
    const std::optional<at::Tensor>& cos,                    // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,                    // (L_max, 64) [FP16] (optional)
    bool is_causal,                                          // scalar [bool]
    const std::optional<at::Tensor>& offsets                 // (B,) [Int32] (optional)
) {
    // 1. Input validation
    at::cuda::CUDAGuard device_guard(q.device());

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda(), "q, k, v, o must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q and cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(
        k.device() == q.device() && v.device() == q.device() && o.device() == q.device() &&
        cu_seqlens_q.device() == q.device() && cu_seqlens_k.device() == q.device(),
        "All input tensors must reside on the same CUDA device"
    );

    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous");
    TORCH_CHECK(
        q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
        v.scalar_type() == at::kHalf && o.scalar_type() == at::kHalf,
        "q, k, v, o must be float16"
    );
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_q and cu_seqlens_k must be int32");

    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && o.dim() == 3, "q, k, v, o must be 3D tensors (Total_L, H, 64)");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q and cu_seqlens_k must be 1D tensors");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0) && cu_seqlens_q.size(0) >= 1, "cu_seqlens_q and cu_seqlens_k must have identical length >= 1");

    const int64_t Total_Lq = q.size(0);
    const int64_t H_q      = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t Total_Lk = k.size(0);
    const int64_t H_kv     = k.size(1);
    const int64_t B        = cu_seqlens_q.size(0) - 1;

    TORCH_CHECK(head_dim == 64 && k.size(2) == 64 && v.size(2) == 64 && o.size(2) == 64, "head_dim must be fixed to 64");
    TORCH_CHECK(o.size(0) == Total_Lq && o.size(1) == H_q, "o shape must match (Total_Lq, H_q, 64)");
    TORCH_CHECK(v.size(0) == Total_Lk && v.size(1) == H_kv, "v shape must match k shape (Total_Lk, H_kv, 64)");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be positive and divisible by H_kv for GQA");

    // Verify 128-bit vector alignment on strided THD tensors
    TORCH_CHECK(
        q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.stride(2) == 1,
        "q, k, v, o last dimension must have stride 1"
    );
    TORCH_CHECK(
        (q.stride(0) % 8 == 0) && (q.stride(1) % 8 == 0) &&
        (k.stride(0) % 8 == 0) && (k.stride(1) % 8 == 0) &&
        (v.stride(0) % 8 == 0) && (v.stride(1) % 8 == 0) &&
        (o.stride(0) % 8 == 0) && (o.stride(1) % 8 == 0),
        "q, k, v, o token and head strides must be multiples of 8 for 128-bit vector access"
    );

    const bool has_lse = lse_opt.has_value();
    if (has_lse) {
        TORCH_CHECK(lse_opt->is_cuda() && lse_opt->device() == q.device(), "lse must reside on the same CUDA device as q");
        TORCH_CHECK(lse_opt->is_contiguous(), "lse must be contiguous");
        TORCH_CHECK(lse_opt->scalar_type() == at::kFloat, "lse must be float32");
        TORCH_CHECK(lse_opt->dim() == 2 && lse_opt->size(0) == H_q && lse_opt->size(1) == Total_Lq, "lse must have shape (H_q, Total_Lq)");
    }

    const bool has_cos = cos.has_value();
    const bool has_sin = sin.has_value();
    TORCH_CHECK(has_cos == has_sin, "cos and sin must either both be provided or both be None");

    const bool has_rope = has_cos;
    int64_t l_max = 0;
    if (has_rope) {
        TORCH_CHECK(cos->is_cuda() && sin->is_cuda(), "cos and sin must be CUDA tensors");
        TORCH_CHECK(cos->device() == q.device() && sin->device() == q.device(), "cos and sin must reside on the same device as q");
        TORCH_CHECK(cos->is_contiguous() && sin->is_contiguous(), "cos and sin must be contiguous");
        TORCH_CHECK(cos->scalar_type() == at::kHalf && sin->scalar_type() == at::kHalf, "cos and sin must be float16");
        TORCH_CHECK(cos->dim() == 2 && sin->dim() == 2, "cos and sin must be 2D tensors (L_max, 64)");
        TORCH_CHECK(cos->size(1) == 64 && sin->size(1) == 64, "cos and sin last dimension must be 64");
        TORCH_CHECK(cos->size(0) == sin->size(0), "cos and sin must have matching L_max");
        if (Total_Lq > 0 || Total_Lk > 0) {
            TORCH_CHECK(cos->size(0) > 0, "cos and sin must have positive L_max for non-empty sequences");
        }
        l_max = cos->size(0);
    }

    const bool has_offsets = offsets.has_value();
    if (has_offsets) {
        TORCH_CHECK(offsets->is_cuda() && offsets->device() == q.device(), "offsets must reside on the same CUDA device as q");
        TORCH_CHECK(offsets->is_contiguous(), "offsets must be contiguous");
        TORCH_CHECK(offsets->scalar_type() == at::kInt, "offsets must be int32");
        TORCH_CHECK(offsets->dim() == 1 && offsets->size(0) == B, "offsets must be a 1D tensor of shape (B,)");
    }

    // 2. Kernel launch setup
    if (Total_Lq == 0 || max_seqlen_q <= 0 || B <= 0) {
        if (Total_Lq > 0) {
            o.zero_();
        }
        if (has_lse && lse_opt->numel() > 0) {
            lse_opt->fill_(-std::numeric_limits<float>::infinity());
        }
        return;
    }

    using Element = cutlass::half_t;
    constexpr int TILE_M      = 64;
    constexpr int TILE_N      = 64;
    constexpr int TILE_K      = 64;
    constexpr int NUM_THREADS = 128;
    constexpr int smem_size   = 2 * TILE_M * TILE_K * static_cast<int>(sizeof(Element)); // 16 KB
    const float scale         = 1.0f / std::sqrt(static_cast<float>(TILE_K));

    const int64_t num_q_blocks = (max_seqlen_q + TILE_M - 1) / TILE_M;
    dim3 grid(
        static_cast<unsigned int>(num_q_blocks),
        static_cast<unsigned int>(H_q),
        static_cast<unsigned int>(B)
    );
    dim3 block(NUM_THREADS, 1, 1);

    using SmemLayoutKMajor = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), make_stride(Int<TILE_K>{}, Int<1>{}))
    ));
    using SmemLayoutMNMajor = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<TILE_K>{}, Int<TILE_N>{}), make_stride(Int<1>{}, Int<TILE_K>{}))
    ));

    using CopyAtomG2R = Copy_Atom<UniversalCopy<uint4>, Element>;
    using TiledCopyG2R = decltype(make_tiled_copy(
        CopyAtomG2R{},
        make_layout(make_shape(Int<16>{}, Int<TILE_K / 8>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<8>{}), LayoutRight{})
    ));
    using CopyAtomLSE = Copy_Atom<UniversalCopy<float>, float>;

    using S2RCopyAtomA         = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomB_KMajor  = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomB_MNMajor = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    using MMAAtom      = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMAType = TiledMMA<MMAAtom, Layout<Shape<_4, _1, _1>, Stride<_1, _4, _4>>>;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const Element* q_ptr       = reinterpret_cast<const Element*>(q.data_ptr<at::Half>());
    const Element* k_ptr       = reinterpret_cast<const Element*>(k.data_ptr<at::Half>());
    const Element* v_ptr       = reinterpret_cast<const Element*>(v.data_ptr<at::Half>());
    const int32_t* cu_q_ptr    = cu_seqlens_q.data_ptr<int32_t>();
    const int32_t* cu_k_ptr    = cu_seqlens_k.data_ptr<int32_t>();
    const Element* cos_ptr     = has_rope ? reinterpret_cast<const Element*>(cos->data_ptr<at::Half>()) : nullptr;
    const Element* sin_ptr     = has_rope ? reinterpret_cast<const Element*>(sin->data_ptr<at::Half>()) : nullptr;
    const int32_t* offsets_ptr = has_offsets ? offsets->data_ptr<int32_t>() : nullptr;
    Element*       o_ptr       = reinterpret_cast<Element*>(o.data_ptr<at::Half>());
    float*         lse_ptr     = has_lse ? lse_opt->data_ptr<float>() : nullptr;

    const int64_t q_stride_l = q.stride(0);
    const int64_t q_stride_h = q.stride(1);
    const int64_t k_stride_l = k.stride(0);
    const int64_t k_stride_h = k.stride(1);
    const int64_t v_stride_l = v.stride(0);
    const int64_t v_stride_h = v.stride(1);
    const int64_t o_stride_l = o.stride(0);
    const int64_t o_stride_h = o.stride(1);

    {
        auto launch_fwd = [&](auto has_rope_tag, auto is_causal_tag) {
            constexpr bool kHasRope  = decltype(has_rope_tag)::value;
            constexpr bool kIsCausal = decltype(is_causal_tag)::value;

            kernel_fused_mha_varlen_fwd<
                Element,
                TILE_M,
                TILE_N,
                TILE_K,
                NUM_THREADS,
                kHasRope,
                kIsCausal,
                SmemLayoutKMajor,
                SmemLayoutMNMajor,
                TiledCopyG2R,
                CopyAtomLSE,
                TiledMMAType,
                S2RCopyAtomA,
                S2RCopyAtomB_KMajor,
                S2RCopyAtomB_MNMajor
            ><<<grid, block, smem_size, stream>>>(
                q_ptr,
                k_ptr,
                v_ptr,
                cu_q_ptr,
                cu_k_ptr,
                cos_ptr,
                sin_ptr,
                offsets_ptr,
                o_ptr,
                lse_ptr,
                q_stride_l,
                q_stride_h,
                k_stride_l,
                k_stride_h,
                v_stride_l,
                v_stride_h,
                o_stride_l,
                o_stride_h,
                Total_Lq,
                l_max,
                static_cast<int>(H_q),
                static_cast<int>(H_kv),
                scale
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        };

        if (has_rope) {
            if (is_causal) {
                launch_fwd(std::true_type{}, std::true_type{});
            } else {
                launch_fwd(std::true_type{}, std::false_type{});
            }
        } else {
            if (is_causal) {
                launch_fwd(std::false_type{}, std::true_type{});
            } else {
                launch_fwd(std::false_type{}, std::false_type{});
            }
        }
    }
}

// Public host wrapper for fused_mha_varlen_fwd.
// Returns (o: (Total_Lq, H_q, 64) [FP16], lse: (H_q, Total_Lq) [FP32]).

std::tuple<at::Tensor, at::Tensor> fused_mha_varlen_fwd(
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& k,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // scalar [int64]
    const std::optional<at::Tensor>& cos,                    // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,                    // (L_max, 64) [FP16] (optional)
    bool is_causal,                                          // scalar [bool]
    const std::optional<at::Tensor>& offsets                 // (B,) [Int32] (optional)
) {
    TORCH_CHECK(q.dim() == 3, "q must be a 3D tensor (Total_Lq, H_q, 64)");
    const int64_t Total_Lq = q.size(0);
    const int64_t H_q      = q.size(1);

    at::Tensor o   = torch::empty({Total_Lq, H_q, 64}, q.options());
    at::Tensor lse = torch::empty({H_q, Total_Lq}, q.options().dtype(at::kFloat));

    launch_fused_mha_varlen_fwd_into(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        o, lse, cos, sin, is_causal, offsets
    );

    return std::make_tuple(o, lse);
}
