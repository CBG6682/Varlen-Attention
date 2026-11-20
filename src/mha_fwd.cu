#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "mha_common.cuh"
#include "mha_ops.cuh"

using namespace cute;

// =========================================================================================
// PHẦN 1: GLOBAL KERNEL (kernel_fused_mha_varlen_fwd)
// =========================================================================================
// Tóm tắt Flow thực thi:
//   - Khởi tạo (Rule 6, 13, 14, 23): Xác định (block_q_idx, head_q_idx, seq_idx), đọc biên
//     cu_seqlens_q/k và offsets[seq_idx]; khai báo 1 lần toàn bộ tensor GMEM, SMEM (16 KB chia
//     sBuf0 8 KB + sBuf1 8 KB gắn Swizzle<3, 3, 3>), tensor phân mảnh, predication (tQpQ, tKpK,
//     tLpLSE) và cặp thanh ghi dùng chung (tCrA, tCrB) cho MMA.
//   - Prologue: Nạp block Q (g2r 128-bit) + áp dụng RoPE thuận trên thanh ghi (nếu HAS_ROPE),
//     ghi vào sBuf0 (r2s uint4), nạp qua LDSM_N vào tCrQ và giữ cố định suốt vòng đời kernel;
//     sau đó prefetch K_0 (+ RoPE) vào sBuf0.
//   - Main Loop (dừng sớm khi IS_CAUSAL && block_k > block_q):
//     + Pha 1: Prefetch V_curr (g2r) song song với LDSM_N nạp K từ sBuf0 và gemm S = Q @ K^T,
//       ghi V_curr xuống sBuf1 (r2s) và __syncthreads().
//     + Pha 2: Nhân scale, áp dụng Causal Mask & chặn biên lk_i, tính Online Softmax (FP32,
//       quad_reduce_max/sum), rescale tCrO, rồi chuyển tiếp trực tiếp P từ tCrS sang tCrA
//       trên thanh ghi (Rule 19 - Lựa chọn 1B, không qua SMEM và không cần __syncthreads()).
//     + Pha 3: Prefetch K_next (g2r + RoPE) song song với LDSM_T nạp V chuyển vị từ sBuf1 và
//       gemm O += P @ V; ghi K_next vào sBuf0 (đã rảnh từ cuối Pha 1) và __syncthreads().
//   - Epilogue: Chuẩn hóa tCrO /= l_i, ghi gLSE qua cute::copy_if (lane leader predication),
//     ép kiểu tCrO -> FP16 staging qua sBuf1 (Swizzle) rồi dùng TiledCopy 128-bit ghi ra gO.
// =========================================================================================

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
    const Element* __restrict__ gQ_ptr,          // shape: (H_q, Total_Lq, 64), dtype: FP16
    const Element* __restrict__ gK_ptr,          // shape: (H_kv, Total_Lk, 64), dtype: FP16
    const Element* __restrict__ gV_ptr,          // shape: (H_kv, Total_Lk, 64), dtype: FP16
    const int32_t* __restrict__ cu_seqlens_q,    // shape: (B + 1,), dtype: int32
    const int32_t* __restrict__ cu_seqlens_k,    // shape: (B + 1,), dtype: int32
    const Element* __restrict__ gCos_ptr,        // shape: (L_max, 64), dtype: FP16 (nullptr nếu !HAS_ROPE)
    const Element* __restrict__ gSin_ptr,        // shape: (L_max, 64), dtype: FP16 (nullptr nếu !HAS_ROPE)
    const int32_t* __restrict__ offsets_ptr,     // shape: (B,), dtype: int32 (nullptr nếu offset = 0)
    Element*       __restrict__ gO_ptr,          // shape: (H_q, Total_Lq, 64), dtype: FP16
    float*         __restrict__ gLSE_ptr,        // shape: (H_q, Total_Lq), dtype: FP32
    int64_t total_lq,                            // scalar: int64
    int64_t total_lk,                            // scalar: int64
    int64_t l_max,                               // scalar: int64
    int H_q,                                     // scalar: int
    int H_kv,                                    // scalar: int
    float scale                                  // scalar: float (1 / sqrt(64) = 0.125)
) {
    // -------------------------------------------------------------------------------------
    // Giai đoạn 1: Tính toán Khởi tạo & Đọc biên Varlen (Rule 6, 13, 14)
    // -------------------------------------------------------------------------------------
    const int block_q_idx = static_cast<int>(blockIdx.x);
    const int head_q_idx  = static_cast<int>(blockIdx.y);
    const int seq_idx     = static_cast<int>(blockIdx.z);

    const int rep         = H_q / H_kv;
    const int head_kv_idx = head_q_idx / rep;

    // Đọc biên chuỗi và offset trực tiếp từ con trỏ GMEM (Rule 14)
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

    // Dịch con trỏ cơ sở tới vị trí (head, seq_start) tương ứng
    const Element* q_base_ptr   = gQ_ptr   + (static_cast<int64_t>(head_q_idx) * total_lq + q_start) * TILE_K;
    const Element* k_base_ptr   = gK_ptr   + (static_cast<int64_t>(head_kv_idx) * total_lk + k_start) * TILE_K;
    const Element* v_base_ptr   = gV_ptr   + (static_cast<int64_t>(head_kv_idx) * total_lk + k_start) * TILE_K;
    Element*       o_base_ptr   = gO_ptr   + (static_cast<int64_t>(head_q_idx) * total_lq + q_start) * TILE_K;
    float*         lse_head_ptr = gLSE_ptr + (static_cast<int64_t>(head_q_idx) * total_lq + q_start);

    const Element* cos_seq_ptr  = HAS_ROPE ? (gCos_ptr + static_cast<int64_t>(seq_offset) * TILE_K) : nullptr;
    const Element* sin_seq_ptr  = HAS_ROPE ? (gSin_ptr + static_cast<int64_t>(seq_offset) * TILE_K) : nullptr;
    const int64_t  cos_rows     = HAS_ROPE ? (l_max - static_cast<int64_t>(seq_offset)) : 0;

    // -------------------------------------------------------------------------------------
    // Giai đoạn 2: Khai báo & Phân mảnh Tensor 1 lần duy nhất ở đầu Kernel (Rule 6, 23)
    // -------------------------------------------------------------------------------------
    // Phân bổ 16 KB SMEM động thành 2 buffer 8 KB (sBuf0 cho Q/K, sBuf1 cho V/O)
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Element* sBuf0_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* sBuf1_ptr = reinterpret_cast<Element*>(smem_raw + TILE_M * TILE_K * sizeof(Element));

    Tensor sBuf0      = make_tensor(make_smem_ptr(sBuf0_ptr), SmemLayoutKMajor{});
    Tensor sBuf1      = make_tensor(make_smem_ptr(sBuf1_ptr), SmemLayoutKMajor{});
    Tensor sBuf1_V_mn = make_tensor(make_smem_ptr(sBuf1_ptr), SmemLayoutMNMajor{});

    // Tensor GMEM dạng packed cho chuỗi hiện tại
    Tensor mQ   = make_tensor(make_gmem_ptr(q_base_ptr),  make_shape(lq_i, Int<TILE_K>{}),     make_stride(Int<TILE_K>{}, Int<1>{}));
    Tensor mK   = make_tensor(make_gmem_ptr(k_base_ptr),  make_shape(lk_i, Int<TILE_K>{}),     make_stride(Int<TILE_K>{}, Int<1>{}));
    Tensor mV   = make_tensor(make_gmem_ptr(v_base_ptr),  make_shape(lk_i, Int<TILE_K>{}),     make_stride(Int<TILE_K>{}, Int<1>{}));
    Tensor mO   = make_tensor(make_gmem_ptr(o_base_ptr),  make_shape(lq_i, Int<TILE_K>{}),     make_stride(Int<TILE_K>{}, Int<1>{}));
    Tensor mCos = make_tensor(make_gmem_ptr(cos_seq_ptr), make_shape(cos_rows, Int<TILE_K>{}), make_stride(Int<TILE_K>{}, Int<1>{}));
    Tensor mSin = make_tensor(make_gmem_ptr(sin_seq_ptr), make_shape(cos_rows, Int<TILE_K>{}), make_stride(Int<TILE_K>{}, Int<1>{}));

    // Cắt tile Q và O cho block hiện tại
    Tensor gQ = local_tile(mQ, make_tile(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(block_q_idx, 0));
    Tensor gO = local_tile(mO, make_tile(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(block_q_idx, 0));

    // Phân mảnh GMEM <-> SMEM bằng TiledCopy 128-bit
    TiledCopyG2R tiled_copy_g2r;
    auto thr_copy_g2r = tiled_copy_g2r.get_slice(tid);

    auto tQgQ    = thr_copy_g2r.partition_S(gQ);
    auto tQsBuf0 = thr_copy_g2r.partition_D(sBuf0);
    auto tVsBuf1 = thr_copy_g2r.partition_D(sBuf1);
    auto tOsBuf1 = thr_copy_g2r.partition_S(sBuf1);
    auto tOgO    = thr_copy_g2r.partition_D(gO);

    // Khai báo fragment thanh ghi staging căn chỉnh 16-byte cho g2r -> RoPE -> r2s
    constexpr int FRAG_ELEMS = (TILE_M * TILE_K) / NUM_THREADS;
    auto frag_layout = make_layout(shape(tQgQ));
    alignas(16) Element buf_stage[FRAG_ELEMS];
    alignas(16) Element buf_cos[FRAG_ELEMS];
    alignas(16) Element buf_sin[FRAG_ELEMS];
    Tensor rStage = make_tensor(make_rmem_ptr(buf_stage), frag_layout);
    Tensor rCos   = make_tensor(make_rmem_ptr(buf_cos),   frag_layout);
    Tensor rSin   = make_tensor(make_rmem_ptr(buf_sin),   frag_layout);

    // Tính toán Predication chặn biên 1 lần duy nhất trước vòng lặp (Rule 13)
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

    // Khai báo tensor và predication ghi gLSE cho lane leader (Rule 14, 18)
    Tensor tLgLSE = make_tensor(
        make_gmem_ptr(lse_head_ptr + q_start_row + warp_id * 16 + quad_id),
        make_shape(Int<1>{}, Int<2>{}),
        make_stride(Int<1>{}, Int<8>{})
    );
    Tensor tLrLSE = make_tensor<float>(make_shape(Int<1>{}, Int<2>{}), make_stride(Int<1>{}, Int<1>{}));
    Tensor tLpLSE = make_tensor<bool>(make_shape(Int<1>{}, Int<2>{}), make_stride(Int<1>{}, Int<1>{}));
    tLpLSE(0, 0) = (lane_in_quad == 0) && (global_row_a < lq_i);
    tLpLSE(0, 1) = (lane_in_quad == 0) && (global_row_b < lq_i);

    // Khai báo TiledMMA, S2R Copy Atoms và Fragment (Rule 19: đúng 1 cặp dùng chung tCrA, tCrB)
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

    // tCrQ giữ Q cố định trên thanh ghi; (tCrA, tCrB) dùng chung cho S = Q@K^T và O += P@V
    Tensor tCrQ = thr_mma.partition_fragment_A(sBuf0);
    Tensor tCrA = thr_mma.partition_fragment_A(sBuf0);
    Tensor tCrB = thr_mma.partition_fragment_B(sBuf0);

    auto tCrQ_s2r_view = thr_s2r_a.retile_D(tCrQ);
    auto tCrB_km_view  = thr_s2r_b_km.retile_D(tCrB);
    auto tCrB_mn_view  = thr_s2r_b_mn.retile_D(tCrB);

    // -------------------------------------------------------------------------------------
    // Giai đoạn 3: Khởi tạo Accumulator, Online Softmax & Helper Lambdas (Rule 6, 22)
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

    // Lambda g2r: Nạp vector 128-bit từ GMEM vào thanh ghi staging
    auto g2r_load = [&](auto const& tSrc, auto& rDst, auto const& pred, bool in_bounds) {
        if (in_bounds) {
            cute::copy(tiled_copy_g2r, tSrc, rDst);
        } else {
            cute::clear(rDst);
            cute::copy_if(tiled_copy_g2r, pred, tSrc, rDst);
        }
    };

    // Lambda r2s: Ghi trực tiếp 128-bit uint4 qua phân vùng Swizzled SMEM (Rule 24B)
    auto r2s_store = [](auto const& rSrc, auto& tSmemD) {
        #pragma unroll
        for (int m = 0; m < size<1>(rSrc); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(rSrc); ++n) {
                *reinterpret_cast<uint4*>(&tSmemD(0, m, n)) = *reinterpret_cast<const uint4*>(&rSrc(0, m, n));
            }
        }
    };

    // Lambda RoPE thuận: Xoay trực tiếp trên thanh ghi staging dùng __shfl_xor_sync(..., 4)
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
    // Bước 2 (Prologue): Nạp Q (+ RoPE) giữ cố định vào tCrQ, sau đó Prefetch K_0 vào sBuf0
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

    // Đọc Q đã xoay từ sBuf0 lên tCrQ qua LDSM_N và giữ cố định suốt vòng đời kernel
    cute::copy(s2r_copy_a, tAsBuf0_s2r, tCrQ_s2r_view);
    __syncthreads();

    // Prefetch block K_0 (+ RoPE) vào sBuf0 chuẩn bị cho vòng lặp đầu tiên
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
    // Bước 3 (Main Loop): Pipeline 2 lần đọc / 1 stage qua các block K, V của chuỗi seq_idx
    // -------------------------------------------------------------------------------------
    for (int k_tile = 0; k_tile < num_k_blocks; ++k_tile) {
        const int  k_start_col      = k_tile * TILE_N;
        const bool curr_k_in_bounds = (k_tile < last_k_block_idx) || last_k_in_bounds;

        // --- Pha 1: S = Q @ K^T song song với Prefetch V_curr ---
        Tensor gV_curr = local_tile(mV, make_tile(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(k_tile, 0));
        g2r_load(thr_copy_g2r.partition_S(gV_curr), rStage, tKpK, curr_k_in_bounds);

        // Nạp trọn vẹn K từ sBuf0 vào tCrB (Rule 24B) rồi thực hiện GEMM S = Q @ K^T
        cute::copy(s2r_copy_b_km, tBsBuf0_s2r, tCrB_km_view);
        cute::clear(tCrS);
        cute::gemm(tiled_mma, tCrQ, tCrB, tCrS);

        // Xả V_curr đã prefetch vào sBuf1 (r2s) và đồng bộ cho Pha 3
        r2s_store(rStage, tVsBuf1);
        __syncthreads();

        // --- Pha 2: Scale, Causal Mask, Online Softmax & Chuyển tiếp trực tiếp P -> tCrA (Rule 19) ---
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
            // Rescale accumulator tCrO theo tỉ lệ alpha mới
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

            // Rule 19 (Lựa chọn 1B): Chuyển tiếp trực tiếp tCrS -> tCrA trên thanh ghi (0 SMEM, 0 sync)
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

        // --- Pha 3: O += P @ V song song với Prefetch K_next ---
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

        // Nạp V chuyển vị bằng LDSM_T từ sBuf1 vào tCrB (Rule 16, 24B) rồi tính O += P @ V
        cute::copy(s2r_copy_b_mn, tBsBuf1_s2r, tCrB_mn_view);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrO);

        // Xoay RoPE và xả K_next xuống sBuf0 (sBuf0 đã hoàn tất đọc từ cuối Pha 1)
        if (has_next) {
            if constexpr (HAS_ROPE) {
                apply_rope_reg();
            }
            r2s_store(rStage, tQsBuf0);
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------------------
    // Bước 4 (Epilogue): Chuẩn hóa O, ghi gLSE và Staging O qua sBuf1 -> gO (Rule 14, 17, 18)
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

    // Ghi LSE (FP32) ra GMEM bằng CopyAtom với lane leader predication (Rule 14, 18)
    tLrLSE(0, 0) = (l_old_a > 0.0f && m_old_a != NEG_INF) ? (m_old_a + __logf(l_old_a)) : NEG_INF;
    tLrLSE(0, 1) = (l_old_b > 0.0f && m_old_b != NEG_INF) ? (m_old_b + __logf(l_old_b)) : NEG_INF;
    cute::copy_if(CopyAtomLSE{}, tLpLSE, tLrLSE, tLgLSE);

    // Quy trình 2 bước (Rule 17, 18): Ép kiểu sang fragment đệm FP16 -> R2S vào sBuf1 có Swizzle
    Tensor rO_fp16 = make_tensor<Element>(shape(tCrO));
    #pragma unroll
    for (int i = 0; i < size(tCrO); ++i) {
        rO_fp16(i) = float_to_out<Element>(tCrO(i));
    }
    cute::copy(rO_fp16, tCsBuf1_C);
    __syncthreads();

    // S2G: Đọc từ sBuf1 qua Swizzle và ghi vector hóa 128-bit ra gO
    if (q_in_bounds) {
        cute::copy(tiled_copy_g2r, tOsBuf1, tOgO);
    } else {
        cute::copy_if(tiled_copy_g2r, tQpQ, tOsBuf1, tOgO);
    }
}

// =========================================================================================
// PHẦN 2: HOST FUNCTION (fused_mha_varlen_fwd)
// =========================================================================================
// Tóm tắt Flow thực thi:
//   - Giai đoạn 1 (Torch Check): Kiểm tra thiết bị CUDA, tính liên tục, kiểu dữ liệu,
//     số chiều, ràng buộc head_dim = 64 và H_q % H_kv == 0 (cũng như cos/sin/offsets nếu có).
//   - Giai đoạn 2 (Launch Procedure 8 bước): Cấp phát trước o (FP16) và lse (FP32);
//     khai báo hằng số template (TILE 64x64x64, 128 threads, 16 KB SMEM), Layout SMEM gắn
//     Swizzle<3, 3, 3>, Copy Atom 128-bit & LDSM, TiledMMA Shape<_4, _1, _1>; lấy CUDA stream
//     và launch kernel trong khối scope { ... } riêng biệt theo <HAS_ROPE, IS_CAUSAL>.
//   - Giai đoạn 3 (Return): Trả về tuple (o, lse).
// Shapes & Dtypes trả về:
//   1. o:   (H_q, Total_Lq, 64) [FP16] - Kết quả Attention raw dạng packed
//   2. lse: (H_q, Total_Lq)     [FP32] - Log-Sum-Exp dùng cho Backward
// =========================================================================================

std::tuple<at::Tensor, at::Tensor> fused_mha_varlen_fwd(
    const at::Tensor& q,                                     // (H_q, Total_Lq, 64) [FP16]
    const at::Tensor& k,                                     // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& v,                                     // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // scalar [int64]
    const std::optional<at::Tensor>& cos,                    // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,                    // (L_max, 64) [FP16] (optional)
    bool is_causal,                                          // scalar [bool]
    const std::optional<at::Tensor>& offsets                 // (B,) [Int32] (optional)
) {
    // -------------------------------------------------------------------------------------
    // Giai đoạn 1: Xác thực Dữ liệu (Torch Check - Rule 5)
    // -------------------------------------------------------------------------------------
    at::cuda::CUDAGuard device_guard(q.device());

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q and cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(
        k.device() == q.device() && v.device() == q.device() &&
        cu_seqlens_q.device() == q.device() && cu_seqlens_k.device() == q.device(),
        "All input tensors must reside on the same CUDA device"
    );

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, v must be contiguous");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous");

    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf, "q, k, v must be float16");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_q and cu_seqlens_k must be int32");

    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be 3D tensors (H, Total_L, D)");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q and cu_seqlens_k must be 1D tensors");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0) && cu_seqlens_q.size(0) >= 2, "cu_seqlens_q and cu_seqlens_k must have identical length >= 2");

    const int64_t H_q      = q.size(0);
    const int64_t Total_Lq = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t H_kv     = k.size(0);
    const int64_t Total_Lk = k.size(1);
    const int64_t B        = cu_seqlens_q.size(0) - 1;

    TORCH_CHECK(head_dim == 64 && k.size(2) == 64 && v.size(2) == 64, "head_dim must be fixed to 64");
    TORCH_CHECK(v.size(0) == H_kv && v.size(1) == Total_Lk, "v shape must match k shape (H_kv, Total_Lk, 64)");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be positive and divisible by H_kv for GQA");

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
        TORCH_CHECK(cos->size(0) == sin->size(0) && cos->size(0) > 0, "cos and sin must have matching positive L_max");
        l_max = cos->size(0);
    }

    const bool has_offsets = offsets.has_value();
    if (has_offsets) {
        TORCH_CHECK(offsets->is_cuda() && offsets->device() == q.device(), "offsets must reside on the same CUDA device as q");
        TORCH_CHECK(offsets->is_contiguous(), "offsets must be contiguous");
        TORCH_CHECK(offsets->scalar_type() == at::kInt, "offsets must be int32");
        TORCH_CHECK(offsets->dim() == 1 && offsets->size(0) == B, "offsets must be a 1D tensor of shape (B,)");
    }

    // -------------------------------------------------------------------------------------
    // Giai đoạn 2: Quy trình Khởi chạy Kernel 8 bước (Rule 5, 8, 10, 11, 15, 16)
    // -------------------------------------------------------------------------------------
    // Bước 1: Cấp phát trước tensor output o (FP16) và lse (FP32)
    at::Tensor o   = torch::empty_like(q);
    at::Tensor lse = torch::empty({H_q, Total_Lq}, q.options().dtype(at::kFloat));

    if (Total_Lq == 0 || max_seqlen_q <= 0 || B <= 0) {
        return std::make_tuple(o, lse);
    }

    // Bước 2: Hằng số & Cấu hình Launch
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

    // Bước 3: Khai báo Layout SMEM gắn Swizzle<3, 3, 3> (Rule 15, 16)
    using SmemLayoutKMajor = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), make_stride(Int<TILE_K>{}, Int<1>{}))
    ));
    using SmemLayoutMNMajor = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<TILE_K>{}, Int<TILE_N>{}), make_stride(Int<1>{}, Int<TILE_K>{}))
    ));

    // Bước 4: Khai báo Copy Atom G2R/S2G 128-bit, LSE Copy Atom và LDSM S2R (Rule 12, 16, 24B)
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

    // Bước 5: Khai báo MMA Atom & TiledMMA Shape<_4, _1, _1> (Rule 15, 19 - Lựa chọn 1B)
    using MMAAtom      = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMAType = TiledMMA<MMAAtom, Layout<Shape<_4, _1, _1>, Stride<_1, _4, _4>>>;

    // Bước 6: Lấy CUDA Stream hiện tại
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
    float*         lse_ptr     = lse.data_ptr<float>();

    // Bước 7: Cô lập Launch Kernel trong khối scope riêng biệt (Rule 8, 10)
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
                Total_Lq,
                Total_Lk,
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

    // Bước 8: Không có tensor trung gian cần giải phóng

    // -------------------------------------------------------------------------------------
    // Giai đoạn 3: Trả về Kết quả (Rule 5)
    // -------------------------------------------------------------------------------------
    return std::make_tuple(o, lse);
}
