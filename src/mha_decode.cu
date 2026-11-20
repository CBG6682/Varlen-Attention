// Kernel CUDA/CuTe tính Flash Decoding + Fused RoPE biến thiên độ dài (Varlen, L_q = 1) cho kiến trúc SM75 (Turing)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "mha_common.cuh"
#include "mha_ops.cuh"

using namespace cute;

// Ngưỡng sentinel âm hữu hạn tương thích cờ --use_fast_math (tránh -inf - (-inf) = NaN)
static constexpr float DECODE_NEG_INF = -1e30f;
static constexpr float DECODE_VALID_THRESH = -1e29f;

// =================================================================================================
// PHẦN 1: DEVICE KERNELS (__global__)
// =================================================================================================

/*
 * Tóm tắt luồng thực thi kernel_fused_mha_varlen_decode_splitk (Không dùng MMA):
 *   Grid = (num_splits, H_q, B), Block = (128, 1, 1) (4 warps).
 *   Mỗi CTA xử lý 1 phân đoạn token K/V thuộc [split_start, split_end) của cặp (head_q_idx, seq_idx):
 *   - Giai đoạn 1 (Khởi tạo): Đọc biên chuỗi k_start/k_end, chia phạm vi split; nếu split rỗng thì ghi
 *     o_part = 0, lse_part = DECODE_NEG_INF và return sớm. Nạp broadcast q, áp RoPE tại (offset + lk_i - 1)
 *     và nhân sm_scale ở FP32; khai báo một lần SMEM (4,224 B), Copy Atoms và 2-stage Prefetch buffers.
 *   - Giai đoạn 2 (Main Loop - Rule 24A 2-stage Register Prefetch): Duyệt các tile 16 token K/V, nạp xen kẽ
 *     vào s0/s1 bằng uint4 (áp RoPE cho raw K_cache tại vị trí offset + t), tính dot-product FP32 qua
 *     octet_reduce_sum_f32 và cập nhật Online Softmax + tích lũy o_acc (FP32) cục bộ theo từng hàng token.
 *   - Giai đoạn 3 (Epilogue): Ghi rO_acc và (m_local, l_local) xuống SMEM swizzled (Swizzle<3, 2, 4>),
 *     __syncthreads(), rồi gộp 16 hàng về 1 vector theo bố cục 8x16 LayoutLeft bằng octet reduction (FP32)
 *     và ghi ra gO_part (uint4), gLSE_part (lane leader) bằng cute::copy_if.
 */
template <
    int HEAD_DIM = 64,
    int NUM_THREADS = 128,
    int TILE_K_TOKENS = 16,
    bool HAS_ROPE = true
>
__global__ void __launch_bounds__(NUM_THREADS)
kernel_fused_mha_varlen_decode_splitk(
    const cutlass::half_t* __restrict__ Q_ptr,        // (H_q, B, 64) [FP16] - Query decode hiện tại
    const cutlass::half_t* __restrict__ K_ptr,        // (H_kv, Total_Lk, 64) [FP16] - Raw Key cache packed
    const cutlass::half_t* __restrict__ V_ptr,        // (H_kv, Total_Lk, 64) [FP16] - Value cache packed
    const int32_t* __restrict__ cu_seqlens_k_ptr,     // (B + 1,) [Int32] - Độ dài tích lũy K, V
    const cutlass::half_t* __restrict__ cos_ptr,      // (L_max, 64) [FP16] - Bảng cos RoPE (hoặc nullptr)
    const cutlass::half_t* __restrict__ sin_ptr,      // (L_max, 64) [FP16] - Bảng sin RoPE (hoặc nullptr)
    const int32_t* __restrict__ offsets_ptr,          // (B,) [Int32] - Offset RoPE từng chuỗi (hoặc nullptr)
    float* __restrict__ O_part_ptr,                   // (H_q, B, num_splits, 64) [FP32] - Output cục bộ mỗi split
    float* __restrict__ LSE_part_ptr,                 // (H_q, B, num_splits) [FP32] - LSE cục bộ mỗi split
    int H_q,                                          // scalar [int] - Số head Query
    int H_kv,                                         // scalar [int] - Số head Key/Value
    int B,                                            // scalar [int] - Batch size
    int Total_Lk,                                     // scalar [int] - Tổng số token trong KV cache
    int num_splits,                                   // scalar [int] - Tổng số phân đoạn Split-K
    float sm_scale                                    // scalar [float] - Hệ số scale 1 / sqrt(64)
) {
    // ==================== GIAI ĐOẠN 1: KHỞI TẠO & CẤU HÌNH TENSOR ====================
    const int tid = static_cast<int>(threadIdx.x);
    const int split_idx = static_cast<int>(blockIdx.x);
    const int head_q_idx = static_cast<int>(blockIdx.y);
    const int seq_idx = static_cast<int>(blockIdx.z);
    const int head_kv_idx = head_q_idx / (H_q / H_kv);

    const int k_start = cu_seqlens_k_ptr[seq_idx];
    const int k_end = cu_seqlens_k_ptr[seq_idx + 1];
    const int lk_i = k_end - k_start;

    const int tokens_per_split = (lk_i + num_splits - 1) / num_splits;
    const int split_start = split_idx * tokens_per_split;
    const int split_end = min(split_start + tokens_per_split, lk_i);
    const int split_len = split_end - split_start;

    // Bố trí Dynamic Shared Memory (4,224 bytes) phục vụ gộp 16 -> 1 ở Epilogue
    extern __shared__ uint8_t smem_raw[];
    float* sO_ptr = reinterpret_cast<float*>(smem_raw);                    // 16 x 64 floats (4,096 B)
    float* sM_ptr = sO_ptr + TILE_K_TOKENS * HEAD_DIM;                     // 16 floats (64 B)
    float* sL_ptr = sM_ptr + TILE_K_TOKENS;                                // 16 floats (64 B)

    using Element = cutlass::half_t;
    using SmemLayoutReduceO = decltype(composition(
        Swizzle<3, 2, 4>{},
        Layout<Shape<Int<TILE_K_TOKENS>, Int<HEAD_DIM>>, Stride<Int<HEAD_DIM>, _1>>{}));

    // Copy Atom 16x8 LayoutRight: mỗi hàng 16 token do 8 threads nạp (8 halfs = 1 uint4/thread)
    using TiledCopyKV = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        make_layout(Shape<_16, _8>{}, LayoutRight{}),
        make_layout(Shape<_1, _8>{}, LayoutRight{})));

    // Copy Atom 16x8 FP32 ghi o_acc từ thanh ghi xuống sO
    using TiledCopyF32_16x8 = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        make_layout(Shape<_16, _8>{}, LayoutRight{}),
        make_layout(Shape<_1, _8>{}, LayoutRight{})));

    // Copy Atom 8x16 LayoutLeft đọc sO để giảm 16 -> 1 theo 8 cặp hàng song song và ghi gO_part
    using TiledCopyReduceRead = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        make_layout(Shape<_8, _16>{}, LayoutLeft{}),
        make_layout(Shape<_1, _4>{}, LayoutRight{})));

    // Copy Atom 1D ghi LSE_part qua predication leader (Rule 14)
    using TiledCopyStoreLSE = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<float>, float>{},
        make_layout(Shape<Int<NUM_THREADS>, _1>{}, LayoutRight{}),
        make_layout(Shape<_1, _1>{}, LayoutRight{})));

    TiledCopyKV tiled_copy_kv;
    auto thr_copy_kv = tiled_copy_kv.get_thread_slice(tid);

    TiledCopyF32_16x8 tiled_copy_f32_16x8;
    auto thr_copy_f32_16x8 = tiled_copy_f32_16x8.get_thread_slice(tid);

    TiledCopyReduceRead tiled_copy_reduce;
    auto thr_copy_reduce = tiled_copy_reduce.get_thread_slice(tid);

    TiledCopyStoreLSE tiled_copy_lse;
    auto thr_copy_lse = tiled_copy_lse.get_thread_slice(tid);

    const int row_in_tile = tid >> 3;      // 0..15
    const int col_vec = tid & 7;           // 0..7 (8 phần tử FP16 / hàng)
    const bool is_second_half = (col_vec >= 4);
    const int row8 = tid & 7;              // 0..7 (bước giảm Epilogue 8x16)

    Tensor sO = make_tensor(make_smem_ptr(sO_ptr), SmemLayoutReduceO{});
    Tensor tOsO_W = thr_copy_f32_16x8.partition_D(sO);
    Tensor tOsO_R = thr_copy_reduce.partition_S(sO);

    // GMEM output buffers cho split hiện tại (chỉ lane row8 == 0 / tid == 0 được ghi qua copy_if)
    const int64_t lse_part_offset =
        (static_cast<int64_t>(head_q_idx) * B + seq_idx) * num_splits + split_idx;
    const int64_t o_part_offset = lse_part_offset * HEAD_DIM;

    Tensor gO_part_bcast = make_tensor(
        make_gmem_ptr(O_part_ptr + o_part_offset),
        make_shape(_8{}, Int<HEAD_DIM>{}),
        make_stride(_0{}, _1{}));
    Tensor tOgO_part = thr_copy_reduce.partition_D(gO_part_bcast);
    Tensor rO_out = make_fragment_like(tOgO_part);

    Tensor tPredStoreO = make_tensor<bool>(make_shape(size<1>(tOgO_part), size<2>(tOgO_part)));
    tPredStoreO(0, 0) = (row8 == 0);

    Tensor gLSE_bcast = make_tensor(
        make_gmem_ptr(LSE_part_ptr + lse_part_offset),
        make_shape(Int<NUM_THREADS>{}, _1{}),
        make_stride(_0{}, _1{}));
    Tensor tLgLSE = thr_copy_lse.partition_D(gLSE_bcast);
    Tensor rLSE = make_fragment_like(tLgLSE);

    Tensor tPredLeader = make_tensor<bool>(make_shape(size<1>(tLgLSE), size<2>(tLgLSE)));
    tPredLeader(0, 0) = (tid == 0);

    // Thoát sớm đồng nhất nếu phân đoạn split này rỗng hoặc chuỗi có lk_i == 0
    if (split_len <= 0) {
        clear(rO_out);
        rLSE(0) = DECODE_NEG_INF;
        cute::copy_if(tiled_copy_reduce, tPredStoreO, rO_out, tOgO_part);
        cute::copy_if(tiled_copy_lse, tPredLeader, rLSE, tLgLSE);
        return;
    }

    const int offset = (HAS_ROPE && offsets_ptr != nullptr) ? offsets_ptr[seq_idx] : 0;

    // Khai báo trước một lần các fragment cho 2-stage Register Prefetch (Rule 23 & 24A)
    Tensor gQ_bcast = make_tensor(
        make_gmem_ptr(Q_ptr + (static_cast<int64_t>(head_q_idx) * B + seq_idx) * HEAD_DIM),
        make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
        make_stride(_0{}, _1{}));
    Tensor tQgQ = thr_copy_kv.partition_S(gQ_bcast);
    Tensor rQ = make_fragment_like(tQgQ);

    Tensor rK_s0 = make_fragment_like(tQgQ);
    Tensor rV_s0 = make_fragment_like(tQgQ);
    Tensor rCos_s0 = make_fragment_like(tQgQ);
    Tensor rSin_s0 = make_fragment_like(tQgQ);

    Tensor rK_s1 = make_fragment_like(tQgQ);
    Tensor rV_s1 = make_fragment_like(tQgQ);
    Tensor rCos_s1 = make_fragment_like(tQgQ);
    Tensor rSin_s1 = make_fragment_like(tQgQ);

    Tensor rO_acc = make_fragment_like(tOsO_W);
    Tensor rO_pair = make_fragment_like(tOsO_R);
    CUTE_STATIC_ASSERT_V(size(rK_s0) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size(rO_acc) == Int<8>{});

    // Nạp broadcast vector q (16 hàng trỏ chung địa chỉ để phần cứng L1/LSU gộp giao dịch - Rule 14)
    cute::copy(tiled_copy_kv, tQgQ, rQ);
    if constexpr (HAS_ROPE) {
        const int64_t q_rope_offset = static_cast<int64_t>(offset + lk_i - 1) * HEAD_DIM;
        Tensor gCos_q_bcast = make_tensor(
            make_gmem_ptr(cos_ptr + q_rope_offset),
            make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
            make_stride(_0{}, _1{}));
        Tensor gSin_q_bcast = make_tensor(
            make_gmem_ptr(sin_ptr + q_rope_offset),
            make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
            make_stride(_0{}, _1{}));
        cute::copy(tiled_copy_kv, thr_copy_kv.partition_S(gCos_q_bcast), rCos_s0);
        cute::copy(tiled_copy_kv, thr_copy_kv.partition_S(gSin_q_bcast), rSin_s0);

        apply_rope_fwd_8elem_shfl(
            &rQ(0, 0, 0),
            reinterpret_cast<const __half2*>(&rCos_s0(0, 0, 0)),
            reinterpret_cast<const __half2*>(&rSin_s0(0, 0, 0)),
            is_second_half);
    }

    float rQ_f32[8];
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
        rQ_f32[v] = half_to_float(rQ(v)) * sm_scale;
    }

    // Tham số lặp các tile 16 token KV và predicate cho tile cuối cùng
    const int num_tiles = (split_len + TILE_K_TOKENS - 1) / TILE_K_TOKENS;
    const int last_valid_rows = split_end - (split_start + (num_tiles - 1) * TILE_K_TOKENS);
    Tensor tPredLast = make_tensor<bool>(make_shape(size<1>(tQgQ), size<2>(tQgQ)));
    tPredLast(0, 0) = (row_in_tile < last_valid_rows);

    const int64_t kv_head_base = static_cast<int64_t>(head_kv_idx) * Total_Lk * HEAD_DIM;

    // ==================== GIAI ĐOẠN 2: MAIN LOOP (RULE 24A PREFETCH 2 STAGE) ====================
    clear(rO_acc);
    float m_local = DECODE_NEG_INF;
    float l_local = 0.0f;

    auto load_tile = [&](int tile_idx, auto& rK_buf, auto& rV_buf, auto& rCos_buf, auto& rSin_buf) {
        const int token_base = split_start + tile_idx * TILE_K_TOKENS;
        const int64_t kv_tile_off = kv_head_base + static_cast<int64_t>(k_start + token_base) * HEAD_DIM;

        Tensor gK_tile = make_tensor(
            make_gmem_ptr(K_ptr + kv_tile_off),
            make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
            make_stride(Int<HEAD_DIM>{}, _1{}));
        Tensor gV_tile = make_tensor(
            make_gmem_ptr(V_ptr + kv_tile_off),
            make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
            make_stride(Int<HEAD_DIM>{}, _1{}));

        Tensor tKgK = thr_copy_kv.partition_S(gK_tile);
        Tensor tVgV = thr_copy_kv.partition_S(gV_tile);

        const bool is_last_tile = (tile_idx == num_tiles - 1);
        if (!is_last_tile) {
            cute::copy(tiled_copy_kv, tKgK, rK_buf);
            cute::copy(tiled_copy_kv, tVgV, rV_buf);
        } else {
            clear(rK_buf);
            clear(rV_buf);
            cute::copy_if(tiled_copy_kv, tPredLast, tKgK, rK_buf);
            cute::copy_if(tiled_copy_kv, tPredLast, tVgV, rV_buf);
        }

        if constexpr (HAS_ROPE) {
            const int64_t rope_tile_off = static_cast<int64_t>(offset + token_base) * HEAD_DIM;
            Tensor gCos_tile = make_tensor(
                make_gmem_ptr(cos_ptr + rope_tile_off),
                make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
                make_stride(Int<HEAD_DIM>{}, _1{}));
            Tensor gSin_tile = make_tensor(
                make_gmem_ptr(sin_ptr + rope_tile_off),
                make_shape(Int<TILE_K_TOKENS>{}, Int<HEAD_DIM>{}),
                make_stride(Int<HEAD_DIM>{}, _1{}));

            Tensor tCos = thr_copy_kv.partition_S(gCos_tile);
            Tensor tSin = thr_copy_kv.partition_S(gSin_tile);

            if (!is_last_tile) {
                cute::copy(tiled_copy_kv, tCos, rCos_buf);
                cute::copy(tiled_copy_kv, tSin, rSin_buf);
            } else {
                clear(rCos_buf);
                clear(rSin_buf);
                cute::copy_if(tiled_copy_kv, tPredLast, tCos, rCos_buf);
                cute::copy_if(tiled_copy_kv, tPredLast, tSin, rSin_buf);
            }
        }
    };

    auto consume_tile = [&](auto& rK_buf, auto& rV_buf, auto& rCos_buf, auto& rSin_buf, bool is_last_tile) {
        if constexpr (HAS_ROPE) {
            // Cả 8 làn cùng gọi __shfl_xor_sync(..., 4) bên trong helper để tránh phân kỳ warp
            apply_rope_fwd_8elem_shfl(
                &rK_buf(0, 0, 0),
                reinterpret_cast<const __half2*>(&rCos_buf(0, 0, 0)),
                reinterpret_cast<const __half2*>(&rSin_buf(0, 0, 0)),
                is_second_half);
        }

        float partial_dot = 0.0f;
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
            partial_dot = fmaf(rQ_f32[v], half_to_float(rK_buf(v)), partial_dot);
        }
        const float s_t = octet_reduce_sum_f32(partial_dot);

        if (!is_last_tile || tPredLast(0, 0)) {
            const float m_new = fmaxf(m_local, s_t);
            const float alpha = (l_local == 0.0f) ? 0.0f : expf(m_local - m_new);
            const float p_t = expf(s_t - m_new);
            l_local = fmaf(l_local, alpha, p_t);
            CUTE_UNROLL
            for (int v = 0; v < 8; ++v) {
                rO_acc(v) = fmaf(rO_acc(v), alpha, p_t * half_to_float(rV_buf(v)));
            }
            m_local = m_new;
        }
    };

    // Phát trước 2 stage nạp GMEM đầu tiên trước khi bắt đầu tính toán (Rule 24A)
    load_tile(0, rK_s0, rV_s0, rCos_s0, rSin_s0);
    if (num_tiles > 1) {
        load_tile(1, rK_s1, rV_s1, rCos_s1, rSin_s1);
    }

    for (int tile_idx = 0; tile_idx + 1 < num_tiles; tile_idx += 2) {
        consume_tile(rK_s0, rV_s0, rCos_s0, rSin_s0, false);
        if (tile_idx + 2 < num_tiles) {
            load_tile(tile_idx + 2, rK_s0, rV_s0, rCos_s0, rSin_s0);
        }

        consume_tile(rK_s1, rV_s1, rCos_s1, rSin_s1, (tile_idx + 1 == num_tiles - 1));
        if (tile_idx + 3 < num_tiles) {
            load_tile(tile_idx + 3, rK_s1, rV_s1, rCos_s1, rSin_s1);
        }
    }

    if (num_tiles & 1) {
        consume_tile(rK_s0, rV_s0, rCos_s0, rSin_s0, true);
    }

    // ==================== GIAI ĐOẠN 3: EPILOGUE (GỘP 16 HÀNG & XUẤT GMEM) ====================
    cute::copy(tiled_copy_f32_16x8, rO_acc, tOsO_W);
    if (col_vec == 0) {
        sM_ptr[row_in_tile] = m_local;
        sL_ptr[row_in_tile] = l_local;
    }
    __syncthreads();

    // Đọc 2 hàng cách nhau 8 (row8 và row8 + 8) theo bố cục 8x16 LayoutLeft không xung đột bank
    cute::copy(tiled_copy_reduce, tOsO_R, rO_pair);

    const float m0 = sM_ptr[row8];
    const float l0 = sL_ptr[row8];
    const float m1 = sM_ptr[row8 + 8];
    const float l1 = sL_ptr[row8 + 8];

    const float m_global = octet_reduce_max_f32(fmaxf(m0, m1));
    const float w0 = (l0 > 0.0f) ? expf(m0 - m_global) : 0.0f;
    const float w1 = (l1 > 0.0f) ? expf(m1 - m_global) : 0.0f;

    const float l_global = octet_reduce_sum_f32(fmaf(l0, w0, l1 * w1));
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    CUTE_UNROLL
    for (int v = 0; v < 4; ++v) {
        rO_out(v, 0, 0) = octet_reduce_sum_f32(
            fmaf(rO_pair(v, 0, 0), w0, rO_pair(v, 1, 0) * w1)) * inv_l;
    }
    rLSE(0) = (l_global > 0.0f) ? (m_global + logf(l_global)) : DECODE_NEG_INF;

    cute::copy_if(tiled_copy_reduce, tPredStoreO, rO_out, tOgO_part);
    cute::copy_if(tiled_copy_lse, tPredLeader, rLSE, tLgLSE);
}

/*
 * Tóm tắt luồng thực thi kernel_fused_mha_varlen_decode_reduce (Không SMEM, không __syncthreads):
 *   Grid = (H_q, B, 1), Block = (128, 1, 1) (8x16 LayoutLeft: mỗi octet giảm trục splits cho 4 phần tử head).
 *   - Bước 1: Quét gLSE_part tìm lse_max chung qua octet_reduce_max_f32.
 *   - Bước 2: Dùng 2-stage Register Prefetch (Rule 24A) nạp gO_part theo từng tile SPLIT_TILE = 8,
 *     tích lũy Z_local và o_acc[4] với trọng số exp(lse_s - lse_max) ở FP32.
 *   - Bước 3: Giảm trên octet để lấy o_norm[4], ghép cặp 4+4 = 8 phần tử qua __shfl_xor_sync(..., 8),
 *     ép FP32 -> FP16 trên thanh ghi và ghi ra gO bằng 128-bit copy_if (8 lane dẫn đầu).
 */
template <
    int HEAD_DIM = 64,
    int NUM_THREADS = 128,
    int SPLIT_TILE = 8
>
__global__ void __launch_bounds__(NUM_THREADS)
kernel_fused_mha_varlen_decode_reduce(
    const float* __restrict__ O_part_ptr,   // (H_q, B, num_splits, 64) [FP32] - Đầu ra từng split
    const float* __restrict__ LSE_part_ptr, // (H_q, B, num_splits) [FP32] - LSE từng split
    cutlass::half_t* __restrict__ O_ptr,    // (H_q, B, 64) [FP16] - Đầu ra Attention cuối cùng
    int B,                                  // scalar [int] - Batch size
    int num_splits                          // scalar [int] - Tổng số phân đoạn Split-K
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int head_q_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);

    const int row8 = tid & 7; // 0..7: 8 làn cùng nhóm cột col16 (tid >> 3) nằm gọn trong 1 warp

    using Element = cutlass::half_t;
    using TiledCopyOpartF32 = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        make_layout(Shape<Int<SPLIT_TILE>, _16>{}, LayoutLeft{}),
        make_layout(Shape<_1, _4>{}, LayoutRight{})));
    using TiledCopyStoreO_FP16 = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        make_layout(Shape<_16, _8>{}, LayoutLeft{}),
        make_layout(Shape<_1, _8>{}, LayoutRight{})));

    TiledCopyOpartF32 tiled_copy_opart;
    auto thr_copy_opart = tiled_copy_opart.get_thread_slice(tid);

    TiledCopyStoreO_FP16 tiled_copy_store_fp16;
    auto thr_copy_store_fp16 = tiled_copy_store_fp16.get_thread_slice(tid);

    // Khai báo tensor đầu ra & fragment prefetch cố định một lần (Rule 23)
    const int64_t head_seq_idx = static_cast<int64_t>(head_q_idx) * B + seq_idx;
    const float* lse_part_base = LSE_part_ptr + head_seq_idx * num_splits;
    const float* o_part_base = O_part_ptr + head_seq_idx * num_splits * HEAD_DIM;

    Tensor gO_bcast = make_tensor(
        make_gmem_ptr(O_ptr + head_seq_idx * HEAD_DIM),
        make_shape(_16{}, Int<HEAD_DIM>{}),
        make_stride(_0{}, _1{}));
    Tensor tOgO = thr_copy_store_fp16.partition_D(gO_bcast);
    Tensor rO_fp16 = make_fragment_like(tOgO);
    Tensor tPredStoreFP16 = make_tensor<bool>(make_shape(size<1>(tOgO), size<2>(tOgO)));
    tPredStoreFP16(0, 0) = ((tid & 15) == 0);

    Tensor gO_part_dummy = make_tensor(
        make_gmem_ptr(o_part_base),
        make_shape(Int<SPLIT_TILE>{}, Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, _1{}));
    Tensor tOgO_part_dummy = thr_copy_opart.partition_S(gO_part_dummy);
    Tensor rO_s0 = make_fragment_like(tOgO_part_dummy);
    Tensor rO_s1 = make_fragment_like(tOgO_part_dummy);
    CUTE_STATIC_ASSERT_V(size(rO_s0) == Int<4>{});
    CUTE_STATIC_ASSERT_V(size(rO_fp16) == Int<8>{});

    const int num_split_tiles = (num_splits + SPLIT_TILE - 1) / SPLIT_TILE;
    const int last_valid_splits = num_splits - (num_split_tiles - 1) * SPLIT_TILE;
    Tensor tPredLastSplit = make_tensor<bool>(
        make_shape(size<1>(tOgO_part_dummy), size<2>(tOgO_part_dummy)));
    tPredLastSplit(0, 0) = (row8 < last_valid_splits);

    // 1. Tìm lse_max chung cho mọi splits trong (head_q_idx, seq_idx)
    float lse_max_local = DECODE_NEG_INF;
    for (int s = row8; s < num_splits; s += SPLIT_TILE) {
        lse_max_local = fmaxf(lse_max_local, lse_part_base[s]);
    }
    const float lse_max = octet_reduce_max_f32(lse_max_local);

    // 2. Tích lũy chuẩn hóa qua 2-stage Register Prefetch (Rule 24A)
    float Z_local = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    auto load_split_tile = [&](int tile_idx, auto& rO_buf) {
        const int split_base = tile_idx * SPLIT_TILE;
        Tensor gO_part_tile = make_tensor(
            make_gmem_ptr(o_part_base + static_cast<int64_t>(split_base) * HEAD_DIM),
            make_shape(Int<SPLIT_TILE>{}, Int<HEAD_DIM>{}),
            make_stride(Int<HEAD_DIM>{}, _1{}));
        Tensor tOgO_part = thr_copy_opart.partition_S(gO_part_tile);

        if (tile_idx < num_split_tiles - 1) {
            cute::copy(tiled_copy_opart, tOgO_part, rO_buf);
        } else {
            clear(rO_buf);
            cute::copy_if(tiled_copy_opart, tPredLastSplit, tOgO_part, rO_buf);
        }
    };

    auto consume_split_tile = [&](int tile_idx, const auto& rO_buf) {
        const int split_pos = tile_idx * SPLIT_TILE + row8;
        const bool valid_split = (split_pos < num_splits);
        const float lse_s = valid_split ? lse_part_base[split_pos] : DECODE_NEG_INF;
        const float w_s = (valid_split && lse_max > DECODE_VALID_THRESH && lse_s > DECODE_VALID_THRESH)
            ? expf(lse_s - lse_max)
            : 0.0f;

        Z_local += w_s;
        CUTE_UNROLL
        for (int v = 0; v < 4; ++v) {
            o_acc[v] = fmaf(rO_buf(v), w_s, o_acc[v]);
        }
    };

    load_split_tile(0, rO_s0);
    if (num_split_tiles > 1) {
        load_split_tile(1, rO_s1);
    }

    for (int tile_idx = 0; tile_idx + 1 < num_split_tiles; tile_idx += 2) {
        consume_split_tile(tile_idx, rO_s0);
        if (tile_idx + 2 < num_split_tiles) {
            load_split_tile(tile_idx + 2, rO_s0);
        }

        consume_split_tile(tile_idx + 1, rO_s1);
        if (tile_idx + 3 < num_split_tiles) {
            load_split_tile(tile_idx + 3, rO_s1);
        }
    }

    if (num_split_tiles & 1) {
        consume_split_tile(num_split_tiles - 1, rO_s0);
    }

    // 3. Chuẩn hóa theo tổng Z, ghép 4+4 = 8 phần tử qua warp shuffle và xuất uint4 ra GMEM
    const float Z_global = octet_reduce_sum_f32(Z_local);
    const float inv_Z = (Z_global > 0.0f) ? (1.0f / Z_global) : 0.0f;

    float o_norm[4];
    CUTE_UNROLL
    for (int v = 0; v < 4; ++v) {
        o_norm[v] = octet_reduce_sum_f32(o_acc[v]) * inv_Z;
    }

    // Ghép lane col16 chẵn [0..3] với lane col16+1 lẻ [4..7] qua tid ^ 8 rồi ép FP16 trên thanh ghi
    CUTE_UNROLL
    for (int v = 0; v < 4; ++v) {
        const float partner_v = __shfl_xor_sync(0xffffffff, o_norm[v], 8);
        rO_fp16(v) = float_to_out<Element>(o_norm[v]);
        rO_fp16(v + 4) = float_to_out<Element>(partner_v);
    }

    cute::copy_if(tiled_copy_store_fp16, tPredStoreFP16, rO_fp16, tOgO);
}

// =================================================================================================
// PHẦN 2: HOST FUNCTION
// =================================================================================================

/*
 * Tóm tắt hàm Host fused_mha_varlen_decode:
 *   - Bước 1 (Kiểm tra đầu vào): Xác thực CUDA device, contiguous, dtype (FP16 cho q/k_cache/v_cache/cos/sin,
 *     Int32 cho cu_seqlens_k/offsets), số chiều, head_dim == 64 và tỷ lệ GQA H_q % H_kv == 0.
 *   - Bước 2 (Tính num_splits động & Launch): Đo số SM và occupancy thực tế qua CUDA API để tính num_splits,
 *     cấp phát bộ đệm trung gian (o_part, lse_part) và output o, chạy kernel_fused_mha_varlen_decode_splitk
 *     rồi kernel_fused_mha_varlen_decode_reduce trong các scope cô lập, sau đó giải phóng bộ đệm trung gian.
 *   - Bước 3 (Hoàn tất): Trả về tensor:
 *       + o: (H_q, B, 64) [FP16]
 */
at::Tensor fused_mha_varlen_decode(
    const at::Tensor& q,                              // (H_q, B, 64) [FP16] - Query decode hiện tại
    const at::Tensor& k_cache,                        // (H_kv, Total_Lk, 64) [FP16] - Raw Key cache packed
    const at::Tensor& v_cache,                        // (H_kv, Total_Lk, 64) [FP16] - Value cache packed
    const at::Tensor& cu_seqlens_k,                   // (B + 1,) [Int32] - Độ dài tích lũy K, V
    int64_t max_seqlen_k,                             // scalar [int64] - Độ dài K lớn nhất trong batch
    const std::optional<at::Tensor>& cos,             // (L_max, 64) [FP16] - Bảng cos RoPE (optional)
    const std::optional<at::Tensor>& sin,             // (L_max, 64) [FP16] - Bảng sin RoPE (optional)
    const std::optional<at::Tensor>& offsets          // (B,) [Int32] - Offset RoPE từng chuỗi (optional)
) {
    // 1. Torch Checks
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be a CUDA tensor");

    TORCH_CHECK(k_cache.device() == q.device() && v_cache.device() == q.device() &&
                cu_seqlens_k.device() == q.device(),
                "All input tensors must reside on the same CUDA device");

    TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() &&
                v_cache.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "All input tensors must be contiguous");

    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf, "k_cache must be float16");
    TORCH_CHECK(v_cache.scalar_type() == at::kHalf, "v_cache must be float16");
    TORCH_CHECK(cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_k must be int32");

    TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 3 && v_cache.dim() == 3,
                "q, k_cache, v_cache must be 3D tensors");
    TORCH_CHECK(q.size(2) == 64 && k_cache.size(2) == 64 && v_cache.size(2) == 64,
                "Head dimension must be 64");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
                "v_cache shape must match k_cache shape");

    TORCH_CHECK(cu_seqlens_k.dim() == 1 && cu_seqlens_k.size(0) >= 1,
                "cu_seqlens_k must be a 1D tensor of length >= 1");
    TORCH_CHECK(max_seqlen_k >= 0, "max_seqlen_k must be non-negative");

    const int64_t H_q = q.size(0);
    const int64_t B = q.size(1);
    const int64_t H_kv = k_cache.size(0);
    const int64_t Total_Lk = k_cache.size(1);

    TORCH_CHECK(cu_seqlens_k.size(0) == B + 1,
                "cu_seqlens_k length must equal B + 1");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0),
                "H_q must be a positive multiple of H_kv");

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
        if (B > 0 && Total_Lk > 0) {
            TORCH_CHECK(cos->size(0) > 0,
                        "cos/sin table must have L_max > 0 for non-empty KV cache");
        }
    }

    // 2. Cấu hình tài nguyên GPU động và khởi chạy 2 kernel
    const c10::cuda::CUDAGuard device_guard(q.device());

    if (B == 0 || max_seqlen_k == 0 || Total_Lk == 0) {
        return at::zeros_like(q);
    }

    constexpr int HEAD_DIM = 64;
    constexpr int NUM_THREADS = 128;
    constexpr int TILE_K_TOKENS = 16;
    constexpr int SPLIT_TILE = 8;
    constexpr int64_t SPLIT_MIN_LEN = 64;
    constexpr int smem_size_splitk =
        (TILE_K_TOKENS * HEAD_DIM + 2 * TILE_K_TOKENS) * static_cast<int>(sizeof(float));

    const int dev_id = q.get_device();
    int sm_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id));

    int max_blocks_per_sm = 0;
    if (has_rope) {
        auto kernel_splitk = kernel_fused_mha_varlen_decode_splitk<
            HEAD_DIM, NUM_THREADS, TILE_K_TOKENS, true>;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel_splitk, NUM_THREADS, smem_size_splitk));
    } else {
        auto kernel_splitk = kernel_fused_mha_varlen_decode_splitk<
            HEAD_DIM, NUM_THREADS, TILE_K_TOKENS, false>;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel_splitk, NUM_THREADS, smem_size_splitk));
    }

    const int64_t max_concurrent_blocks =
        std::max<int64_t>(1, static_cast<int64_t>(sm_count) * std::max(1, max_blocks_per_sm));
    const int64_t target_splits = std::max<int64_t>(1, max_concurrent_blocks / (B * H_q));
    const int64_t max_possible_splits =
        std::max<int64_t>(1, (max_seqlen_k + SPLIT_MIN_LEN - 1) / SPLIT_MIN_LEN);
    const int num_splits = static_cast<int>(
        std::clamp<int64_t>(target_splits, 1, max_possible_splits));

    at::Tensor o_part = at::empty({H_q, B, num_splits, HEAD_DIM}, q.options().dtype(at::kFloat));
    at::Tensor lse_part = at::empty({H_q, B, num_splits}, q.options().dtype(at::kFloat));
    at::Tensor o = at::empty_like(q);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    const auto* q_ptr = reinterpret_cast<const cutlass::half_t*>(q.data_ptr<at::Half>());
    const auto* k_ptr = reinterpret_cast<const cutlass::half_t*>(k_cache.data_ptr<at::Half>());
    const auto* v_ptr = reinterpret_cast<const cutlass::half_t*>(v_cache.data_ptr<at::Half>());
    const int32_t* cu_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
    const cutlass::half_t* cos_ptr = has_rope
        ? reinterpret_cast<const cutlass::half_t*>(cos->data_ptr<at::Half>())
        : nullptr;
    const cutlass::half_t* sin_ptr = has_rope
        ? reinterpret_cast<const cutlass::half_t*>(sin->data_ptr<at::Half>())
        : nullptr;
    const int32_t* offsets_ptr = offsets.has_value() ? offsets->data_ptr<int32_t>() : nullptr;
    float* o_part_ptr = o_part.data_ptr<float>();
    float* lse_part_ptr = lse_part.data_ptr<float>();
    auto* o_ptr = reinterpret_cast<cutlass::half_t*>(o.data_ptr<at::Half>());

    // Kernel 1: Flash Decoding Split-K
    {
        const dim3 grid_splitk(
            static_cast<unsigned int>(num_splits),
            static_cast<unsigned int>(H_q),
            static_cast<unsigned int>(B));
        const dim3 block_splitk(NUM_THREADS);

        if (has_rope) {
            auto kernel_splitk = kernel_fused_mha_varlen_decode_splitk<
                HEAD_DIM, NUM_THREADS, TILE_K_TOKENS, true>;
            kernel_splitk<<<grid_splitk, block_splitk, smem_size_splitk, stream>>>(
                q_ptr,
                k_ptr,
                v_ptr,
                cu_k_ptr,
                cos_ptr,
                sin_ptr,
                offsets_ptr,
                o_part_ptr,
                lse_part_ptr,
                static_cast<int>(H_q),
                static_cast<int>(H_kv),
                static_cast<int>(B),
                static_cast<int>(Total_Lk),
                num_splits,
                sm_scale);
        } else {
            auto kernel_splitk = kernel_fused_mha_varlen_decode_splitk<
                HEAD_DIM, NUM_THREADS, TILE_K_TOKENS, false>;
            kernel_splitk<<<grid_splitk, block_splitk, smem_size_splitk, stream>>>(
                q_ptr,
                k_ptr,
                v_ptr,
                cu_k_ptr,
                cos_ptr,
                sin_ptr,
                offsets_ptr,
                o_part_ptr,
                lse_part_ptr,
                static_cast<int>(H_q),
                static_cast<int>(H_kv),
                static_cast<int>(B),
                static_cast<int>(Total_Lk),
                num_splits,
                sm_scale);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel 2: Split-K Log-Sum-Exp Reduction
    {
        const dim3 grid_reduce(
            static_cast<unsigned int>(H_q),
            static_cast<unsigned int>(B),
            1U);
        const dim3 block_reduce(NUM_THREADS);

        auto kernel_reduce = kernel_fused_mha_varlen_decode_reduce<
            HEAD_DIM, NUM_THREADS, SPLIT_TILE>;
        kernel_reduce<<<grid_reduce, block_reduce, 0, stream>>>(
            o_part_ptr,
            lse_part_ptr,
            o_ptr,
            static_cast<int>(B),
            num_splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Giải phóng bộ đệm trung gian Split-K (Rule 5 - Bước 8)
    o_part = at::Tensor();
    lse_part = at::Tensor();

    // 3. Trả về kết quả Attention cuối cùng
    return o;
}
