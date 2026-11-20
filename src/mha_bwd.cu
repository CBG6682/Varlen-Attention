// Kernel CUDA/CuTe tính Backward FlashAttention + Fused RoPE biến thiên độ dài (Varlen) cho kiến trúc SM75 (Turing)

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
// PHẦN 1: DEVICE KERNEL (__global__) VÀ HELPERS
// =================================================================================================

/** Áp dụng RoPE Forward trực tiếp trên fragment thanh ghi (32x8 threads, 8 halfs/bước). */
template <int TILE_K, typename FragFP16>
__device__ __forceinline__ void apply_rope_fwd_32x8(
    FragFP16& rTile,                              // ((1,8), 2, 1) [FP16] - Fragment thanh ghi cần xoay
    const cutlass::half_t* __restrict__ cos_ptr,  // (L_max, 64) [FP16] - Bảng cos RoPE
    const cutlass::half_t* __restrict__ sin_ptr,  // (L_max, 64) [FP16] - Bảng sin RoPE
    int tile_start_pos,                           // scalar [int] - Vị trí bắt đầu tile theo chuỗi
    int offset,                                   // scalar [int] - Offset RoPE của chuỗi
    int valid_rows,                               // scalar [int] - Số hàng hợp lệ còn lại
    int row_m0_g2r,                               // scalar [int] - Hàng bước m=0
    int row_m1_g2r,                               // scalar [int] - Hàng bước m=1
    int col_vec_g2r,                              // scalar [int] - Chỉ số nhóm 8 cột (0..7)
    bool is_second_half_g2r                       // scalar [bool] - Thuộc nửa sau [32..63]
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

/** Áp dụng RoPE Backward tại chỗ trên fragment FP32 (16x16 threads, 4 floats/bước). */
template <int TILE_K, typename FragF32>
__device__ __forceinline__ void apply_rope_bwd_f32_16x16(
    FragF32& rF32,                                // ((1,4), 4, 1) [FP32] - Gradient trước RoPE (tại chỗ)
    const cutlass::half_t* __restrict__ cos_ptr,  // (L_max, 64) [FP16] - Bảng cos RoPE
    const cutlass::half_t* __restrict__ sin_ptr,  // (L_max, 64) [FP16] - Bảng sin RoPE
    int tile_start_pos,                           // scalar [int] - Vị trí bắt đầu tile theo chuỗi
    int offset,                                   // scalar [int] - Offset RoPE của chuỗi
    int valid_rows,                               // scalar [int] - Số hàng hợp lệ còn lại
    int row_in_step_f32,                          // scalar [int] - Hàng trong nhóm 16 (tid >> 4)
    int col_vec_f32,                              // scalar [int] - Chỉ số nhóm 4 cột (tid & 15)
    bool is_second_half_f32                       // scalar [bool] - Thuộc nửa sau [32..63]
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

        // Tất cả thread trong warp cùng thực hiện __shfl_xor_sync trước khi kiểm tra valid_row
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

/*
 * Tóm tắt luồng thực thi kernel_fused_mha_varlen_bwd:
 *   Grid = (cdiv(max_seqlen_k, 64), B, H_kv), Block = (256, 1, 1) (8 warps).
 *   Mỗi CTA phụ trách 1 tile K/V kích thước (64 x 64) tại (batch_idx, h_kv, block_k_idx):
 *   - Giai đoạn 1 (Khởi tạo): Đọc varlen metadata (cu_seqlens_q/k, offsets), kiểm tra biên sớm,
 *     khởi tạo SMEM views (57,600 bytes), Copy Atoms, MMA và toàn bộ register/predicate tensors.
 *   - Giai đoạn 2 (Prologue): Tải K, V từ GMEM bằng 128-bit copy; áp dụng RoPE Forward lên K một lần
 *     trên thanh ghi rồi lưu K_rot, V vào SMEM; reset tCrDK, tCrDV về 0.
 *   - Giai đoạn 3 (Main Loop theo nhóm GQA h_q và block_q_idx):
 *       3.1 Tải Q (áp RoPE Forward), dO, O từ GMEM; tính D = rowsum(dO * O) vào sD_ptr[64].
 *       3.2 GEMM 1: S = Q @ K^T -> P = exp(scale * S - LSE) (che Causal/padding) -> lưu sP.
 *       3.3 GEMM 2: dV += P^T @ dO (đọc sP_mn, sDO_mn bằng LDSM_T, cộng dồn FP32 vào tCrDV).
 *       3.4 GEMM 3: dP = dO @ V^T -> dS_unscaled = P * (dP - D) -> lưu sdS.
 *       3.5 GEMM 4: dK += dS^T @ Q (đọc sdS_mn, sQ_mn bằng LDSM_T, cộng dồn FP32 vào tCrDK).
 *       3.6 GEMM 5: dQ_local = dS @ K -> nhân sm_scale -> stage FP32 vào sStaging_fp32 ([32KB..48KB)),
 *           tải về rF32 bằng TiledCopyF32, áp dụng RoPE Backward trên thanh ghi -> atomicAdd vào dQ_fp32.
 *   - Giai đoạn 4 (Epilogue): Chuyển tCrDV -> FP16 qua sStaging -> xuất GMEM (128-bit);
 *     nhân sm_scale lên tCrDK, áp dụng RoPE Backward qua sStaging_fp32 (nếu HAS_ROPE) -> xuất GMEM (128-bit).
 */
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
    const cutlass::half_t* __restrict__ dO_ptr,   // (H_q, Total_Lq, 64) [FP16] - Gradient đầu ra dO
    const cutlass::half_t* __restrict__ Q_ptr,    // (H_q, Total_Lq, 64) [FP16] - Query gốc trước RoPE
    const cutlass::half_t* __restrict__ K_ptr,    // (H_kv, Total_Lk, 64) [FP16] - Key gốc trước RoPE
    const cutlass::half_t* __restrict__ V_ptr,    // (H_kv, Total_Lk, 64) [FP16] - Value gốc
    const cutlass::half_t* __restrict__ O_ptr,    // (H_q, Total_Lq, 64) [FP16] - Output từ Forward
    const float* __restrict__ LSE_ptr,            // (H_q, Total_Lq)     [FP32] - Log-Sum-Exp từ Forward
    float* __restrict__ dQ_fp32_ptr,              // (H_q, Total_Lq, 64) [FP32] - Bộ đệm tích lũy atomic dQ
    cutlass::half_t* __restrict__ dK_ptr,         // (H_kv, Total_Lk, 64) [FP16] - Gradient dK đầu ra
    cutlass::half_t* __restrict__ dV_ptr,         // (H_kv, Total_Lk, 64) [FP16] - Gradient dV đầu ra
    const int32_t* __restrict__ cu_seqlens_q_ptr, // (B + 1,) [Int32] - Độ dài tích lũy Q
    const int32_t* __restrict__ cu_seqlens_k_ptr, // (B + 1,) [Int32] - Độ dài tích lũy K, V
    const int32_t* __restrict__ offsets_ptr,      // (B,) [Int32] - Offset RoPE cho từng chuỗi (hoặc nullptr)
    const cutlass::half_t* __restrict__ cos_ptr,  // (L_max, 64) [FP16] - Bảng cos RoPE (hoặc nullptr)
    const cutlass::half_t* __restrict__ sin_ptr,  // (L_max, 64) [FP16] - Bảng sin RoPE (hoặc nullptr)
    int H_q,                                      // scalar [int] - Số lượng head Query
    int H_kv,                                     // scalar [int] - Số lượng head Key/Value
    int Total_Lq,                                 // scalar [int] - Tổng số token Query
    int Total_Lk,                                 // scalar [int] - Tổng số token Key/Value
    float sm_scale                                // scalar [float] - Hệ số scale 1 / sqrt(64)
) {
    // ==================== GIAI ĐOẠN 1: KHỞI TẠO TỌA ĐỘ, SMEM & TENSOR ====================
    const int tid = static_cast<int>(threadIdx.x);
    const int block_k_idx = static_cast<int>(blockIdx.x);
    const int batch_idx = static_cast<int>(blockIdx.y);
    const int h_kv = static_cast<int>(blockIdx.z);

    // Đọc metadata chuỗi biến thiên trực tiếp từ GMEM
    const int q_start_seq = cu_seqlens_q_ptr[batch_idx];
    const int q_end_seq = cu_seqlens_q_ptr[batch_idx + 1];
    const int lq_i = q_end_seq - q_start_seq;

    const int k_start_seq = cu_seqlens_k_ptr[batch_idx];
    const int k_end_seq = cu_seqlens_k_ptr[batch_idx + 1];
    const int lk_i = k_end_seq - k_start_seq;

    const int k_start_col = block_k_idx * TILE_N;
    if (k_start_col >= lk_i) {
        return; // Toàn bộ tile K nằm ngoài chiều dài thực tế của chuỗi
    }

    const int offset = (HAS_ROPE && offsets_ptr != nullptr) ? offsets_ptr[batch_idx] : 0;
    const int num_groups = H_q / H_kv;
    const int valid_k_rows = lk_i - k_start_col;

    // Bố trí Dynamic Shared Memory (57,600 bytes <= 64 KB)
    extern __shared__ uint8_t smem_raw[];
    cutlass::half_t* sQ_ptr = reinterpret_cast<cutlass::half_t*>(smem_raw);
    cutlass::half_t* sK_ptr = sQ_ptr + TILE_M * TILE_K;                // [8 KB .. 16 KB)
    cutlass::half_t* sV_ptr = sK_ptr + TILE_N * TILE_K;                // [16 KB .. 24 KB)
    cutlass::half_t* sDO_ptr = sV_ptr + TILE_N * TILE_K;               // [24 KB .. 32 KB)
    cutlass::half_t* sP_ptr = sDO_ptr + TILE_M * TILE_K;               // [32 KB .. 40 KB)
    cutlass::half_t* sStaging_ptr = sP_ptr + TILE_M * TILE_N;          // [40 KB .. 48 KB)
    cutlass::half_t* sdS_ptr = sStaging_ptr + TILE_M * TILE_N;         // [48 KB .. 56 KB)
    float* sStaging_fp32_ptr = reinterpret_cast<float*>(sP_ptr);       // Tái sử dụng [32 KB .. 48 KB) (16 KB)
    float* sD_ptr = reinterpret_cast<float*>(sdS_ptr + TILE_M * TILE_N); // [56 KB .. 56.25 KB)

    // Khai báo kiểu CuTe Layouts, Copy Atoms và TiledMMA
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

    // Views SMEM (K-major và MN-major không xung đột bank)
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

    // GMEM Tensors cố định của tile K, V, dK, dV cho CTA này
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

    // Khởi tạo Copy Thread Slices, Fragments và Predicates (một lần duy nhất)
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

    // Staging FP32 (16x16 threads x 4 floats = 1 uint4) cho RoPE Backward dQ và dK
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

    // Khởi tạo MMA, Accumulators và S2R Partitions
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCrC = thr_mma.partition_fragment_C(sP); // Tái sử dụng cho GEMM 1, 3, 5
    Tensor tCrDK = make_tensor_like(tCrC);
    Tensor tCrDV = make_tensor_like(tCrC);
    Tensor rC_fp16 = make_tensor_like<Element>(tCrC);

    Tensor tCrA = thr_mma.partition_fragment_A(sQ); // Tái sử dụng cho cả 5 GEMM
    Tensor tCrB = thr_mma.partition_fragment_B(sK); // Tái sử dụng cho cả 5 GEMM

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

    // ==================== GIAI ĐOẠN 2: PROLOGUE (TẢI K, V VÀ ROPE CHO K) ====================
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

    // ==================== GIAI ĐOẠN 3: VÒNG LẶP CHÍNH THEO NHÓM GQA VÀ Q BLOCK ====================
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

            // --- 3.1 Tải Q (áp RoPE), dO, O và tính D = rowsum(dO * O) ---
            clear(rQ);
            clear(rDO);
            clear(rV); // Tạm mượn rV giữ O (rV không dùng đến trước Epilogue)

            cute::copy_if(tiled_copy_g2r, tQpQ, tQgQ, rQ);
            cute::copy_if(tiled_copy_g2r, tQpQ, tDOgDO, rDO);
            cute::copy_if(tiled_copy_g2r, tQpQ, tOgO, rV);

            if constexpr (HAS_ROPE) {
                apply_rope_fwd_32x8<TILE_K>(
                    rQ, cos_ptr, sin_ptr, q_start_row, offset, valid_q_rows,
                    row_m0_g2r, row_m1_g2r, col_vec_g2r, is_second_half_g2r);
            }

            // Tính D_i = sum_d (dO[i, d] * O[i, d]) bằng giảm 8 làn trong warp
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

            // --- 3.2 GEMM 1: S = Q @ K^T -> P = exp(scale * S - LSE) -> lưu sP ---
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

            // --- 3.3 GEMM 2: dV += P^T @ dO (tích lũy FP32 vào tCrDV) ---
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_mn, tAsP_mn(_, _, k_step), tArA_mn(_, _, k_step));
                cute::copy(s2r_copy_B_mn, tBsDO_mn(_, _, k_step), tBrB_mn(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrDV);
            }

            // --- 3.4 GEMM 3: dP = dO @ V^T -> dS_unscaled = P * (dP - D) -> lưu sdS ---
            clear(tCrC);
            CUTE_UNROLL
            for (int k_step = 0; k_step < size<2>(tCrA); ++k_step) {
                cute::copy(s2r_copy_A_k, tAsDO(_, _, k_step), tArA_k(_, _, k_step));
                cute::copy(s2r_copy_B_k, tBsV(_, _, k_step), tBrB_k(_, _, k_step));
                cute::gemm(tiled_mma, tCrA(_, _, k_step), tCrB(_, _, k_step), tCrC);
            }

            // Tải P từ sP về rC_fp16 để tính dS trên thanh ghi (tránh nhân sm_scale sớm để bảo toàn dynamic range FP16)
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

            // --- 3.5 GEMM 4: dK += dS^T @ Q (tích lũy FP32 vào tCrDK) ---
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

            // Ghi dQ_local FP32 vào sStaging_fp32 (vùng [32KB..48KB) hoàn toàn tách rời sdS [48KB..56KB))
            cute::copy(tCrC, tCsStagingF32_C);
            __syncthreads();

            cute::copy(tiled_copy_f32, tStagingF32_S, rF32);

            if constexpr (HAS_ROPE) {
                apply_rope_bwd_f32_16x16<TILE_K>(
                    rF32, cos_ptr, sin_ptr, q_start_row, offset, valid_q_rows,
                    row_in_step_f32, col_vec_f32, is_second_half_f32);
            }

            // Cộng dồn an toàn giữa các block_k_idx vào bộ đệm FP32 GMEM
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
            __syncthreads(); // Bảo vệ toàn bộ SMEM trước lượt Q kế tiếp hoặc Epilogue
        }
    }

    // ==================== GIAI ĐOẠN 4: EPILOGUE (XUẤT dV VÀ dK) ====================
    // 4.1 Xuất dV: FP32 thanh ghi -> FP16 thanh ghi -> sStaging -> GMEM 128-bit coalesced
    CUTE_UNROLL
    for (int i = 0; i < size(tCrDV); ++i) {
        rC_fp16(i) = float_to_out<Element>(tCrDV(i));
    }
    cute::copy(rC_fp16, tCsStaging_C);
    __syncthreads();

    cute::copy(tiled_copy_g2r, tStaging_S, rV);
    cute::copy_if(tiled_copy_g2r, tKpK, rV, tDVgDV);
    __syncthreads(); // Chờ đọc xong sStaging trước khi dK dùng sStaging_fp32 hoặc sStaging

    // 4.2 Xuất dK: Nhân sm_scale -> RoPE Backward (nếu HAS_ROPE) -> sStaging -> GMEM 128-bit coalesced
    CUTE_UNROLL
    for (int i = 0; i < size(tCrDK); ++i) {
        tCrDK(i) *= sm_scale;
    }

    if constexpr (HAS_ROPE) {
        cute::copy(tCrDK, tCsStagingF32_C);
        __syncthreads();

        cute::copy(tiled_copy_f32, tStagingF32_S, rF32);
        __syncthreads(); // Bắt buộc đợi toàn bộ đọc xong sStaging_fp32 trước khi ghi sStaging ([40KB..48KB))

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
// PHẦN 2: HOST FUNCTION
// =================================================================================================

/*
 * Tóm tắt hàm Host fused_mha_varlen_bwd:
 *   - Bước 1 (Kiểm tra đầu vào): Xác thực CUDA device, contiguous, dtype (FP16 cho do_grad/q/k/v/o/cos/sin,
 *     FP32 cho lse, Int32 cho cu_seqlens_q/k/offsets), số chiều và tỷ lệ GQA H_q % H_kv == 0.
 *   - Bước 2 (Cấu hình & Launch): Khởi tạo CUDAGuard, cấp phát bộ đệm dq_fp32 (FP32) và dk, dv (FP16),
 *     thiết lập Dynamic SMEM (57,600 bytes) bằng cudaFuncSetAttribute và khởi chạy kernel theo <HAS_ROPE, IS_CAUSAL>.
 *   - Bước 3 (Hoàn tất đầu ra): Ép kiểu dq = dq_fp32.to(FP16), giải phóng bộ đệm dq_fp32 và trả về tuple:
 *       + dq: (H_q, Total_Lq, 64)  [FP16]
 *       + dk: (H_kv, Total_Lk, 64) [FP16]
 *       + dv: (H_kv, Total_Lk, 64) [FP16]
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_mha_varlen_bwd(
    const at::Tensor& do_grad,                        // (H_q, Total_Lq, 64) [FP16] - Gradient đầu ra dO
    const at::Tensor& q,                              // (H_q, Total_Lq, 64) [FP16] - Query gốc trước RoPE
    const at::Tensor& k,                              // (H_kv, Total_Lk, 64) [FP16] - Key gốc trước RoPE
    const at::Tensor& v,                              // (H_kv, Total_Lk, 64) [FP16] - Value gốc
    const at::Tensor& o,                              // (H_q, Total_Lq, 64) [FP16] - Output từ Forward
    const at::Tensor& lse,                            // (H_q, Total_Lq)     [FP32] - Log-Sum-Exp từ Forward
    const at::Tensor& cu_seqlens_q,                   // (B + 1,) [Int32] - Độ dài tích lũy của Q
    const at::Tensor& cu_seqlens_k,                   // (B + 1,) [Int32] - Độ dài tích lũy của K, V
    const std::optional<at::Tensor>& offsets,         // (B,) [Int32] - Offset RoPE cho từng chuỗi (optional)
    int64_t max_seqlen_k,                             // scalar [int64] - Độ dài K lớn nhất trong batch
    const std::optional<at::Tensor>& cos,             // (L_max, 64) [FP16] - Bảng cos RoPE (optional)
    const std::optional<at::Tensor>& sin,             // (L_max, 64) [FP16] - Bảng sin RoPE (optional)
    bool is_causal                                    // scalar [bool] - Cờ Causal Mask
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

    // 2. Cấp phát bộ đệm và khởi chạy kernel
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

    // 3. Ép kiểu dQ về FP16, giải phóng bộ đệm FP32 và trả kết quả
    at::Tensor dq = dq_fp32.to(at::kHalf);
    dq_fp32 = at::Tensor();

    return std::make_tuple(dq, dk, dv);
}
