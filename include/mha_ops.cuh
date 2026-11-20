#pragma once

#include <torch/types.h>
#include <optional>
#include <tuple>
#include <cstdint>

// =========================================================================================
// Khai báo Operator cho Varlen FlashAttention + Fused RoPE (Turing SM75 / CuTe)
// =========================================================================================
// - Toàn bộ input/output ở dạng packed varlen (không có chiều batch tách biệt),
//   bố trí theo head-first (H, Total_L, 64) để tối ưu truy xuất 128-bit coalesced.
// - Hỗ trợ Grouped Query Attention (GQA: H_q % H_kv == 0), Cross-Attention (Lq != Lk),
//   Causal Mask, và RoPE độc lập theo từng chuỗi qua mảng offsets (B,).
// =========================================================================================

// -----------------------------------------------------------------------------------------
// 1. Nhánh Forward: FlashAttention Varlen + Fused RoPE
// -----------------------------------------------------------------------------------------
// Flow thực thi:
//   - Xác thực tensor đầu vào (CUDA, contiguous, FP16/Int32, head_dim = 64, H_q % H_kv == 0).
//   - Nạp Q lên thanh ghi, áp dụng RoPE thuận R(theta) trên thanh ghi và giữ cố định suốt kernel.
//   - Pipeline 2 lần đọc / 1 stage qua các block K, V (16 KB SMEM chia 2 buffer 8 KB có Swizzle):
//     S = Q @ K^T -> Online Softmax + chuyển tiếp P trực tiếp từ tCrS sang tCrA (Rule 19) -> O += P @ V.
//   - Chuẩn hóa O /= l_i, staging qua SMEM và xuất O (FP16) cùng LSE (FP32) ra GMEM.
// Outputs:
//   1. o:   (H_q, Total_Lq, 64) [FP16] - Kết quả Attention raw dạng packed
//   2. lse: (H_q, Total_Lq)     [FP32] - Log-Sum-Exp dùng cho Backward
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> fused_mha_varlen_fwd(
    const at::Tensor& q,                              // (H_q, Total_Lq, 64) [FP16] - Query packed
    const at::Tensor& k,                              // (H_kv, Total_Lk, 64) [FP16] - Key packed
    const at::Tensor& v,                              // (H_kv, Total_Lk, 64) [FP16] - Value packed
    const at::Tensor& cu_seqlens_q,                   // (B + 1,) [Int32] - Độ dài tích lũy của Q
    const at::Tensor& cu_seqlens_k,                   // (B + 1,) [Int32] - Độ dài tích lũy của K, V
    int64_t max_seqlen_q,                             // scalar [int64] - Độ dài Q lớn nhất trong batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16] - Bảng cos RoPE (optional)
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16] - Bảng sin RoPE (optional)
    bool is_causal = false,                                  // scalar [bool] - Cờ bật/tắt Causal Mask
    const std::optional<at::Tensor>& offsets = std::nullopt  // (B,) [Int32] - Offset RoPE cho từng chuỗi (optional)
);

// -----------------------------------------------------------------------------------------
// 2. Nhánh Backward: FlashAttention Varlen Backward + Fused RoPE Backward
// -----------------------------------------------------------------------------------------
// Flow thực thi:
//   - Cấu hình 256 threads (1 block/SM) và 64 KB dynamic SMEM (Swizzle) cho SM75.
//   - Mỗi block xử lý 1 block K, V (64 tokens) và lặp qua toàn bộ rep = H_q / H_kv heads (GQA Group Loop).
//   - Tính trực tiếp D = rowsum(dO * O) bằng warp shuffle ngay khi nạp dO, O.
//   - Dùng chung 1 cặp thanh ghi toán hạng (tCrA, tCrB) cho 5 GEMM (S -> P, dV, dP -> dS, dK, dQ).
//   - Áp dụng RoPE Backward R(-theta) trên thanh ghi cho dQ và dK; staging dQ ở FP32 (16 KB) rồi atomicAdd.
// Outputs:
//   1. dq: (H_q, Total_Lq, 64)  [FP16] - Gradient theo Query (đã qua RoPE backward)
//   2. dk: (H_kv, Total_Lk, 64) [FP16] - Gradient theo Key (đã qua RoPE backward & cộng gộp GQA)
//   3. dv: (H_kv, Total_Lk, 64) [FP16] - Gradient theo Value (đã cộng gộp GQA)
// -----------------------------------------------------------------------------------------
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
    const std::optional<at::Tensor>& cos = std::nullopt, // (L_max, 64) [FP16] - Bảng cos RoPE (optional)
    const std::optional<at::Tensor>& sin = std::nullopt, // (L_max, 64) [FP16] - Bảng sin RoPE (optional)
    bool is_causal = false                            // scalar [bool] - Cờ Causal Mask
);

// -----------------------------------------------------------------------------------------
// 3. Nhánh Inference: Flash Decoding Varlen (Không sử dụng MMA)
// -----------------------------------------------------------------------------------------
// Flow thực thi:
//   - Tính num_splits động dựa trên SM count và occupancy thực tế của GPU.
//   - Kernel 1 (splitk): Nạp broadcast q, xoay RoPE tại vị trí (offset + lk_i - 1), nạp K, V
//     bằng Register Prefetch (Rule 24A), xoay RoPE cho từng token K (vị trí offset + t),
//     tính dot-product & Online Softmax ở FP32, ghi o_part và lse_part.
//   - Kernel 2 (reduce): Gộp Log-Sum-Exp chính xác các splits ở FP32 và xuất kết quả o (FP16).
// Outputs:
//   1. o: (H_q, B, 64) [FP16] - Kết quả Attention đầu ra cho bước decode Lq = 1
// -----------------------------------------------------------------------------------------
at::Tensor fused_mha_varlen_decode(
    const at::Tensor& q,                              // (H_q, B, 64) [FP16] - Query decode hiện tại
    const at::Tensor& k_cache,                        // (H_kv, Total_Lk, 64) [FP16] - Raw Key cache packed
    const at::Tensor& v_cache,                        // (H_kv, Total_Lk, 64) [FP16] - Value cache packed
    const at::Tensor& cu_seqlens_k,                   // (B + 1,) [Int32] - Độ dài tích lũy của K, V
    int64_t max_seqlen_k,                             // scalar [int64] - Độ dài K lớn nhất trong batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16] - Bảng cos RoPE (optional)
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16] - Bảng sin RoPE (optional)
    const std::optional<at::Tensor>& offsets = std::nullopt  // (B,) [Int32] - Offset RoPE cho từng chuỗi (optional)
);
