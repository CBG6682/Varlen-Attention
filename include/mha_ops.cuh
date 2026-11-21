#pragma once

#include <torch/types.h>
#include <optional>
#include <tuple>
#include <cstdint>

// Varlen FlashAttention + Fused RoPE operators (Turing SM75 / CuTe)
// Packed head-first layout (H, Total_L, 64) for 128-bit coalesced access.
// Supports GQA (H_q % H_kv == 0), cross-attention (Lq != Lk), causal masking, and per-sequence RoPE offsets.

// 1. Forward: Varlen FlashAttention + Fused RoPE
// Outputs: (o: (H_q, Total_Lq, 64) FP16, lse: (H_q, Total_Lq) FP32)
std::tuple<at::Tensor, at::Tensor> fused_mha_varlen_fwd(
    const at::Tensor& q,                                     // (H_q, Total_Lq, 64) [FP16]
    const at::Tensor& k,                                     // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& v,                                     // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false,                                  // causal mask flag
    const std::optional<at::Tensor>& offsets = std::nullopt  // (B,) [Int32] per-sequence RoPE offset
);

// 2. Backward: Varlen FlashAttention Backward + Fused RoPE Backward
// Outputs: (dq: (H_q, Total_Lq, 64) FP16, dk: (H_kv, Total_Lk, 64) FP16, dv: (H_kv, Total_Lk, 64) FP16)
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_mha_varlen_bwd(
    const at::Tensor& do_grad,                               // (H_q, Total_Lq, 64) [FP16]
    const at::Tensor& q,                                     // (H_q, Total_Lq, 64) [FP16]
    const at::Tensor& k,                                     // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& v,                                     // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& o,                                     // (H_q, Total_Lq, 64) [FP16]
    const at::Tensor& lse,                                   // (H_q, Total_Lq)     [FP32]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets,                // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false                                   // causal mask flag
);

// 3. Inference: Adaptive Varlen Flash Decoding (Lq = 1)
// Dispatches between Direct 1-Block FP16 and Split-K based on B * H_q vs GPU occupancy.
// Output: (H_q, B, 64) [FP16]
at::Tensor fused_mha_varlen_decode(
    const at::Tensor& q,                                     // (H_q, B, 64) [FP16]
    const at::Tensor& k_cache,                               // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& v_cache,                               // (H_kv, Total_Lk, 64) [FP16]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& offsets = std::nullopt, // (B,) [Int32]
    std::optional<bool> use_splitk = std::nullopt            // force Split-K on/off (None = auto)
);
