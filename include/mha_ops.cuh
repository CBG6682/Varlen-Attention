#pragma once

#include <torch/types.h>
#include <optional>
#include <tuple>
#include <cstdint>

// Varlen FlashAttention + Fused RoPE operators (Turing SM75 / CuTe)
// Packed THD layout (Total_L, H, 64) with 128-bit coalesced access (strided views supported).
// Supports GQA (H_q % H_kv == 0), cross-attention (Lq != Lk), causal masking, and per-sequence RoPE offsets.

// 1. Forward: Varlen FlashAttention + Fused RoPE
// Launcher writing directly into preallocated o (and optional lse)
void launch_fused_mha_varlen_fwd_into(
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& k,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    at::Tensor& o,                                           // (Total_Lq, H_q, 64) [FP16]
    const std::optional<at::Tensor>& lse_opt = std::nullopt, // (H_q, Total_Lq) [FP32]
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false,                                  // causal mask flag
    const std::optional<at::Tensor>& offsets = std::nullopt  // (B,) [Int32] per-sequence RoPE offset
);

// Outputs: (o: (Total_Lq, H_q, 64) FP16, lse: (H_q, Total_Lq) FP32)
std::tuple<at::Tensor, at::Tensor> fused_mha_varlen_fwd(
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& k,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false,                                  // causal mask flag
    const std::optional<at::Tensor>& offsets = std::nullopt  // (B,) [Int32] per-sequence RoPE offset
);

// 2. Backward: Varlen FlashAttention Backward + Fused RoPE Backward
// Launcher writing directly into preallocated dq_out, dk_out, dv_out
void launch_fused_mha_varlen_bwd_into(
    const at::Tensor& do_grad,                               // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& k,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& o,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& lse,                                   // (H_q, Total_Lq) [FP32]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets,                // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    at::Tensor& dq_out,                                      // (Total_Lq, H_q, 64) [FP16]
    at::Tensor& dk_out,                                      // (Total_Lk, H_kv, 64) [FP16]
    at::Tensor& dv_out,                                      // (Total_Lk, H_kv, 64) [FP16]
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false                                   // causal mask flag
);

// Outputs: (dq: (Total_Lq, H_q, 64) FP16, dk: (Total_Lk, H_kv, 64) FP16, dv: (Total_Lk, H_kv, 64) FP16)
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_mha_varlen_bwd(
    const at::Tensor& do_grad,                               // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& k,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,                                     // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& o,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& lse,                                   // (H_q, Total_Lq) [FP32]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets,                // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false                                   // causal mask flag
);

// 3. Paged KV Cache Management (PAGE_SIZE = 16, HEAD_DIM = 64)
void launch_prefill_to_paged_kv(
    const at::Tensor& k,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_k,  // (B + 1,) [Int32]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,       // (B,) [Int32]
    int64_t max_seqlen_k             // max K sequence length in batch
);

void prefill_to_paged_kv(
    const at::Tensor& k,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_k,  // (B + 1,) [Int32]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,       // (B,) [Int32]
    int64_t max_seqlen_k             // max K sequence length in batch
);

void launch_paged_kv_append(
    const at::Tensor& new_k,         // (B, H_kv, 64) [FP16]
    const at::Tensor& new_v,         // (B, H_kv, 64) [FP16]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens        // (B,) [Int32]
);

void paged_kv_append(
    const at::Tensor& new_k,         // (B, H_kv, 64) [FP16]
    const at::Tensor& new_v,         // (B, H_kv, 64) [FP16]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens        // (B,) [Int32]
);

// 4. Inference: Adaptive Paged Varlen Flash Decoding (Lq = 1)
// Dispatches between Direct 1-Block FP16 and Split-K based on B * H_q vs GPU occupancy.
void launch_fused_mha_varlen_decode_into(
    const at::Tensor& q,                                     // (B, H_q, 64) [FP16]
    const at::Tensor& k_cache,                               // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& v_cache,                               // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,                           // (B, max_pages_per_seq) [Int32]
    const at::Tensor& cache_seqlens,                         // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    at::Tensor& o,                                           // (B, H_q, 64) [FP16]
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& offsets = std::nullopt, // (B,) [Int32]
    std::optional<bool> use_splitk = std::nullopt            // force Split-K on/off (None = auto)
);

// Output: o (B, H_q, 64) [FP16]
at::Tensor fused_mha_varlen_decode(
    const at::Tensor& q,                                     // (B, H_q, 64) [FP16]
    const at::Tensor& k_cache,                               // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& v_cache,                               // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,                           // (B, max_pages_per_seq) [Int32]
    const at::Tensor& cache_seqlens,                         // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& offsets = std::nullopt, // (B,) [Int32]
    std::optional<bool> use_splitk = std::nullopt            // force Split-K on/off (None = auto)
);

// 5. Fused GQA Linear + Varlen Attention Blocks (bias-free, cuBLAS via at::mm)
// Self-Attention Forward (Training / Non-cached Inference)
// Outputs: (out, qkv, attn_out, lse)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_gqa_varlen_fwd(
    const at::Tensor& x,                                     // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& w_qkv,                                 // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                                 // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    int64_t H_q,                                             // number of Q heads
    int64_t H_kv,                                            // number of K/V heads
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false,                                  // causal mask flag
    const std::optional<at::Tensor>& offsets = std::nullopt, // (B,) [Int32]
    bool save_for_bwd = true                                 // retain qkv/attn_out/lse for backward
);

// Self-Attention Prefill with Paged KV Cache
// Output: out (Total_Lq, embed_dim) [FP16]
at::Tensor fused_gqa_varlen_prefill(
    const at::Tensor& x,                                     // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& w_qkv,                                 // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                                 // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    int64_t H_q,                                             // number of Q heads
    int64_t H_kv,                                            // number of K/V heads
    at::Tensor& k_cache,                                     // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,                                     // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,                           // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,                               // (B,) [Int32]
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false,                                  // causal mask flag
    const std::optional<at::Tensor>& offsets = std::nullopt  // (B,) [Int32]
);

// Self-Attention Backward
// Outputs: (dx, dw_qkv, dw_out)
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_gqa_varlen_bwd(
    const at::Tensor& grad_out,                              // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& x,                                     // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& qkv,                                   // (Total_Lq, H_q + 2*H_kv, 64) [FP16]
    const at::Tensor& attn_out,                              // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& lse,                                   // (H_q, Total_Lq) [FP32]
    const at::Tensor& w_qkv,                                 // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                                 // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets,                // (B,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    int64_t H_q,                                             // number of Q heads
    int64_t H_kv,                                            // number of K/V heads
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool is_causal = false,                                  // causal mask flag
    bool needs_dx = true,                                    // compute dx
    bool needs_dw_qkv = true,                                // compute dw_qkv
    bool needs_dw_out = true                                 // compute dw_out
);

// Cross-Attention Forward
// Outputs: (out, q, kv, attn_out, lse)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_cross_gqa_varlen_fwd(
    const at::Tensor& x,                                     // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& context,                               // (Total_Lk, context_dim) [FP16]
    const at::Tensor& w_q,                                   // (H_q*64, embed_dim) [FP16]
    const at::Tensor& w_kv,                                  // (2*H_kv*64, context_dim) [FP16]
    const at::Tensor& w_out,                                 // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                    // max Q sequence length in batch
    int64_t H_q,                                             // number of Q heads
    int64_t H_kv,                                            // number of K/V heads
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& offsets = std::nullopt, // (B,) [Int32]
    bool save_for_bwd = true                                 // retain q/kv/attn_out/lse for backward
);

// Cross-Attention Backward
// Outputs: (dx, d_context, dw_q, dw_kv, dw_out)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_cross_gqa_varlen_bwd(
    const at::Tensor& grad_out,                              // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& x,                                     // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& context,                               // (Total_Lk, context_dim) [FP16]
    const at::Tensor& q,                                     // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& kv,                                    // (Total_Lk, 2*H_kv, 64) [FP16]
    const at::Tensor& attn_out,                              // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& lse,                                   // (H_q, Total_Lq) [FP32]
    const at::Tensor& w_q,                                   // (H_q*64, embed_dim) [FP16]
    const at::Tensor& w_kv,                                  // (2*H_kv*64, context_dim) [FP16]
    const at::Tensor& w_out,                                 // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,                          // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,                          // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets,                // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    int64_t H_q,                                             // number of Q heads
    int64_t H_kv,                                            // number of K/V heads
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    bool needs_dx = true,                                    // compute dx
    bool needs_dctx = true,                                  // compute d_context
    bool needs_dw_q = true,                                  // compute dw_q
    bool needs_dw_kv = true,                                 // compute dw_kv
    bool needs_dw_out = true                                 // compute dw_out
);

// Self-Attention Decode Step (Lq = 1)
// Output: out (B, embed_dim) [FP16]
at::Tensor fused_gqa_varlen_decode(
    const at::Tensor& x,                                     // (B, embed_dim) [FP16]
    const at::Tensor& w_qkv,                                 // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                                 // (embed_dim, H_q*64) [FP16]
    at::Tensor& k_cache,                                     // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,                                     // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,                           // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,                               // (B,) [Int32]
    int64_t max_seqlen_k,                                    // max K sequence length in batch
    int64_t H_q,                                             // number of Q heads
    int64_t H_kv,                                            // number of K/V heads
    const std::optional<at::Tensor>& cos = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& sin = std::nullopt,     // (L_max, 64) [FP16]
    const std::optional<at::Tensor>& offsets = std::nullopt, // (B,) [Int32]
    std::optional<bool> use_splitk = std::nullopt            // force Split-K on/off (None = auto)
);
