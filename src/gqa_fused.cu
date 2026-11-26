// Fused GQA Linear (at::mm, bias-free) + Varlen Attention / Paged Decode C++ orchestrators

#include <cstdint>
#include <optional>
#include <tuple>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "mha_ops.cuh"

namespace {

// Shared implementation for Self-Attention Forward Training and Prefill-with-Cache
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> gqa_varlen_fwd_impl(
    const at::Tensor& x,                                      // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& w_qkv,                                  // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                                  // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,                           // (B + 1,) [Int32]
    int64_t max_seqlen_q,                                     // scalar: int64
    int64_t H_q,                                              // scalar: int64
    int64_t H_kv,                                             // scalar: int64
    const std::optional<at::Tensor>& cos,                     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,                     // (L_max, 64) [FP16] (optional)
    bool is_causal,                                           // scalar: bool
    const std::optional<at::Tensor>& offsets,                 // (B,) [Int32] (optional)
    bool save_for_bwd,                                        // scalar: bool
    std::optional<at::Tensor> k_cache_opt,                    // (num_pages, 16, H_kv, 64) [FP16] (optional)
    std::optional<at::Tensor> v_cache_opt,                    // (num_pages, 16, H_kv, 64) [FP16] (optional)
    const std::optional<at::Tensor>& block_table_opt,         // (B, max_pages_per_seq) [Int32] (optional)
    std::optional<at::Tensor> cache_seqlens_opt               // (B,) [Int32] (optional)
) {
    // 1. Input validation
    TORCH_CHECK(x.is_cuda() && w_qkv.is_cuda() && w_out.is_cuda() && cu_seqlens_q.is_cuda(),
                "x, w_qkv, w_out, and cu_seqlens_q must be CUDA tensors");
    TORCH_CHECK(
        w_qkv.device() == x.device() && w_out.device() == x.device() && cu_seqlens_q.device() == x.device(),
        "All input tensors must reside on the same CUDA device"
    );
    TORCH_CHECK(
        x.scalar_type() == at::kHalf && w_qkv.scalar_type() == at::kHalf && w_out.scalar_type() == at::kHalf,
        "x, w_qkv, and w_out must be float16"
    );
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_q.is_contiguous(),
                "cu_seqlens_q must be contiguous int32");
    TORCH_CHECK(x.dim() == 2 && w_qkv.dim() == 2 && w_out.dim() == 2 && cu_seqlens_q.dim() == 1,
                "Expected 2D x, w_qkv, w_out and 1D cu_seqlens_q");
    TORCH_CHECK(cu_seqlens_q.size(0) >= 1, "cu_seqlens_q must have length >= 1");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be a positive multiple of H_kv");
    TORCH_CHECK(max_seqlen_q >= 0, "max_seqlen_q must be non-negative");

    const int64_t Total_Lq  = x.size(0);
    const int64_t embed_dim = H_q * 64;
    const int64_t qkv_heads = H_q + 2 * H_kv;
    const int64_t qkv_dim   = qkv_heads * 64;

    TORCH_CHECK(x.size(1) == embed_dim, "x.size(1) must equal H_q * 64");
    TORCH_CHECK(w_qkv.size(0) == qkv_dim && w_qkv.size(1) == embed_dim,
                "w_qkv must have shape ((H_q + 2*H_kv)*64, H_q*64)");
    TORCH_CHECK(w_out.size(0) == embed_dim && w_out.size(1) == embed_dim,
                "w_out must have shape (H_q*64, H_q*64)");

    const c10::cuda::CUDAGuard device_guard(x.device());

    // 2. QKV projection (cuBLAS via at::mm) -> zero-copy strided slices
    at::Tensor qkv = at::mm(x, w_qkv.t()).view({Total_Lq, qkv_heads, 64});
    at::Tensor q   = qkv.narrow(1, 0, H_q);
    at::Tensor k   = qkv.narrow(1, H_q, H_kv);
    at::Tensor v   = qkv.narrow(1, H_q + H_kv, H_kv);

    // 3. Optional prefill write to Paged KV Cache
    if (k_cache_opt.has_value()) {
        TORCH_CHECK(
            v_cache_opt.has_value() && block_table_opt.has_value() && cache_seqlens_opt.has_value(),
            "k_cache, v_cache, block_table, and cache_seqlens must all be provided together"
        );
        launch_prefill_to_paged_kv(
            k, v, cu_seqlens_q,
            *k_cache_opt, *v_cache_opt, *block_table_opt, *cache_seqlens_opt,
            max_seqlen_q
        );
    }

    // 4. Varlen FlashAttention + Fused RoPE Forward
    at::Tensor attn_out = at::empty({Total_Lq, H_q, 64}, x.options());
    std::optional<at::Tensor> lse_opt = std::nullopt;
    if (save_for_bwd) {
        lse_opt = at::empty({H_q, Total_Lq}, x.options().dtype(at::kFloat));
    }

    launch_fused_mha_varlen_fwd_into(
        q, k, v, cu_seqlens_q, cu_seqlens_q, max_seqlen_q,
        attn_out, lse_opt, cos, sin, is_causal, offsets
    );

    // 5. Output projection (cuBLAS via at::mm)
    at::Tensor out = at::mm(attn_out.view({Total_Lq, embed_dim}), w_out.t());

    if (!save_for_bwd) {
        return std::make_tuple(out, at::Tensor(), at::Tensor(), at::Tensor());
    }
    return std::make_tuple(out, qkv, attn_out, *lse_opt);
}

} // namespace

// =================================================================================================
// Public Host Functions
// =================================================================================================

// 1. Self-Attention Forward (Training / Inference without KV Cache)
// Returns (out: (Total_Lq, embed_dim), qkv: (Total_Lq, H_q+2*H_kv, 64), attn_out: (Total_Lq, H_q, 64), lse: (H_q, Total_Lq)).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_gqa_varlen_fwd(
    const at::Tensor& x,                      // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& w_qkv,                  // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                  // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,           // (B + 1,) [Int32]
    int64_t max_seqlen_q,                     // scalar: int64
    int64_t H_q,                              // scalar: int64
    int64_t H_kv,                             // scalar: int64
    const std::optional<at::Tensor>& cos,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,     // (L_max, 64) [FP16] (optional)
    bool is_causal,                           // scalar: bool
    const std::optional<at::Tensor>& offsets, // (B,) [Int32] (optional)
    bool save_for_bwd                         // scalar: bool
) {
    return gqa_varlen_fwd_impl(
        x, w_qkv, w_out, cu_seqlens_q, max_seqlen_q, H_q, H_kv,
        cos, sin, is_causal, offsets, save_for_bwd,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt
    );
}

// 2. Self-Attention Prefill with Paged KV Cache
// Returns out: (Total_Lq, embed_dim) [FP16].
at::Tensor fused_gqa_varlen_prefill(
    const at::Tensor& x,                      // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& w_qkv,                  // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                  // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,           // (B + 1,) [Int32]
    int64_t max_seqlen_q,                     // scalar: int64
    int64_t H_q,                              // scalar: int64
    int64_t H_kv,                             // scalar: int64
    at::Tensor& k_cache,                      // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,                      // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,            // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,                // (B,) [Int32]
    const std::optional<at::Tensor>& cos,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,     // (L_max, 64) [FP16] (optional)
    bool is_causal,                           // scalar: bool
    const std::optional<at::Tensor>& offsets  // (B,) [Int32] (optional)
) {
    auto res = gqa_varlen_fwd_impl(
        x, w_qkv, w_out, cu_seqlens_q, max_seqlen_q, H_q, H_kv,
        cos, sin, is_causal, offsets, false,
        k_cache, v_cache, block_table, cache_seqlens
    );
    return std::get<0>(res);
}

// 3. Self-Attention Backward (single d_qkv allocation, zero cat/transpose)
// Returns (dx: (Total_Lq, embed_dim), dw_qkv: ((H_q + 2*H_kv)*64, embed_dim), dw_out: (embed_dim, embed_dim)).
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_gqa_varlen_bwd(
    const at::Tensor& grad_out,               // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& x,                      // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& qkv,                    // (Total_Lq, H_q + 2*H_kv, 64) [FP16]
    const at::Tensor& attn_out,               // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& lse,                    // (H_q, Total_Lq) [FP32]
    const at::Tensor& w_qkv,                  // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                  // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,           // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets, // (B,) [Int32] (optional)
    int64_t max_seqlen_q,                     // scalar: int64
    int64_t H_q,                              // scalar: int64
    int64_t H_kv,                             // scalar: int64
    const std::optional<at::Tensor>& cos,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,     // (L_max, 64) [FP16] (optional)
    bool is_causal,                           // scalar: bool
    bool needs_dx,                            // scalar: bool
    bool needs_dw_qkv,                        // scalar: bool
    bool needs_dw_out                         // scalar: bool
) {
    // 1. Input validation
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && qkv.is_cuda() && attn_out.is_cuda() &&
                lse.is_cuda() && w_qkv.is_cuda() && w_out.is_cuda() && cu_seqlens_q.is_cuda(),
                "All inputs to fused_gqa_varlen_bwd must be CUDA tensors");
    TORCH_CHECK(
        x.device() == grad_out.device() && qkv.device() == grad_out.device() &&
        attn_out.device() == grad_out.device() && lse.device() == grad_out.device() &&
        w_qkv.device() == grad_out.device() && w_out.device() == grad_out.device() &&
        cu_seqlens_q.device() == grad_out.device(),
        "All inputs must reside on the same CUDA device"
    );
    TORCH_CHECK(
        grad_out.scalar_type() == at::kHalf && x.scalar_type() == at::kHalf &&
        qkv.scalar_type() == at::kHalf && attn_out.scalar_type() == at::kHalf &&
        w_qkv.scalar_type() == at::kHalf && w_out.scalar_type() == at::kHalf &&
        lse.scalar_type() == at::kFloat,
        "Invalid tensor dtypes for fused_gqa_varlen_bwd"
    );
    TORCH_CHECK(qkv.is_contiguous() && attn_out.is_contiguous() && lse.is_contiguous(),
                "Saved qkv, attn_out, and lse must be contiguous");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be a positive multiple of H_kv");

    const int64_t Total_Lq  = x.size(0);
    const int64_t embed_dim = H_q * 64;
    const int64_t qkv_heads = H_q + 2 * H_kv;
    const int64_t qkv_dim   = qkv_heads * 64;

    TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == Total_Lq && grad_out.size(1) == embed_dim,
                "grad_out must have shape (Total_Lq, H_q * 64)");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == embed_dim, "x must have shape (Total_Lq, H_q * 64)");
    TORCH_CHECK(qkv.dim() == 3 && qkv.size(0) == Total_Lq && qkv.size(1) == qkv_heads && qkv.size(2) == 64,
                "qkv must have shape (Total_Lq, H_q + 2*H_kv, 64)");
    TORCH_CHECK(attn_out.dim() == 3 && attn_out.size(0) == Total_Lq && attn_out.size(1) == H_q && attn_out.size(2) == 64,
                "attn_out must have shape (Total_Lq, H_q, 64)");
    TORCH_CHECK(w_qkv.dim() == 2 && w_qkv.size(0) == qkv_dim && w_qkv.size(1) == embed_dim,
                "w_qkv must have shape ((H_q + 2*H_kv)*64, H_q*64)");
    TORCH_CHECK(w_out.dim() == 2 && w_out.size(0) == embed_dim && w_out.size(1) == embed_dim,
                "w_out must have shape (H_q*64, H_q*64)");

    const c10::cuda::CUDAGuard device_guard(grad_out.device());

    // 2. Output projection backward
    at::Tensor dw_out;
    if (needs_dw_out) {
        dw_out = at::mm(grad_out.t(), attn_out.view({Total_Lq, embed_dim}));
    }

    at::Tensor dx;
    at::Tensor dw_qkv;
    if (needs_dx || needs_dw_qkv) {
        at::Tensor d_attn_out = at::mm(grad_out, w_out).view({Total_Lq, H_q, 64});
        at::Tensor q = qkv.narrow(1, 0, H_q);
        at::Tensor k = qkv.narrow(1, H_q, H_kv);
        at::Tensor v = qkv.narrow(1, H_q + H_kv, H_kv);

        // Single d_qkv allocation receives dq, dk, dv directly without torch::cat
        at::Tensor d_qkv = at::empty({Total_Lq, qkv_heads, 64}, x.options());
        at::Tensor dq = d_qkv.narrow(1, 0, H_q);
        at::Tensor dk = d_qkv.narrow(1, H_q, H_kv);
        at::Tensor dv = d_qkv.narrow(1, H_q + H_kv, H_kv);

        launch_fused_mha_varlen_bwd_into(
            d_attn_out, q, k, v, attn_out, lse,
            cu_seqlens_q, cu_seqlens_q, offsets, max_seqlen_q,
            dq, dk, dv, cos, sin, is_causal
        );
        d_attn_out = at::Tensor();

        at::Tensor d_qkv_2d = d_qkv.view({Total_Lq, qkv_dim});
        if (needs_dx) {
            dx = at::mm(d_qkv_2d, w_qkv);
        }
        if (needs_dw_qkv) {
            dw_qkv = at::mm(d_qkv_2d.t(), x);
        }
    }

    return std::make_tuple(dx, dw_qkv, dw_out);
}

// 4. Cross-Attention Forward
// Returns (out: (Total_Lq, embed_dim), q: (Total_Lq, H_q, 64), kv: (Total_Lk, 2*H_kv, 64), attn_out: (Total_Lq, H_q, 64), lse: (H_q, Total_Lq)).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_cross_gqa_varlen_fwd(
    const at::Tensor& x,                      // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& context,                // (Total_Lk, context_dim) [FP16]
    const at::Tensor& w_q,                    // (H_q*64, embed_dim) [FP16]
    const at::Tensor& w_kv,                   // (2*H_kv*64, context_dim) [FP16]
    const at::Tensor& w_out,                  // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,           // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,           // (B + 1,) [Int32]
    int64_t max_seqlen_q,                     // scalar: int64
    int64_t H_q,                              // scalar: int64
    int64_t H_kv,                             // scalar: int64
    const std::optional<at::Tensor>& cos,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& offsets, // (B,) [Int32] (optional)
    bool save_for_bwd                         // scalar: bool
) {
    // 1. Input validation
    TORCH_CHECK(x.is_cuda() && context.is_cuda() && w_q.is_cuda() && w_kv.is_cuda() && w_out.is_cuda() &&
                cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(),
                "All inputs to fused_cross_gqa_varlen_fwd must be CUDA tensors");
    TORCH_CHECK(
        context.device() == x.device() && w_q.device() == x.device() && w_kv.device() == x.device() &&
        w_out.device() == x.device() && cu_seqlens_q.device() == x.device() && cu_seqlens_k.device() == x.device(),
        "All inputs must reside on the same CUDA device"
    );
    TORCH_CHECK(
        x.scalar_type() == at::kHalf && context.scalar_type() == at::kHalf &&
        w_q.scalar_type() == at::kHalf && w_kv.scalar_type() == at::kHalf && w_out.scalar_type() == at::kHalf,
        "x, context, w_q, w_kv, and w_out must be float16"
    );
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be a positive multiple of H_kv");
    TORCH_CHECK(max_seqlen_q >= 0, "max_seqlen_q must be non-negative");

    const int64_t Total_Lq    = x.size(0);
    const int64_t Total_Lk    = context.size(0);
    const int64_t embed_dim   = H_q * 64;
    const int64_t kv_dim      = 2 * H_kv * 64;

    TORCH_CHECK(x.dim() == 2 && x.size(1) == embed_dim, "x must have shape (Total_Lq, H_q * 64)");
    TORCH_CHECK(context.dim() == 2, "context must be a 2D tensor (Total_Lk, context_dim)");
    const int64_t context_dim = context.size(1);

    TORCH_CHECK(w_q.dim() == 2 && w_q.size(0) == embed_dim && w_q.size(1) == embed_dim,
                "w_q must have shape (H_q*64, H_q*64)");
    TORCH_CHECK(w_kv.dim() == 2 && w_kv.size(0) == kv_dim && w_kv.size(1) == context_dim,
                "w_kv must have shape (2*H_kv*64, context_dim)");
    TORCH_CHECK(w_out.dim() == 2 && w_out.size(0) == embed_dim && w_out.size(1) == embed_dim,
                "w_out must have shape (H_q*64, H_q*64)");

    const c10::cuda::CUDAGuard device_guard(x.device());

    // 2. Q & KV projections
    at::Tensor q  = at::mm(x, w_q.t()).view({Total_Lq, H_q, 64});
    at::Tensor kv = at::mm(context, w_kv.t()).view({Total_Lk, 2 * H_kv, 64});
    at::Tensor k  = kv.narrow(1, 0, H_kv);
    at::Tensor v  = kv.narrow(1, H_kv, H_kv);

    // 3. Cross-attention forward (is_causal always false)
    at::Tensor attn_out = at::empty({Total_Lq, H_q, 64}, x.options());
    std::optional<at::Tensor> lse_opt = std::nullopt;
    if (save_for_bwd) {
        lse_opt = at::empty({H_q, Total_Lq}, x.options().dtype(at::kFloat));
    }

    launch_fused_mha_varlen_fwd_into(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        attn_out, lse_opt, cos, sin, false, offsets
    );

    // 4. Output projection
    at::Tensor out = at::mm(attn_out.view({Total_Lq, embed_dim}), w_out.t());

    if (!save_for_bwd) {
        return std::make_tuple(out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor());
    }
    return std::make_tuple(out, q, kv, attn_out, *lse_opt);
}

// 5. Cross-Attention Backward
// Returns (dx, d_context, dw_q, dw_kv, dw_out).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_cross_gqa_varlen_bwd(
    const at::Tensor& grad_out,               // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& x,                      // (Total_Lq, embed_dim) [FP16]
    const at::Tensor& context,                // (Total_Lk, context_dim) [FP16]
    const at::Tensor& q,                      // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& kv,                     // (Total_Lk, 2*H_kv, 64) [FP16]
    const at::Tensor& attn_out,               // (Total_Lq, H_q, 64) [FP16]
    const at::Tensor& lse,                    // (H_q, Total_Lq) [FP32]
    const at::Tensor& w_q,                    // (H_q*64, embed_dim) [FP16]
    const at::Tensor& w_kv,                   // (2*H_kv*64, context_dim) [FP16]
    const at::Tensor& w_out,                  // (embed_dim, H_q*64) [FP16]
    const at::Tensor& cu_seqlens_q,           // (B + 1,) [Int32]
    const at::Tensor& cu_seqlens_k,           // (B + 1,) [Int32]
    const std::optional<at::Tensor>& offsets, // (B,) [Int32] (optional)
    int64_t max_seqlen_k,                     // scalar: int64
    int64_t H_q,                              // scalar: int64
    int64_t H_kv,                             // scalar: int64
    const std::optional<at::Tensor>& cos,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,     // (L_max, 64) [FP16] (optional)
    bool needs_dx,                            // scalar: bool
    bool needs_dctx,                          // scalar: bool
    bool needs_dw_q,                          // scalar: bool
    bool needs_dw_kv,                         // scalar: bool
    bool needs_dw_out                         // scalar: bool
) {
    // 1. Input validation
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && context.is_cuda() && q.is_cuda() && kv.is_cuda() &&
                attn_out.is_cuda() && lse.is_cuda() && w_q.is_cuda() && w_kv.is_cuda() && w_out.is_cuda() &&
                cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(),
                "All inputs to fused_cross_gqa_varlen_bwd must be CUDA tensors");
    TORCH_CHECK(
        x.device() == grad_out.device() && context.device() == grad_out.device() &&
        q.device() == grad_out.device() && kv.device() == grad_out.device() &&
        attn_out.device() == grad_out.device() && lse.device() == grad_out.device() &&
        w_q.device() == grad_out.device() && w_kv.device() == grad_out.device() &&
        w_out.device() == grad_out.device() && cu_seqlens_q.device() == grad_out.device() &&
        cu_seqlens_k.device() == grad_out.device(),
        "All inputs must reside on the same CUDA device"
    );
    TORCH_CHECK(
        grad_out.scalar_type() == at::kHalf && x.scalar_type() == at::kHalf &&
        context.scalar_type() == at::kHalf && q.scalar_type() == at::kHalf &&
        kv.scalar_type() == at::kHalf && attn_out.scalar_type() == at::kHalf &&
        w_q.scalar_type() == at::kHalf && w_kv.scalar_type() == at::kHalf &&
        w_out.scalar_type() == at::kHalf && lse.scalar_type() == at::kFloat,
        "Invalid tensor dtypes for fused_cross_gqa_varlen_bwd"
    );
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && attn_out.is_contiguous() && lse.is_contiguous(),
                "Saved q, kv, attn_out, and lse must be contiguous");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be a positive multiple of H_kv");

    const int64_t Total_Lq    = x.size(0);
    const int64_t Total_Lk    = context.size(0);
    const int64_t embed_dim   = H_q * 64;
    const int64_t context_dim = context.size(1);
    const int64_t kv_dim      = 2 * H_kv * 64;

    TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == Total_Lq && grad_out.size(1) == embed_dim,
                "grad_out must have shape (Total_Lq, H_q * 64)");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == embed_dim, "x must have shape (Total_Lq, H_q * 64)");
    TORCH_CHECK(context.dim() == 2, "context must be 2D");
    TORCH_CHECK(q.dim() == 3 && q.size(0) == Total_Lq && q.size(1) == H_q && q.size(2) == 64,
                "q must have shape (Total_Lq, H_q, 64)");
    TORCH_CHECK(kv.dim() == 3 && kv.size(0) == Total_Lk && kv.size(1) == 2 * H_kv && kv.size(2) == 64,
                "kv must have shape (Total_Lk, 2*H_kv, 64)");
    TORCH_CHECK(attn_out.dim() == 3 && attn_out.size(0) == Total_Lq && attn_out.size(1) == H_q && attn_out.size(2) == 64,
                "attn_out must have shape (Total_Lq, H_q, 64)");
    TORCH_CHECK(w_q.dim() == 2 && w_q.size(0) == embed_dim && w_q.size(1) == embed_dim,
                "w_q must have shape (H_q*64, H_q*64)");
    TORCH_CHECK(w_kv.dim() == 2 && w_kv.size(0) == kv_dim && w_kv.size(1) == context_dim,
                "w_kv must have shape (2*H_kv*64, context_dim)");
    TORCH_CHECK(w_out.dim() == 2 && w_out.size(0) == embed_dim && w_out.size(1) == embed_dim,
                "w_out must have shape (H_q*64, H_q*64)");

    const c10::cuda::CUDAGuard device_guard(grad_out.device());

    // 2. Output projection backward
    at::Tensor dw_out;
    if (needs_dw_out) {
        dw_out = at::mm(grad_out.t(), attn_out.view({Total_Lq, embed_dim}));
    }

    at::Tensor dx;
    at::Tensor d_context;
    at::Tensor dw_q;
    at::Tensor dw_kv;
    if (needs_dx || needs_dctx || needs_dw_q || needs_dw_kv) {
        at::Tensor d_attn_out = at::mm(grad_out, w_out).view({Total_Lq, H_q, 64});
        at::Tensor k = kv.narrow(1, 0, H_kv);
        at::Tensor v = kv.narrow(1, H_kv, H_kv);

        at::Tensor dq   = at::empty({Total_Lq, H_q, 64}, x.options());
        at::Tensor d_kv = at::empty({Total_Lk, 2 * H_kv, 64}, context.options());
        at::Tensor dk   = d_kv.narrow(1, 0, H_kv);
        at::Tensor dv   = d_kv.narrow(1, H_kv, H_kv);

        launch_fused_mha_varlen_bwd_into(
            d_attn_out, q, k, v, attn_out, lse,
            cu_seqlens_q, cu_seqlens_k, offsets, max_seqlen_k,
            dq, dk, dv, cos, sin, false
        );
        d_attn_out = at::Tensor();

        at::Tensor dq_2d   = dq.view({Total_Lq, embed_dim});
        at::Tensor d_kv_2d = d_kv.view({Total_Lk, kv_dim});
        if (needs_dx) {
            dx = at::mm(dq_2d, w_q);
        }
        if (needs_dw_q) {
            dw_q = at::mm(dq_2d.t(), x);
        }
        if (needs_dctx) {
            d_context = at::mm(d_kv_2d, w_kv);
        }
        if (needs_dw_kv) {
            dw_kv = at::mm(d_kv_2d.t(), context);
        }
    }

    return std::make_tuple(dx, d_context, dw_q, dw_kv, dw_out);
}

// 6. Self-Attention Decode (Lq = 1): QKV proj -> Paged KV Append -> Decode -> Out proj
// Returns out: (B, embed_dim) [FP16].
at::Tensor fused_gqa_varlen_decode(
    const at::Tensor& x,                      // (B, embed_dim) [FP16]
    const at::Tensor& w_qkv,                  // ((H_q + 2*H_kv)*64, embed_dim) [FP16]
    const at::Tensor& w_out,                  // (embed_dim, H_q*64) [FP16]
    at::Tensor& k_cache,                      // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,                      // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,            // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,                // (B,) [Int32]
    int64_t max_seqlen_k,                     // scalar: int64
    int64_t H_q,                              // scalar: int64
    int64_t H_kv,                             // scalar: int64
    const std::optional<at::Tensor>& cos,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& sin,     // (L_max, 64) [FP16] (optional)
    const std::optional<at::Tensor>& offsets, // (B,) [Int32] (optional)
    std::optional<bool> use_splitk            // scalar: optional<bool>
) {
    // 1. Input validation
    TORCH_CHECK(x.is_cuda() && w_qkv.is_cuda() && w_out.is_cuda(),
                "x, w_qkv, and w_out must be CUDA tensors");
    TORCH_CHECK(w_qkv.device() == x.device() && w_out.device() == x.device(),
                "x, w_qkv, and w_out must reside on the same CUDA device");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf && w_qkv.scalar_type() == at::kHalf && w_out.scalar_type() == at::kHalf,
        "x, w_qkv, and w_out must be float16"
    );
    TORCH_CHECK(x.dim() == 2 && w_qkv.dim() == 2 && w_out.dim() == 2,
                "Expected 2D x, w_qkv, and w_out");
    TORCH_CHECK(H_q > 0 && H_kv > 0 && (H_q % H_kv == 0), "H_q must be a positive multiple of H_kv");

    const int64_t B         = x.size(0);
    const int64_t embed_dim = H_q * 64;
    const int64_t qkv_heads = H_q + 2 * H_kv;
    const int64_t qkv_dim   = qkv_heads * 64;

    TORCH_CHECK(x.size(1) == embed_dim, "x.size(1) must equal H_q * 64");
    TORCH_CHECK(w_qkv.size(0) == qkv_dim && w_qkv.size(1) == embed_dim,
                "w_qkv must have shape ((H_q + 2*H_kv)*64, H_q*64)");
    TORCH_CHECK(w_out.size(0) == embed_dim && w_out.size(1) == embed_dim,
                "w_out must have shape (H_q*64, H_q*64)");

    const c10::cuda::CUDAGuard device_guard(x.device());
    if (B == 0) {
        return at::empty({0, embed_dim}, x.options());
    }

    // 2. QKV projection -> zero-copy strided slices
    at::Tensor qkv   = at::mm(x, w_qkv.t()).view({B, qkv_heads, 64});
    at::Tensor q     = qkv.narrow(1, 0, H_q);
    at::Tensor new_k = qkv.narrow(1, H_q, H_kv);
    at::Tensor new_v = qkv.narrow(1, H_q + H_kv, H_kv);

    // 3. O(B) Paged KV Append + Paged Flash Decoding
    launch_paged_kv_append(new_k, new_v, k_cache, v_cache, block_table, cache_seqlens);

    at::Tensor attn_out = at::empty({B, H_q, 64}, x.options());
    launch_fused_mha_varlen_decode_into(
        q, k_cache, v_cache, block_table, cache_seqlens,
        max_seqlen_k, attn_out, cos, sin, offsets, use_splitk
    );

    // 4. Output projection
    return at::mm(attn_out.view({B, embed_dim}), w_out.t());
}
