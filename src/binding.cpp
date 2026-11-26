#include <torch/extension.h>
#include <torch/library.h>
#include "mha_ops.cuh"

// =========================================================================================
// 1. PyTorch Dispatcher Registration (TORCH_LIBRARY)
// =========================================================================================

TORCH_LIBRARY(varlen_transformer, m) {
    // 1. Core Varlen FlashAttention + Fused RoPE Forward (THD layout)
    m.def(
        "fused_mha_varlen_fwd("
        "Tensor q, Tensor k, Tensor v, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False, Tensor? offsets=None"
        ") -> (Tensor, Tensor)"
    );

    // 2. Core Varlen FlashAttention + Fused RoPE Backward (THD layout)
    m.def(
        "fused_mha_varlen_bwd("
        "Tensor do_grad, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? offsets, int max_seqlen_k, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False"
        ") -> (Tensor, Tensor, Tensor)"
    );

    // 3. Paged KV Cache Prefill & Append
    m.def(
        "prefill_to_paged_kv("
        "Tensor k, Tensor v, Tensor cu_seqlens_k, "
        "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor block_table, Tensor(c!) cache_seqlens, "
        "int max_seqlen_k"
        ") -> ()"
    );

    m.def(
        "paged_kv_append("
        "Tensor new_k, Tensor new_v, "
        "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor block_table, Tensor(c!) cache_seqlens"
        ") -> ()"
    );

    // 4. Core Paged Varlen Flash Decoding (Lq = 1)
    m.def(
        "fused_mha_varlen_decode("
        "Tensor q, Tensor k_cache, Tensor v_cache, "
        "Tensor block_table, Tensor cache_seqlens, int max_seqlen_k, "
        "Tensor? cos=None, Tensor? sin=None, Tensor? offsets=None, bool? use_splitk=None"
        ") -> Tensor"
    );

    // 5. Fused GQA Linear + Varlen Attention Blocks (bias-free)
    m.def(
        "fused_gqa_varlen_fwd("
        "Tensor x, Tensor w_qkv, Tensor w_out, Tensor cu_seqlens_q, int max_seqlen_q, "
        "int h_q, int h_kv, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False, Tensor? offsets=None, "
        "bool save_for_bwd=True"
        ") -> (Tensor, Tensor, Tensor, Tensor)"
    );

    m.def(
        "fused_gqa_varlen_prefill("
        "Tensor x, Tensor w_qkv, Tensor w_out, Tensor cu_seqlens_q, int max_seqlen_q, "
        "int h_q, int h_kv, "
        "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor block_table, Tensor(c!) cache_seqlens, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False, Tensor? offsets=None"
        ") -> Tensor"
    );

    m.def(
        "fused_gqa_varlen_bwd("
        "Tensor grad_out, Tensor x, Tensor qkv, Tensor attn_out, Tensor lse, "
        "Tensor w_qkv, Tensor w_out, Tensor cu_seqlens_q, Tensor? offsets, "
        "int max_seqlen_q, int h_q, int h_kv, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False, "
        "bool needs_dx=True, bool needs_dw_qkv=True, bool needs_dw_out=True"
        ") -> (Tensor, Tensor, Tensor)"
    );

    m.def(
        "fused_cross_gqa_varlen_fwd("
        "Tensor x, Tensor context, Tensor w_q, Tensor w_kv, Tensor w_out, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, int h_q, int h_kv, "
        "Tensor? cos=None, Tensor? sin=None, Tensor? offsets=None, bool save_for_bwd=True"
        ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    );

    m.def(
        "fused_cross_gqa_varlen_bwd("
        "Tensor grad_out, Tensor x, Tensor context, Tensor q, Tensor kv, Tensor attn_out, Tensor lse, "
        "Tensor w_q, Tensor w_kv, Tensor w_out, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? offsets, "
        "int max_seqlen_k, int h_q, int h_kv, "
        "Tensor? cos=None, Tensor? sin=None, "
        "bool needs_dx=True, bool needs_dctx=True, bool needs_dw_q=True, "
        "bool needs_dw_kv=True, bool needs_dw_out=True"
        ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    );

    m.def(
        "fused_gqa_varlen_decode("
        "Tensor x, Tensor w_qkv, Tensor w_out, "
        "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor block_table, Tensor(c!) cache_seqlens, "
        "int max_seqlen_k, int h_q, int h_kv, "
        "Tensor? cos=None, Tensor? sin=None, Tensor? offsets=None, bool? use_splitk=None"
        ") -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(varlen_transformer, CUDA, m) {
    m.impl("fused_mha_varlen_fwd", &fused_mha_varlen_fwd);
    m.impl("fused_mha_varlen_bwd", &fused_mha_varlen_bwd);
    m.impl("prefill_to_paged_kv", &prefill_to_paged_kv);
    m.impl("paged_kv_append", &paged_kv_append);
    m.impl("fused_mha_varlen_decode", &fused_mha_varlen_decode);
    m.impl("fused_gqa_varlen_fwd", &fused_gqa_varlen_fwd);
    m.impl("fused_gqa_varlen_prefill", &fused_gqa_varlen_prefill);
    m.impl("fused_gqa_varlen_bwd", &fused_gqa_varlen_bwd);
    m.impl("fused_cross_gqa_varlen_fwd", &fused_cross_gqa_varlen_fwd);
    m.impl("fused_cross_gqa_varlen_bwd", &fused_cross_gqa_varlen_bwd);
    m.impl("fused_gqa_varlen_decode", &fused_gqa_varlen_decode);
}

// =========================================================================================
// 2. Python C++ Extension Module (PYBIND11_MODULE)
// Direct low-overhead eager bindings via varlen_transformer._C
// =========================================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Varlen Transformer CUDA Extension (Pure THD + Paged KV + Fused GQA / SM75)";

    m.def(
        "fused_mha_varlen_fwd",
        &fused_mha_varlen_fwd,
        "Varlen FlashAttention + Fused RoPE Forward (THD)\n\n"
        "Outputs:\n"
        "  1. o:   (Total_Lq, H_q, 64) [FP16]\n"
        "  2. lse: (H_q, Total_Lq)     [FP32]",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("max_seqlen_q"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("is_causal") = false,
        py::arg("offsets") = py::none()
    );

    m.def(
        "fused_mha_varlen_bwd",
        &fused_mha_varlen_bwd,
        "Varlen FlashAttention + Fused RoPE Backward (THD)\n\n"
        "Outputs:\n"
        "  1. dq: (Total_Lq, H_q, 64)  [FP16]\n"
        "  2. dk: (Total_Lk, H_kv, 64) [FP16]\n"
        "  3. dv: (Total_Lk, H_kv, 64) [FP16]",
        py::arg("do_grad"),
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("o"),
        py::arg("lse"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("offsets"),
        py::arg("max_seqlen_k"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("is_causal") = false
    );

    m.def(
        "prefill_to_paged_kv",
        &prefill_to_paged_kv,
        "Copy packed THD K/V into Paged KV Cache pool and initialize cache_seqlens",
        py::arg("k"),
        py::arg("v"),
        py::arg("cu_seqlens_k"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("block_table"),
        py::arg("cache_seqlens"),
        py::arg("max_seqlen_k")
    );

    m.def(
        "paged_kv_append",
        &paged_kv_append,
        "Append 1 token per sequence into Paged KV Cache pool and increment cache_seqlens",
        py::arg("new_k"),
        py::arg("new_v"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("block_table"),
        py::arg("cache_seqlens")
    );

    m.def(
        "fused_mha_varlen_decode",
        &fused_mha_varlen_decode,
        "Paged Varlen Flash Decoding + Fused RoPE (L_q = 1)\n\n"
        "Output:\n"
        "  1. o: (B, H_q, 64) [FP16]",
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("block_table"),
        py::arg("cache_seqlens"),
        py::arg("max_seqlen_k"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("offsets") = py::none(),
        py::arg("use_splitk") = py::none()
    );

    m.def(
        "fused_gqa_varlen_fwd",
        &fused_gqa_varlen_fwd,
        "Fused Self-Attention Linear + Varlen FlashAttention Forward (bias-free)",
        py::arg("x"),
        py::arg("w_qkv"),
        py::arg("w_out"),
        py::arg("cu_seqlens_q"),
        py::arg("max_seqlen_q"),
        py::arg("h_q"),
        py::arg("h_kv"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("is_causal") = false,
        py::arg("offsets") = py::none(),
        py::arg("save_for_bwd") = true
    );

    m.def(
        "fused_gqa_varlen_prefill",
        &fused_gqa_varlen_prefill,
        "Fused Self-Attention Linear + Paged KV Prefill + Varlen FlashAttention Forward",
        py::arg("x"),
        py::arg("w_qkv"),
        py::arg("w_out"),
        py::arg("cu_seqlens_q"),
        py::arg("max_seqlen_q"),
        py::arg("h_q"),
        py::arg("h_kv"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("block_table"),
        py::arg("cache_seqlens"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("is_causal") = false,
        py::arg("offsets") = py::none()
    );

    m.def(
        "fused_gqa_varlen_bwd",
        &fused_gqa_varlen_bwd,
        "Fused Self-Attention Backward (single d_qkv buffer, bias-free)",
        py::arg("grad_out"),
        py::arg("x"),
        py::arg("qkv"),
        py::arg("attn_out"),
        py::arg("lse"),
        py::arg("w_qkv"),
        py::arg("w_out"),
        py::arg("cu_seqlens_q"),
        py::arg("offsets"),
        py::arg("max_seqlen_q"),
        py::arg("h_q"),
        py::arg("h_kv"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("is_causal") = false,
        py::arg("needs_dx") = true,
        py::arg("needs_dw_qkv") = true,
        py::arg("needs_dw_out") = true
    );

    m.def(
        "fused_cross_gqa_varlen_fwd",
        &fused_cross_gqa_varlen_fwd,
        "Fused Cross-Attention Linear + Varlen FlashAttention Forward (bias-free)",
        py::arg("x"),
        py::arg("context"),
        py::arg("w_q"),
        py::arg("w_kv"),
        py::arg("w_out"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("max_seqlen_q"),
        py::arg("h_q"),
        py::arg("h_kv"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("offsets") = py::none(),
        py::arg("save_for_bwd") = true
    );

    m.def(
        "fused_cross_gqa_varlen_bwd",
        &fused_cross_gqa_varlen_bwd,
        "Fused Cross-Attention Backward (bias-free)",
        py::arg("grad_out"),
        py::arg("x"),
        py::arg("context"),
        py::arg("q"),
        py::arg("kv"),
        py::arg("attn_out"),
        py::arg("lse"),
        py::arg("w_q"),
        py::arg("w_kv"),
        py::arg("w_out"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("offsets"),
        py::arg("max_seqlen_k"),
        py::arg("h_q"),
        py::arg("h_kv"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("needs_dx") = true,
        py::arg("needs_dctx") = true,
        py::arg("needs_dw_q") = true,
        py::arg("needs_dw_kv") = true,
        py::arg("needs_dw_out") = true
    );

    m.def(
        "fused_gqa_varlen_decode",
        &fused_gqa_varlen_decode,
        "Fused Self-Attention Decode Step (QKV proj -> Paged KV Append -> Decode -> Out proj)",
        py::arg("x"),
        py::arg("w_qkv"),
        py::arg("w_out"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("block_table"),
        py::arg("cache_seqlens"),
        py::arg("max_seqlen_k"),
        py::arg("h_q"),
        py::arg("h_kv"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("offsets") = py::none(),
        py::arg("use_splitk") = py::none()
    );
}
