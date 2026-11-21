#include <torch/extension.h>
#include <torch/library.h>
#include "mha_ops.cuh"

// =========================================================================================
// 1. PyTorch Dispatcher Registration (TORCH_LIBRARY)
// Enables torch.compile / TorchInductor without graph breaks
// =========================================================================================

TORCH_LIBRARY(varlen_transformer, m) {
    // -------------------------------------------------------------------------------------
    // 1. fused_mha_varlen_fwd
    // Outputs (Tensor, Tensor):
    //   1. o:   (H_q, Total_Lq, 64) [FP16] - Packed attention output
    //   2. lse: (H_q, Total_Lq)     [FP32] - Log-Sum-Exp for backward pass
    // -------------------------------------------------------------------------------------
    m.def(
        "fused_mha_varlen_fwd("
        "Tensor q, Tensor k, Tensor v, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False, Tensor? offsets=None"
        ") -> (Tensor, Tensor)"
    );

    // -------------------------------------------------------------------------------------
    // 2. fused_mha_varlen_bwd
    // Outputs (Tensor, Tensor, Tensor):
    //   1. dq: (H_q, Total_Lq, 64)  [FP16] - Gradient theo Query
    //   2. dk: (H_kv, Total_Lk, 64) [FP16] - Gradient theo Key
    //   3. dv: (H_kv, Total_Lk, 64) [FP16] - Gradient theo Value
    // -------------------------------------------------------------------------------------
    m.def(
        "fused_mha_varlen_bwd("
        "Tensor do_grad, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? offsets, int max_seqlen_k, "
        "Tensor? cos=None, Tensor? sin=None, bool is_causal=False"
        ") -> (Tensor, Tensor, Tensor)"
    );

    // -------------------------------------------------------------------------------------
    // 3. fused_mha_varlen_decode
    // Output (Tensor):
    //   1. o: (H_q, B, 64) [FP16] - Flash Decoding output
    // -------------------------------------------------------------------------------------
    m.def(
        "fused_mha_varlen_decode("
        "Tensor q, Tensor k_cache, Tensor v_cache, "
        "Tensor cu_seqlens_k, int max_seqlen_k, "
        "Tensor? cos=None, Tensor? sin=None, Tensor? offsets=None, bool? use_splitk=None"
        ") -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(varlen_transformer, CUDA, m) {
    m.impl("fused_mha_varlen_fwd", &fused_mha_varlen_fwd);
    m.impl("fused_mha_varlen_bwd", &fused_mha_varlen_bwd);
    m.impl("fused_mha_varlen_decode", &fused_mha_varlen_decode);
}

// =========================================================================================
// 2. Python C++ Extension Module (PYBIND11_MODULE)
// Direct Python bindings via varlen_transformer._C
// =========================================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Varlen Transformer CUDA Extension (Fused RoPE + FlashAttention + Flash Decoding / SM75)";

    m.def(
        "fused_mha_varlen_fwd",
        &fused_mha_varlen_fwd,
        "Varlen FlashAttention + Fused RoPE Forward\n\n"
        "Outputs:\n"
        "  1. o:   (H_q, Total_Lq, 64) [FP16]\n"
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
        "Varlen FlashAttention + Fused RoPE Backward\n\n"
        "Outputs:\n"
        "  1. dq: (H_q, Total_Lq, 64)  [FP16]\n"
        "  2. dk: (H_kv, Total_Lk, 64) [FP16]\n"
        "  3. dv: (H_kv, Total_Lk, 64) [FP16]",
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
        "fused_mha_varlen_decode",
        &fused_mha_varlen_decode,
        "Varlen Flash Decoding + Fused RoPE (L_q = 1)\n\n"
        "Output:\n"
        "  1. o: (H_q, B, 64) [FP16]",
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("cu_seqlens_k"),
        py::arg("max_seqlen_k"),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("offsets") = py::none(),
        py::arg("use_splitk") = py::none()
    );
}
