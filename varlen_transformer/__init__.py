"""Varlen Transformer: Pure THD FlashAttention, Paged KV Cache & Fused Eager GQA Modules."""

from varlen_transformer.model import (
    BatchedFlashGQABaselineSpeed,
    FlashGQA,
    RotaryPositionalEmbedding,
    VarlenFlashGQA,
    VarlenFlashGQABaselineCorrectness,
    VarlenKVCache,
    varlen_attention_ref,
    varlen_decode_ref,
)
from varlen_transformer.ops import (
    HAS_CUDA_EXT,
    fused_mha_varlen_bwd,
    fused_mha_varlen_decode,
    fused_mha_varlen_fwd,
    paged_kv_append,
    prefill_to_paged_kv,
    varlen_cross_gqa_attention,
    varlen_flash_attention,
    varlen_flash_decode,
    varlen_gqa_attention,
    varlen_gqa_decode,
    varlen_gqa_prefill,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "HAS_CUDA_EXT",
    "fused_mha_varlen_fwd",
    "fused_mha_varlen_bwd",
    "prefill_to_paged_kv",
    "paged_kv_append",
    "fused_mha_varlen_decode",
    "varlen_flash_attention",
    "varlen_flash_decode",
    "varlen_gqa_attention",
    "varlen_gqa_prefill",
    "varlen_cross_gqa_attention",
    "varlen_gqa_decode",
    "RotaryPositionalEmbedding",
    "VarlenKVCache",
    "varlen_attention_ref",
    "varlen_decode_ref",
    "VarlenFlashGQA",
    "VarlenFlashGQABaselineCorrectness",
    "BatchedFlashGQABaselineSpeed",
    "FlashGQA",
]
