"""Varlen Transformer: High-Performance Variable-Length FlashAttention & Fused RoPE CUDA Kernels."""

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
    varlen_flash_attention,
    varlen_flash_decode,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "HAS_CUDA_EXT",
    "fused_mha_varlen_fwd",
    "fused_mha_varlen_bwd",
    "fused_mha_varlen_decode",
    "varlen_flash_attention",
    "varlen_flash_decode",
    "RotaryPositionalEmbedding",
    "VarlenKVCache",
    "varlen_attention_ref",
    "varlen_decode_ref",
    "VarlenFlashGQA",
    "VarlenFlashGQABaselineCorrectness",
    "BatchedFlashGQABaselineSpeed",
    "FlashGQA",
]
