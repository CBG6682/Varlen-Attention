"""PyTorch C++ extension loader, FakeTensor registrations, and autograd wrappers."""

from typing import Optional, Tuple

import torch

try:
    from varlen_transformer import _C as _C_ext  # noqa: F401
    HAS_CUDA_EXT = True
except ImportError:
    _C_ext = None
    HAS_CUDA_EXT = False


def _require_cuda_ext() -> None:
    if not HAS_CUDA_EXT:
        raise RuntimeError(
            "varlen_transformer._C CUDA extension is not built/installed. "
            "Build it with `pip install -e . --no-build-isolation` before calling CUDA kernels."
        )


def _contiguous_or_none(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _infer_max_seqlen(cu_seqlens: torch.Tensor) -> int:
    if cu_seqlens.numel() <= 1:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


# =========================================================================================
# FakeTensor / Meta Kernel Registrations for torch.compile & TorchInductor
# =========================================================================================

_register_fake = getattr(torch.library, "register_fake", None)
if _register_fake is None:
    _register_fake = getattr(torch.library, "impl_abstract", None)

if HAS_CUDA_EXT and _register_fake is not None:

    @_register_fake("varlen_transformer::fused_mha_varlen_fwd")
    def _fused_mha_varlen_fwd_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o = torch.empty_like(q)
        lse = q.new_empty((q.shape[0], q.shape[1]), dtype=torch.float32)
        return o, lse

    @_register_fake("varlen_transformer::fused_mha_varlen_bwd")
    def _fused_mha_varlen_bwd_fake(
        do_grad: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        offsets: Optional[torch.Tensor],
        max_seqlen_k: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    @_register_fake("varlen_transformer::fused_mha_varlen_decode")
    def _fused_mha_varlen_decode_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.empty_like(q)


# =========================================================================================
# Thin Dispatcher Wrappers around torch.ops.varlen_transformer
# =========================================================================================

def fused_mha_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw dispatcher wrapper for varlen forward attention + optional fused RoPE."""
    _require_cuda_ext()
    return torch.ops.varlen_transformer.fused_mha_varlen_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        int(max_seqlen_q),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        bool(is_causal),
        _contiguous_or_none(offsets),
    )


def fused_mha_varlen_bwd(
    do_grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    offsets: Optional[torch.Tensor],
    max_seqlen_k: int,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw dispatcher wrapper for varlen backward attention + optional fused RoPE."""
    _require_cuda_ext()
    return torch.ops.varlen_transformer.fused_mha_varlen_bwd(
        do_grad.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        o.contiguous(),
        lse.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        _contiguous_or_none(offsets),
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        bool(is_causal),
    )


def fused_mha_varlen_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Raw dispatcher wrapper for Split-K varlen Flash Decoding (Lq = 1)."""
    _require_cuda_ext()
    return torch.ops.varlen_transformer.fused_mha_varlen_decode(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        cu_seqlens_k.contiguous(),
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        _contiguous_or_none(offsets),
    )


# =========================================================================================
# PyTorch Autograd Function & Public High-Level APIs
# =========================================================================================

class _VarlenFlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        is_causal: bool,
        offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        o, lse = torch.ops.varlen_transformer.fused_mha_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, cos, sin, is_causal, offsets
        )
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, cos, sin, offsets)
        ctx.max_seqlen_k = int(max_seqlen_k)
        ctx.is_causal = bool(is_causal)
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, cos, sin, offsets = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        dq, dk, dv = torch.ops.varlen_transformer.fused_mha_varlen_bwd(
            grad_o,
            q,
            k,
            v,
            o,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            offsets,
            ctx.max_seqlen_k,
            cos,
            sin,
            ctx.is_causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def varlen_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Autograd- and torch.compile-compatible variable-length FlashAttention with fused RoPE.

    Args:
        q: (H_q, Total_Lq, 64) float16
        k: (H_kv, Total_Lk, 64) float16
        v: (H_kv, Total_Lk, 64) float16
        cu_seqlens_q: (B + 1,) int32
        cu_seqlens_k: (B + 1,) int32, defaults to cu_seqlens_q
        max_seqlen_q: int, inferred from cu_seqlens_q if None
        max_seqlen_k: int, inferred from cu_seqlens_k if None
        cos: Optional (L_max, 64) float16
        sin: Optional (L_max, 64) float16
        is_causal: bool causal mask flag
        offsets: Optional (B,) int32 RoPE position offsets per sequence
    Returns:
        o: (H_q, Total_Lq, 64) float16
    """
    _require_cuda_ext()

    cu_q = cu_seqlens_q.contiguous()
    if cu_seqlens_k is None:
        cu_k = cu_q
    else:
        cu_k = cu_seqlens_k.contiguous()

    if max_seqlen_q is None:
        max_seqlen_q = _infer_max_seqlen(cu_q)
    if max_seqlen_k is None:
        max_seqlen_k = max_seqlen_q if cu_k is cu_q else _infer_max_seqlen(cu_k)

    return _VarlenFlashAttentionFunc.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_q,
        cu_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        bool(is_causal),
        _contiguous_or_none(offsets),
    )


def varlen_flash_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Variable-length Flash Decoding (Lq = 1) with fused RoPE on raw unrotated KV cache.

    Args:
        q: (H_q, B, 64) float16
        k_cache: (H_kv, Total_Lk, 64) float16 (raw unrotated keys)
        v_cache: (H_kv, Total_Lk, 64) float16
        cu_seqlens_k: (B + 1,) int32
        max_seqlen_k: int, inferred from cu_seqlens_k if None
        cos: Optional (L_max, 64) float16
        sin: Optional (L_max, 64) float16
        offsets: Optional (B,) int32 RoPE position offsets per sequence
    Returns:
        o: (H_q, B, 64) float16
    """
    _require_cuda_ext()
    cu_k = cu_seqlens_k.contiguous()
    if max_seqlen_k is None:
        max_seqlen_k = _infer_max_seqlen(cu_k)

    return torch.ops.varlen_transformer.fused_mha_varlen_decode(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        cu_k,
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        _contiguous_or_none(offsets),
    )


__all__ = [
    "HAS_CUDA_EXT",
    "fused_mha_varlen_fwd",
    "fused_mha_varlen_bwd",
    "fused_mha_varlen_decode",
    "varlen_flash_attention",
    "varlen_flash_decode",
]
