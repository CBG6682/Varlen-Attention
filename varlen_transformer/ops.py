"""PyTorch C++ extension loader, FakeTensor registrations, and low-overhead eager wrappers."""

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


def _ensure_strided_3d(tensor: torch.Tensor) -> torch.Tensor:
    """Keep zero-copy strided (L/B, H, 64) slices when 128-bit aligned; otherwise copy."""
    if (
        tensor.dim() == 3
        and tensor.stride(2) == 1
        and (tensor.stride(0) % 8 == 0)
        and (tensor.stride(1) % 8 == 0)
    ):
        return tensor
    return tensor.contiguous()


def _infer_max_seqlen(cu_seqlens: torch.Tensor) -> int:
    if cu_seqlens.numel() <= 1:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


_SYMINT_TYPE = getattr(torch, "SymInt", ())


def _preserve_symint(val: int) -> int:
    """Convert numeric scalars to Python int while keeping torch.SymInt symbolic."""
    if _SYMINT_TYPE and isinstance(val, _SYMINT_TYPE):
        return val
    return int(val)


# =========================================================================================
# FakeTensor / Meta Kernel Registrations (Compatibility for external export / tracing)
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
        o = q.new_empty((q.shape[0], q.shape[1], 64))
        lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
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

    @_register_fake("varlen_transformer::prefill_to_paged_kv")
    def _prefill_to_paged_kv_fake(
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
    ) -> None:
        return None

    @_register_fake("varlen_transformer::paged_kv_append")
    def _paged_kv_append_fake(
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        return None

    @_register_fake("varlen_transformer::fused_mha_varlen_decode")
    def _fused_mha_varlen_decode_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        use_splitk: Optional[bool] = None,
    ) -> torch.Tensor:
        return q.new_empty((q.shape[0], q.shape[1], 64))

    @_register_fake("varlen_transformer::fused_gqa_varlen_fwd")
    def _fused_gqa_varlen_fwd_fake(
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        offsets: Optional[torch.Tensor] = None,
        save_for_bwd: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_lq = x.shape[0]
        out = x.new_empty((total_lq, h_q * 64))
        if not save_for_bwd:
            empty = x.new_empty((0,))
            return out, empty, empty, x.new_empty((0,), dtype=torch.float32)
        qkv = x.new_empty((total_lq, h_q + 2 * h_kv, 64))
        attn_out = x.new_empty((total_lq, h_q, 64))
        lse = x.new_empty((h_q, total_lq), dtype=torch.float32)
        return out, qkv, attn_out, lse

    @_register_fake("varlen_transformer::fused_gqa_varlen_prefill")
    def _fused_gqa_varlen_prefill_fake(
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        h_q: int,
        h_kv: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], h_q * 64))

    @_register_fake("varlen_transformer::fused_gqa_varlen_bwd")
    def _fused_gqa_varlen_bwd_fake(
        grad_out: torch.Tensor,
        x: torch.Tensor,
        qkv: torch.Tensor,
        attn_out: torch.Tensor,
        lse: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        offsets: Optional[torch.Tensor],
        max_seqlen_q: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        needs_dx: bool = True,
        needs_dw_qkv: bool = True,
        needs_dw_out: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = torch.empty_like(x) if needs_dx else x.new_empty((0,))
        dw_qkv = torch.empty_like(w_qkv) if needs_dw_qkv else x.new_empty((0,))
        dw_out = torch.empty_like(w_out) if needs_dw_out else x.new_empty((0,))
        return dx, dw_qkv, dw_out

    @_register_fake("varlen_transformer::fused_cross_gqa_varlen_fwd")
    def _fused_cross_gqa_varlen_fwd_fake(
        x: torch.Tensor,
        context: torch.Tensor,
        w_q: torch.Tensor,
        w_kv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        save_for_bwd: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_lq = x.shape[0]
        total_lk = context.shape[0]
        out = x.new_empty((total_lq, h_q * 64))
        if not save_for_bwd:
            empty = x.new_empty((0,))
            return out, empty, empty, empty, x.new_empty((0,), dtype=torch.float32)
        q = x.new_empty((total_lq, h_q, 64))
        kv = context.new_empty((total_lk, 2 * h_kv, 64))
        attn_out = x.new_empty((total_lq, h_q, 64))
        lse = x.new_empty((h_q, total_lq), dtype=torch.float32)
        return out, q, kv, attn_out, lse

    @_register_fake("varlen_transformer::fused_cross_gqa_varlen_bwd")
    def _fused_cross_gqa_varlen_bwd_fake(
        grad_out: torch.Tensor,
        x: torch.Tensor,
        context: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_out: torch.Tensor,
        lse: torch.Tensor,
        w_q: torch.Tensor,
        w_kv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        offsets: Optional[torch.Tensor],
        max_seqlen_k: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        needs_dx: bool = True,
        needs_dctx: bool = True,
        needs_dw_q: bool = True,
        needs_dw_kv: bool = True,
        needs_dw_out: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = torch.empty_like(x) if needs_dx else x.new_empty((0,))
        d_ctx = torch.empty_like(context) if needs_dctx else x.new_empty((0,))
        dw_q = torch.empty_like(w_q) if needs_dw_q else x.new_empty((0,))
        dw_kv = torch.empty_like(w_kv) if needs_dw_kv else x.new_empty((0,))
        dw_out = torch.empty_like(w_out) if needs_dw_out else x.new_empty((0,))
        return dx, d_ctx, dw_q, dw_kv, dw_out

    @_register_fake("varlen_transformer::fused_gqa_varlen_decode")
    def _fused_gqa_varlen_decode_fake(
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        use_splitk: Optional[bool] = None,
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], h_q * 64))


# =========================================================================================
# Direct Eager Wrappers around varlen_transformer._C (Pure THD + Paged KV)
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
    """Eager wrapper for varlen forward attention + optional fused RoPE (THD layout)."""
    _require_cuda_ext()
    return _C_ext.fused_mha_varlen_fwd(
        _ensure_strided_3d(q),
        _ensure_strided_3d(k),
        _ensure_strided_3d(v),
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
    """Eager wrapper for varlen backward attention + optional fused RoPE (THD layout)."""
    _require_cuda_ext()
    return _C_ext.fused_mha_varlen_bwd(
        _ensure_strided_3d(do_grad),
        _ensure_strided_3d(q),
        _ensure_strided_3d(k),
        _ensure_strided_3d(v),
        _ensure_strided_3d(o),
        lse.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        _contiguous_or_none(offsets),
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        bool(is_causal),
    )


def prefill_to_paged_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
) -> None:
    """Copy packed THD K/V tensors into Paged KV pool and populate cache_seqlens."""
    _require_cuda_ext()
    _C_ext.prefill_to_paged_kv(
        _ensure_strided_3d(k),
        _ensure_strided_3d(v),
        cu_seqlens_k.contiguous(),
        k_cache,
        v_cache,
        block_table.contiguous(),
        cache_seqlens,
        int(max_seqlen_k),
    )


def paged_kv_append(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> None:
    """Append 1 token per sequence into Paged KV pool and increment cache_seqlens."""
    _require_cuda_ext()
    _C_ext.paged_kv_append(
        _ensure_strided_3d(new_k),
        _ensure_strided_3d(new_v),
        k_cache,
        v_cache,
        block_table.contiguous(),
        cache_seqlens,
    )


def fused_mha_varlen_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    use_splitk: Optional[bool] = None,
) -> torch.Tensor:
    """Eager wrapper for Paged Varlen Flash Decoding (Lq = 1, (B, H_q, 64))."""
    _require_cuda_ext()
    return _C_ext.fused_mha_varlen_decode(
        _ensure_strided_3d(q),
        k_cache,
        v_cache,
        block_table.contiguous(),
        cache_seqlens.contiguous(),
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        _contiguous_or_none(offsets),
        None if use_splitk is None else bool(use_splitk),
    )


# =========================================================================================
# PyTorch Autograd Functions & High-Level APIs (Pure Eager)
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
        o, lse = _C_ext.fused_mha_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, cos, sin, is_causal, offsets
        )
        if any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, cos, sin, offsets)
            ctx.max_seqlen_k = int(max_seqlen_k)
            ctx.is_causal = bool(is_causal)
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        if not any(ctx.needs_input_grad[:3]):
            return (None,) * 11
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, cos, sin, offsets = ctx.saved_tensors
        dq, dk, dv = _C_ext.fused_mha_varlen_bwd(
            _ensure_strided_3d(grad_o),
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
        return (
            dq if ctx.needs_input_grad[0] else None,
            dk if ctx.needs_input_grad[1] else None,
            dv if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _VarlenFusedGQAFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        is_causal: bool,
        offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        save_for_bwd = any(ctx.needs_input_grad[:3])
        out, qkv, attn_out, lse = _C_ext.fused_gqa_varlen_fwd(
            x,
            w_qkv,
            w_out,
            cu_seqlens_q,
            int(max_seqlen_q),
            int(h_q),
            int(h_kv),
            cos,
            sin,
            bool(is_causal),
            offsets,
            save_for_bwd,
        )
        if save_for_bwd:
            ctx.save_for_backward(x, w_qkv, w_out, qkv, attn_out, lse, cu_seqlens_q, cos, sin, offsets)
            ctx.max_seqlen_q = int(max_seqlen_q)
            ctx.h_q = int(h_q)
            ctx.h_kv = int(h_kv)
            ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        needs_dx, needs_dw_qkv, needs_dw_out = ctx.needs_input_grad[:3]
        if not (needs_dx or needs_dw_qkv or needs_dw_out):
            return (None,) * 11
        x, w_qkv, w_out, qkv, attn_out, lse, cu_seqlens_q, cos, sin, offsets = ctx.saved_tensors
        dx, dw_qkv, dw_out = _C_ext.fused_gqa_varlen_bwd(
            grad_out.contiguous(),
            x,
            qkv,
            attn_out,
            lse,
            w_qkv,
            w_out,
            cu_seqlens_q,
            offsets,
            ctx.max_seqlen_q,
            ctx.h_q,
            ctx.h_kv,
            cos,
            sin,
            ctx.is_causal,
            bool(needs_dx),
            bool(needs_dw_qkv),
            bool(needs_dw_out),
        )
        return (
            dx if needs_dx else None,
            dw_qkv if needs_dw_qkv else None,
            dw_out if needs_dw_out else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _VarlenFusedCrossGQAFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        context: torch.Tensor,
        w_q: torch.Tensor,
        w_kv: torch.Tensor,
        w_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        h_q: int,
        h_kv: int,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        save_for_bwd = any(ctx.needs_input_grad[:5])
        out, q, kv, attn_out, lse = _C_ext.fused_cross_gqa_varlen_fwd(
            x,
            context,
            w_q,
            w_kv,
            w_out,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(h_q),
            int(h_kv),
            cos,
            sin,
            offsets,
            save_for_bwd,
        )
        if save_for_bwd:
            ctx.save_for_backward(
                x, context, w_q, w_kv, w_out, q, kv, attn_out, lse, cu_seqlens_q, cu_seqlens_k, cos, sin, offsets
            )
            ctx.max_seqlen_k = int(max_seqlen_k)
            ctx.h_q = int(h_q)
            ctx.h_kv = int(h_kv)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        needs_dx, needs_dctx, needs_dw_q, needs_dw_kv, needs_dw_out = ctx.needs_input_grad[:5]
        if not (needs_dx or needs_dctx or needs_dw_q or needs_dw_kv or needs_dw_out):
            return (None,) * 14
        (
            x, context, w_q, w_kv, w_out, q, kv, attn_out, lse, cu_seqlens_q, cu_seqlens_k, cos, sin, offsets
        ) = ctx.saved_tensors
        dx, d_ctx, dw_q, dw_kv, dw_out = _C_ext.fused_cross_gqa_varlen_bwd(
            grad_out.contiguous(),
            x,
            context,
            q,
            kv,
            attn_out,
            lse,
            w_q,
            w_kv,
            w_out,
            cu_seqlens_q,
            cu_seqlens_k,
            offsets,
            ctx.max_seqlen_k,
            ctx.h_q,
            ctx.h_kv,
            cos,
            sin,
            bool(needs_dx),
            bool(needs_dctx),
            bool(needs_dw_q),
            bool(needs_dw_kv),
            bool(needs_dw_out),
        )
        return (
            dx if needs_dx else None,
            d_ctx if needs_dctx else None,
            dw_q if needs_dw_q else None,
            dw_kv if needs_dw_kv else None,
            dw_out if needs_dw_out else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    Autograd-capable variable-length FlashAttention with fused RoPE in pure THD layout.

    Args:
        q: (Total_Lq, H_q, 64) float16 (supports 8-elem aligned strided slices)
        k: (Total_Lk, H_kv, 64) float16
        v: (Total_Lk, H_kv, 64) float16
        cu_seqlens_q: (B + 1,) int32
        cu_seqlens_k: (B + 1,) int32, defaults to cu_seqlens_q
        max_seqlen_q: int, inferred from cu_seqlens_q if None
        max_seqlen_k: int, inferred from cu_seqlens_k if None
        cos: Optional (L_max, 64) float16
        sin: Optional (L_max, 64) float16
        is_causal: bool causal mask flag
        offsets: Optional (B,) int32 RoPE position offsets per sequence
    Returns:
        o: (Total_Lq, H_q, 64) float16
    """
    _require_cuda_ext()

    cu_q = cu_seqlens_q.contiguous()
    cu_k = cu_q if cu_seqlens_k is None else cu_seqlens_k.contiguous()

    if max_seqlen_q is None:
        max_seqlen_q = _infer_max_seqlen(cu_q)
    if max_seqlen_k is None:
        max_seqlen_k = max_seqlen_q if cu_k is cu_q else _infer_max_seqlen(cu_k)

    return _VarlenFlashAttentionFunc.apply(
        _ensure_strided_3d(q),
        _ensure_strided_3d(k),
        _ensure_strided_3d(v),
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
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    use_splitk: Optional[bool] = None,
) -> torch.Tensor:
    """
    Paged variable-length Flash Decoding (Lq = 1) with fused RoPE on raw KV pool.

    Args:
        q: (B, H_q, 64) float16 (supports 8-elem aligned strided slices)
        k_cache: (num_pages, 16, H_kv, 64) float16
        v_cache: (num_pages, 16, H_kv, 64) float16
        block_table: (B, max_pages_per_seq) int32
        cache_seqlens: (B,) int32
        max_seqlen_k: int, inferred from cache_seqlens if None
        cos, sin: Optional (L_max, 64) float16
        offsets: Optional (B,) int32 base RoPE offsets per sequence
        use_splitk: Optional bool to force Split-K (True) or Direct FP16 (False)
    Returns:
        o: (B, H_q, 64) float16
    """
    _require_cuda_ext()
    bt = block_table.contiguous()
    seqlens = cache_seqlens.contiguous()
    if max_seqlen_k is None:
        max_seqlen_k = int(seqlens.max().item()) if seqlens.numel() > 0 else 0

    return _C_ext.fused_mha_varlen_decode(
        _ensure_strided_3d(q),
        k_cache,
        v_cache,
        bt,
        seqlens,
        int(max_seqlen_k),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        _contiguous_or_none(offsets),
        None if use_splitk is None else bool(use_splitk),
    )


def varlen_gqa_attention(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    h_q: Optional[int] = None,
    h_kv: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused bias-free Self-Attention Linear + FlashAttention + OutProj (eager autograd)."""
    _require_cuda_ext()
    if h_q is None:
        if x.shape[1] % 64 != 0:
            raise ValueError(f"x.shape[1] ({x.shape[1]}) must be divisible by 64")
        h_q = x.shape[1] // 64
    if h_kv is None:
        h_kv = (w_qkv.shape[0] // 64 - h_q) // 2

    cu_q = cu_seqlens_q.contiguous()
    if max_seqlen_q is None:
        max_seqlen_q = _infer_max_seqlen(cu_q)

    return _VarlenFusedGQAFunc.apply(
        x if x.is_contiguous() else x.contiguous(),
        w_qkv,
        w_out,
        cu_q,
        int(max_seqlen_q),
        int(h_q),
        int(h_kv),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        bool(is_causal),
        _contiguous_or_none(offsets),
    )


def varlen_gqa_prefill(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    h_q: Optional[int] = None,
    h_kv: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused bias-free Self-Attention Linear + Paged KV Prefill + FlashAttention + OutProj."""
    _require_cuda_ext()
    if h_q is None:
        h_q = x.shape[1] // 64
    if h_kv is None:
        h_kv = k_cache.shape[2]

    cu_q = cu_seqlens_q.contiguous()
    if max_seqlen_q is None:
        max_seqlen_q = _infer_max_seqlen(cu_q)

    return _C_ext.fused_gqa_varlen_prefill(
        x if x.is_contiguous() else x.contiguous(),
        w_qkv,
        w_out,
        cu_q,
        int(max_seqlen_q),
        int(h_q),
        int(h_kv),
        k_cache,
        v_cache,
        block_table.contiguous(),
        cache_seqlens,
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        bool(is_causal),
        _contiguous_or_none(offsets),
    )


def varlen_cross_gqa_attention(
    x: torch.Tensor,
    context: torch.Tensor,
    w_q: torch.Tensor,
    w_kv: torch.Tensor,
    w_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    h_q: Optional[int] = None,
    h_kv: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused bias-free Cross-Attention Linear + FlashAttention + OutProj (eager autograd)."""
    _require_cuda_ext()
    if h_q is None:
        h_q = x.shape[1] // 64
    if h_kv is None:
        h_kv = w_kv.shape[0] // (2 * 64)

    cu_q = cu_seqlens_q.contiguous()
    cu_k = cu_seqlens_k.contiguous()
    if max_seqlen_q is None:
        max_seqlen_q = _infer_max_seqlen(cu_q)
    if max_seqlen_k is None:
        max_seqlen_k = _infer_max_seqlen(cu_k)

    return _VarlenFusedCrossGQAFunc.apply(
        x if x.is_contiguous() else x.contiguous(),
        context if context.is_contiguous() else context.contiguous(),
        w_q,
        w_kv,
        w_out,
        cu_q,
        cu_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        int(h_q),
        int(h_kv),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        _contiguous_or_none(offsets),
    )


def varlen_gqa_decode(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: Optional[int] = None,
    h_q: Optional[int] = None,
    h_kv: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    use_splitk: Optional[bool] = None,
) -> torch.Tensor:
    """Fused bias-free Self-Attention Decode step (QKV proj -> Paged KV Append -> Decode -> OutProj)."""
    _require_cuda_ext()
    if h_q is None:
        h_q = x.shape[1] // 64
    if h_kv is None:
        h_kv = k_cache.shape[2]

    if max_seqlen_k is None:
        max_seqlen_k = int(cache_seqlens.max().item()) + 1 if cache_seqlens.numel() > 0 else 0

    return _C_ext.fused_gqa_varlen_decode(
        x if x.is_contiguous() else x.contiguous(),
        w_qkv,
        w_out,
        k_cache,
        v_cache,
        block_table.contiguous(),
        cache_seqlens,
        int(max_seqlen_k),
        int(h_q),
        int(h_kv),
        _contiguous_or_none(cos),
        _contiguous_or_none(sin),
        _contiguous_or_none(offsets),
        None if use_splitk is None else bool(use_splitk),
    )


__all__ = [
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
]
