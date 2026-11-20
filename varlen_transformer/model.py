"""RotaryPositionalEmbedding, VarlenKVCache, and FlashGQA modules (kernel, varlen ref, batched SDPA)."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from varlen_transformer.ops import varlen_flash_attention, varlen_flash_decode


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_reshaped = x.view(*x.shape[:-1], -1, 2)
    x_rotated = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)
    return x_rotated.flatten(-2)


def _repeat_kv(x: torch.Tensor, rep: int) -> torch.Tensor:
    if rep == 1:
        return x
    # x: (..., H_kv, L, D) -> (..., H_q, L, D)
    shape = x.shape
    return (
        x.unsqueeze(-3)
        .expand(*shape[:-2], rep, shape[-2], shape[-1])
        .reshape(*shape[:-3], shape[-3] * rep, shape[-2], shape[-1])
    )


class RotaryPositionalEmbedding(nn.Module):
    """Non-interleaved (or interleaved for batched reference) Rotary Positional Embedding."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        interleaved: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.interleaved = interleaved

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._cos_fp16_cache: Optional[torch.Tensor] = None
        self._sin_fp16_cache: Optional[torch.Tensor] = None

        if device is not None:
            self._build_cache(max_seq_len, device=device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len

        target_device = device if device is not None else self.inv_freq.device
        inv_freq = self.inv_freq.to(device=target_device)
        t = torch.arange(seq_len, device=target_device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)

        if not self.interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.repeat_interleave(freqs, 2, dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._seq_len_cached = seq_len
        self._cos_fp16_cache = None
        self._sin_fp16_cache = None

    def get_cos_sin(
        self,
        seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous (L_max, dim) cos/sin tables suitable for CUDA kernels."""
        if self.interleaved:
            raise ValueError("CUDA fused RoPE kernels require interleaved=False")

        target_len = self._seq_len_cached if seq_len is None else int(seq_len)
        if target_len == 0:
            target_len = self.max_seq_len

        target_device = device if device is not None else self.inv_freq.device
        if (
            self.cos_cached is None
            or target_len > self._seq_len_cached
            or self.cos_cached.device != target_device
        ):
            self._build_cache(max(target_len, self.max_seq_len), device=target_device)

        if dtype == torch.float16:
            if (
                self._cos_fp16_cache is None
                or self._cos_fp16_cache.device != target_device
                or self._cos_fp16_cache.shape[0] != self.cos_cached.shape[0]
            ):
                self._cos_fp16_cache = self.cos_cached.to(dtype=torch.float16).contiguous()
                self._sin_fp16_cache = self.sin_cached.to(dtype=torch.float16).contiguous()
            return self._cos_fp16_cache, self._sin_fp16_cache

        return (
            self.cos_cached.to(dtype=dtype).contiguous(),
            self.sin_cached.to(dtype=dtype).contiguous(),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[-2]

        needed = offset + seq_len
        if (
            self.cos_cached is None
            or needed > self._seq_len_cached
            or self.cos_cached.device != x.device
        ):
            self._build_cache(max(needed, self.max_seq_len), device=x.device)

        cos = self.cos_cached[offset : offset + seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype=x.dtype)

        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        if not self.interleaved:
            return (x * cos) + (_rotate_half(x) * sin)
        return (x * cos) + (_rotate_interleaved(x) * sin)


class VarlenKVCache:
    """
    Variable-length packed raw KV cache for Flash Decoding (Method 2: stores raw unrotated keys).

    Shapes:
        k_cache: (H_kv, Total_Lk, head_dim) [FP16] (raw unrotated keys)
        v_cache: (H_kv, Total_Lk, head_dim) [FP16]
        cu_seqlens_k: (B + 1,) [Int32]
        offsets: Optional (B,) [Int32] base RoPE offset per sequence
    """

    def __init__(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: Optional[int] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:
        self.k_cache = k_cache.contiguous()
        self.v_cache = v_cache.contiguous()
        self.cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32).contiguous()
        self.offsets = offsets.to(dtype=torch.int32).contiguous() if offsets is not None else None
        if max_seqlen_k is None:
            seqlens = self.cu_seqlens_k[1:] - self.cu_seqlens_k[:-1]
            self.max_seqlen_k = int(seqlens.max().item()) if seqlens.numel() > 0 else 0
        else:
            self.max_seqlen_k = int(max_seqlen_k)

    @property
    def batch_size(self) -> int:
        return max(0, int(self.cu_seqlens_k.shape[0]) - 1)

    @classmethod
    def from_packed(
        cls,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: Optional[int] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> "VarlenKVCache":
        """Create VarlenKVCache directly from prefill raw packed (H_kv, Total_Lk, D) tensors."""
        return cls(
            k_cache=k.detach().contiguous(),
            v_cache=v.detach().contiguous(),
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            offsets=offsets,
        )

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Append one new decode token per sequence in the batch in a single vectorized step.

        Args:
            new_k: (H_kv, B, head_dim) raw unrotated key for the current token
            new_v: (H_kv, B, head_dim) value for the current token
        """
        B = self.batch_size
        if new_k.dim() != 3 or new_v.dim() != 3 or new_k.shape[1] != B or new_v.shape[1] != B:
            raise ValueError(
                f"Expected new_k/new_v shape (H_kv, {B}, D), got {tuple(new_k.shape)} and {tuple(new_v.shape)}"
            )

        H_kv, total_old, head_dim = self.k_cache.shape
        device = self.k_cache.device
        cu_old = self.cu_seqlens_k

        cu_new = cu_old + torch.arange(B + 1, device=device, dtype=cu_old.dtype)
        total_new = total_old + B

        k_out = self.k_cache.new_empty((H_kv, total_new, head_dim))
        v_out = self.v_cache.new_empty((H_kv, total_new, head_dim))

        if total_old > 0:
            seqlens_old = (cu_old[1:] - cu_old[:-1]).to(dtype=torch.int64)
            seq_ids = torch.arange(B, device=device, dtype=torch.int64).repeat_interleave(seqlens_old)
            old_dst_idx = torch.arange(total_old, device=device, dtype=torch.int64) + seq_ids
            k_out[:, old_dst_idx, :] = self.k_cache
            v_out[:, old_dst_idx, :] = self.v_cache

        if B > 0:
            new_dst_idx = cu_new[1:].to(dtype=torch.int64) - 1
            k_out[:, new_dst_idx, :] = new_k.to(dtype=self.k_cache.dtype)
            v_out[:, new_dst_idx, :] = new_v.to(dtype=self.v_cache.dtype)

        self.k_cache = k_out
        self.v_cache = v_out
        self.cu_seqlens_k = cu_new
        if B > 0:
            self.max_seqlen_k += 1


try:
    from torch.utils._pytree import register_pytree_node

    def _varlen_kv_cache_flatten(cache: "VarlenKVCache"):
        flat_tensors = [cache.k_cache, cache.v_cache, cache.cu_seqlens_k]
        if cache.offsets is not None:
            flat_tensors.append(cache.offsets)
        context = (cache.max_seqlen_k, cache.offsets is not None)
        return flat_tensors, context

    def _varlen_kv_cache_unflatten(flat_tensors, context):
        max_seqlen_k, has_offsets = context
        k, v, cu = flat_tensors[0], flat_tensors[1], flat_tensors[2]
        offsets = flat_tensors[3] if has_offsets else None
        return VarlenKVCache(k, v, cu, max_seqlen_k=max_seqlen_k, offsets=offsets)

    register_pytree_node(VarlenKVCache, _varlen_kv_cache_flatten, _varlen_kv_cache_unflatten)
except Exception:
    pass


# =========================================================================================
# Pure-PyTorch Varlen Reference Functions (for Correctness Verification)
# =========================================================================================

def varlen_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch autograd-capable reference for packed varlen GQA + optional fused RoPE.

    Args:
        q: (H_q, Total_Lq, D)
        k: (H_kv, Total_Lk, D)
        v: (H_kv, Total_Lk, D)
        cu_seqlens_q: (B + 1,) int32
        cu_seqlens_k: Optional (B + 1,) int32
        cos, sin: Optional (L_max, D) non-interleaved RoPE tables
        is_causal: bool upper-triangular mask (col > row)
        offsets: Optional (B,) int32 RoPE offset per sequence
    Returns:
        o: (H_q, Total_Lq, D) same dtype as q
        lse: (H_q, Total_Lq) float32
    """
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    H_q, total_lq, head_dim = q.shape
    H_kv = k.shape[0]
    rep = H_q // H_kv
    batch_size = cu_seqlens_q.shape[0] - 1
    scale = 1.0 / math.sqrt(head_dim)

    if total_lq == 0 or batch_size <= 0:
        return torch.zeros_like(q), q.new_empty((H_q, total_lq), dtype=torch.float32)

    out_chunks = []
    lse_chunks = []

    for i in range(batch_size):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        lq_i = q_end - q_start
        lk_i = k_end - k_start

        if lq_i == 0:
            continue

        if lk_i == 0:
            out_chunks.append(q.new_zeros((H_q, lq_i, head_dim)))
            lse_chunks.append(
                torch.full((H_q, lq_i), float("-inf"), dtype=torch.float32, device=q.device)
            )
            continue

        q_i = q[:, q_start:q_end, :]
        k_i = k[:, k_start:k_end, :]
        v_i = v[:, k_start:k_end, :]

        if cos is not None and sin is not None:
            off_i = int(offsets[i].item()) if offsets is not None else 0
            cos_q = cos[off_i : off_i + lq_i].to(dtype=q_i.dtype).unsqueeze(0)
            sin_q = sin[off_i : off_i + lq_i].to(dtype=q_i.dtype).unsqueeze(0)
            cos_k = cos[off_i : off_i + lk_i].to(dtype=k_i.dtype).unsqueeze(0)
            sin_k = sin[off_i : off_i + lk_i].to(dtype=k_i.dtype).unsqueeze(0)
            q_i = (q_i * cos_q) + (_rotate_half(q_i) * sin_q)
            k_i = (k_i * cos_k) + (_rotate_half(k_i) * sin_k)

        k_exp = _repeat_kv(k_i, rep)
        v_exp = _repeat_kv(v_i, rep)

        scores = torch.matmul(q_i.float(), k_exp.float().transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.triu(
                torch.ones((lq_i, lk_i), device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        lse_i = torch.logsumexp(scores, dim=-1)
        valid_rows = (scores > float("-inf")).any(dim=-1, keepdim=True)
        safe_scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
        probs = torch.where(valid_rows, torch.softmax(safe_scores, dim=-1), torch.zeros_like(scores))

        out_i = torch.matmul(probs, v_exp.float()).to(dtype=q.dtype)
        out_chunks.append(out_i)
        lse_chunks.append(lse_i)

    if not out_chunks:
        return torch.zeros_like(q), q.new_empty((H_q, total_lq), dtype=torch.float32)

    return torch.cat(out_chunks, dim=1), torch.cat(lse_chunks, dim=1)


def varlen_decode_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure-PyTorch reference for Flash Decoding (Lq = 1) on raw unrotated k_cache (Method 2).

    Args:
        q: (H_q, B, D)
        k_cache: (H_kv, Total_Lk, D)
        v_cache: (H_kv, Total_Lk, D)
        cu_seqlens_k: (B + 1,) int32
        cos, sin: Optional (L_max, D) RoPE tables
        offsets: Optional (B,) int32 base RoPE offsets
    Returns:
        o: (H_q, B, D)
    """
    H_q, B, head_dim = q.shape
    H_kv = k_cache.shape[0]
    rep = H_q // H_kv
    scale = 1.0 / math.sqrt(head_dim)

    if B == 0:
        return torch.zeros_like(q)

    out_chunks = []
    for i in range(B):
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        lk_i = k_end - k_start

        if lk_i <= 0:
            out_chunks.append(q.new_zeros((H_q, 1, head_dim)))
            continue

        q_i = q[:, i : i + 1, :]
        k_i = k_cache[:, k_start:k_end, :]
        v_i = v_cache[:, k_start:k_end, :]

        if cos is not None and sin is not None:
            off_i = int(offsets[i].item()) if offsets is not None else 0
            cos_q = cos[off_i + lk_i - 1].to(dtype=q_i.dtype).view(1, 1, head_dim)
            sin_q = sin[off_i + lk_i - 1].to(dtype=q_i.dtype).view(1, 1, head_dim)
            cos_k = cos[off_i : off_i + lk_i].to(dtype=k_i.dtype).unsqueeze(0)
            sin_k = sin[off_i : off_i + lk_i].to(dtype=k_i.dtype).unsqueeze(0)
            q_i = (q_i * cos_q) + (_rotate_half(q_i) * sin_q)
            k_i = (k_i * cos_k) + (_rotate_half(k_i) * sin_k)

        k_exp = _repeat_kv(k_i, rep)
        v_exp = _repeat_kv(v_i, rep)

        scores = torch.matmul(q_i.float(), k_exp.float().transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out_i = torch.matmul(probs, v_exp.float()).to(dtype=q.dtype)
        out_chunks.append(out_i)

    return torch.cat(out_chunks, dim=1)


# =========================================================================================
# GQA Modules (Kernel / Varlen Correctness Baseline / Batched SDPA Speed Baseline)
# =========================================================================================

class _BaseFlashGQA(nn.Module):
    """Shared weight initialization and QKV projection helpers across GQA backends."""

    def __init__(
        self,
        embed_dim: int,
        num_heads_q: int,
        num_heads_kv: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        dropout: float = 0.0,
        bias: bool = False,
        is_cross_attention: bool = False,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads_q != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads_q ({num_heads_q})")
        if num_heads_q % num_heads_kv != 0:
            raise ValueError(f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})")

        self.embed_dim = embed_dim
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = embed_dim // num_heads_q
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_cross_attention = is_cross_attention
        self.rep = num_heads_q // num_heads_kv
        self.rope = None if is_cross_attention else rope
        self.dropout_p = dropout

        if not is_cross_attention:
            total_dim = (num_heads_q + 2 * num_heads_kv) * self.head_dim
            self.qkv_proj = nn.Linear(embed_dim, total_dim, bias=bias)
        else:
            ctx_dim = context_dim or embed_dim
            self.q_proj = nn.Linear(embed_dim, num_heads_q * self.head_dim, bias=bias)
            self.kv_proj = nn.Linear(ctx_dim, 2 * num_heads_kv * self.head_dim, bias=bias)

        self.out_proj = nn.Linear(num_heads_q * self.head_dim, embed_dim, bias=bias)

    @staticmethod
    def _add_causal_mask(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        device = mask.device
        if mask.dtype == torch.bool:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
            return causal_mask.unsqueeze(0).unsqueeze(0) & mask
        causal_mask = torch.full(
            (seq_len, seq_len),
            torch.finfo(mask.dtype).min,
            device=device,
            dtype=mask.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        return causal_mask.unsqueeze(0).unsqueeze(0) + mask

    def _project_qkv_packed(
        self,
        x: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        total_lq = x.shape[0]
        if not self.is_cross_attention:
            qkv = self.qkv_proj(x).view(total_lq, self.num_heads_q + 2 * self.num_heads_kv, self.head_dim)
            q, k, v = qkv.split([self.num_heads_q, self.num_heads_kv, self.num_heads_kv], dim=1)
            cu_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
            eff_cross = False
        else:
            if context is None or cu_seqlens_k is None:
                raise ValueError("Cross-attention varlen requires both context and cu_seqlens_k")
            total_lk = context.shape[0]
            q = self.q_proj(x).view(total_lq, self.num_heads_q, self.head_dim)
            kv = self.kv_proj(context).view(total_lk, 2 * self.num_heads_kv, self.head_dim)
            k, v = kv.split([self.num_heads_kv, self.num_heads_kv], dim=1)
            cu_k = cu_seqlens_k
            eff_cross = True

        # Transpose (Total_L, H, D) -> (H, Total_L, D) contiguous cho CUDA kernel / ref
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        return q, k, v, cu_k, eff_cross

    def _project_qkv_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_cross_attention:
            raise NotImplementedError("Varlen decode step supports self-attention only")
        B = x.shape[0]
        qkv = self.qkv_proj(x).view(B, self.num_heads_q + 2 * self.num_heads_kv, self.head_dim)
        q, k, v = qkv.split([self.num_heads_q, self.num_heads_kv, self.num_heads_kv], dim=1)
        return q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()

    def _resolve_rope_tables(
        self,
        seq_len_hint: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.rope is None or self.is_cross_attention:
            return None, None
        return self.rope.get_cos_sin(seq_len=seq_len_hint, device=device, dtype=dtype)


class VarlenFlashGQA(_BaseFlashGQA):
    """
    Custom SM75 CUDA/CuTe Varlen FlashAttention + Fused RoPE + Split-K Flash Decoding module.
    Supports both training/prefill (`decode=False`, packed `(Total_Lq, embed_dim)`)
    and step decoding (`decode=True`, `(B, embed_dim)` with `VarlenKVCache`), compatible with `torch.compile`.
    """

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        is_causal: bool = False,
        offsets: Optional[torch.Tensor] = None,
        kv_cache: Optional[VarlenKVCache] = None,
        use_cache: bool = False,
        decode: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, VarlenKVCache]]:
        if self.head_dim != 64:
            raise ValueError(f"CUDA kernels currently require head_dim == 64, got {self.head_dim}")

        if decode:
            if kv_cache is None:
                raise ValueError("kv_cache must be provided when decode=True")
            q, new_k, new_v = self._project_qkv_decode(x)
            kv_cache.append(new_k, new_v)
            rope_offsets = offsets if offsets is not None else kv_cache.offsets
            seq_hint = kv_cache.max_seqlen_k if rope_offsets is None else None
            cos, sin = self._resolve_rope_tables(seq_hint, x.device, x.dtype)

            attn_out = varlen_flash_decode(
                q=q,
                k_cache=kv_cache.k_cache,
                v_cache=kv_cache.v_cache,
                cu_seqlens_k=kv_cache.cu_seqlens_k,
                max_seqlen_k=kv_cache.max_seqlen_k,
                cos=cos,
                sin=sin,
                offsets=rope_offsets,
            )  # (H_q, B, D)
            out_flat = attn_out.transpose(0, 1).contiguous().view(x.shape[0], -1)
            final_out = self.out_proj(out_flat)
            if use_cache:
                return final_out, kv_cache
            return final_out

        if cu_seqlens_q is None:
            raise ValueError("cu_seqlens_q must be provided for varlen prefill/training")

        q, k, v, cu_k, eff_cross = self._project_qkv_packed(
            x=x,
            cu_seqlens_q=cu_seqlens_q,
            context=context,
            cu_seqlens_k=cu_seqlens_k,
        )
        eff_causal = False if eff_cross else is_causal

        seq_hint = None
        if offsets is None and max_seqlen_q is not None:
            eff_max_k = max_seqlen_k if max_seqlen_k is not None else max_seqlen_q
            seq_hint = max(int(max_seqlen_q), int(eff_max_k))
        cos, sin = self._resolve_rope_tables(seq_hint, x.device, x.dtype)

        attn_out = varlen_flash_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cos=cos,
            sin=sin,
            is_causal=eff_causal,
            offsets=offsets,
        )  # (H_q, Total_Lq, D)

        out_flat = attn_out.transpose(0, 1).contiguous().view(x.shape[0], -1)
        final_out = self.out_proj(out_flat)

        if use_cache:
            new_cache = VarlenKVCache.from_packed(
                k=k,
                v=v,
                cu_seqlens_k=cu_k,
                max_seqlen_k=max_seqlen_k if max_seqlen_k is not None else max_seqlen_q,
                offsets=offsets,
            )
            return final_out, new_cache
        return final_out


class VarlenFlashGQABaselineCorrectness(_BaseFlashGQA):
    """
    Pure-PyTorch variable-length GQA + RoPE reference baseline for correctness testing.
    Executes exact mathematical per-sequence attention on packed inputs (and VarlenKVCache decode).
    """

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        is_causal: bool = False,
        offsets: Optional[torch.Tensor] = None,
        kv_cache: Optional[VarlenKVCache] = None,
        use_cache: bool = False,
        decode: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, VarlenKVCache]]:
        del max_seqlen_q, max_seqlen_k
        if decode:
            if kv_cache is None:
                raise ValueError("kv_cache must be provided when decode=True")
            q, new_k, new_v = self._project_qkv_decode(x)
            kv_cache.append(new_k, new_v)
            rope_offsets = offsets if offsets is not None else kv_cache.offsets
            cos, sin = self._resolve_rope_tables(None, x.device, x.dtype)

            attn_out = varlen_decode_ref(
                q=q,
                k_cache=kv_cache.k_cache,
                v_cache=kv_cache.v_cache,
                cu_seqlens_k=kv_cache.cu_seqlens_k,
                cos=cos,
                sin=sin,
                offsets=rope_offsets,
            )
            out_flat = attn_out.transpose(0, 1).contiguous().view(x.shape[0], -1)
            final_out = self.out_proj(out_flat)
            if use_cache:
                return final_out, kv_cache
            return final_out

        if cu_seqlens_q is None:
            raise ValueError("cu_seqlens_q must be provided for varlen correctness baseline")

        q, k, v, cu_k, eff_cross = self._project_qkv_packed(
            x=x,
            cu_seqlens_q=cu_seqlens_q,
            context=context,
            cu_seqlens_k=cu_seqlens_k,
        )
        eff_causal = False if eff_cross else is_causal
        cos, sin = self._resolve_rope_tables(None, x.device, x.dtype)

        attn_out, _ = varlen_attention_ref(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_k,
            cos=cos,
            sin=sin,
            is_causal=eff_causal,
            offsets=offsets,
        )
        out_flat = attn_out.transpose(0, 1).contiguous().view(x.shape[0], -1)
        final_out = self.out_proj(out_flat)

        if use_cache:
            new_cache = VarlenKVCache.from_packed(
                k=k,
                v=v,
                cu_seqlens_k=cu_k,
                offsets=offsets,
            )
            return final_out, new_cache
        return final_out


class BatchedFlashGQABaselineSpeed(_BaseFlashGQA):
    """
    Fast padded batched GQA + RoPE baseline backed by `F.scaled_dot_product_attention`.
    Used strictly as the speed baseline for batched PyTorch training and inference benchmarks.
    """

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, Lq, _ = x.shape

        if not self.is_cross_attention:
            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, Lq, self.num_heads_q + 2 * self.num_heads_kv, self.head_dim).transpose(1, 2)
            q, k, v = qkv.split([self.num_heads_q, self.num_heads_kv, self.num_heads_kv], dim=1)

            if self.rope is not None:
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)

            if use_cache and kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            current_kv_cache = (k.detach(), v.detach()) if use_cache else None
        else:
            is_causal = False
            if context is not None and context.dim() == 4:
                context = context.flatten(1, 2)

            q = self.q_proj(x).view(B, Lq, self.num_heads_q, self.head_dim).transpose(1, 2)
            if use_cache and kv_cache is not None:
                k, v = kv_cache
            else:
                if context is None:
                    raise ValueError("context must be provided for batched cross-attention")
                Lk = context.shape[1]
                kv = self.kv_proj(context).view(B, Lk, 2 * self.num_heads_kv, self.head_dim).transpose(1, 2)
                k, v = kv.split([self.num_heads_kv, self.num_heads_kv], dim=1)
            current_kv_cache = (k.detach(), v.detach()) if use_cache else None

        k_exp = _repeat_kv(k, self.rep)
        v_exp = _repeat_kv(v, self.rep)

        actual_is_causal = is_causal
        if kv_cache is not None:
            actual_is_causal = False
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(B, 1, 1, -1)
            if is_causal and kv_cache is None:
                attn_mask = self._add_causal_mask(attn_mask, Lq)
            actual_is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=actual_is_causal,
            scale=self.scale,
        )

        attn_out_flat = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        final_out = self.out_proj(attn_out_flat)

        if use_cache:
            return final_out, current_kv_cache
        return final_out


class FlashGQA(_BaseFlashGQA):
    """Unified GQA module dispatching between CUDA kernel, varlen reference, and batched SDPA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads_q: int,
        num_heads_kv: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        dropout: float = 0.0,
        bias: bool = False,
        is_cross_attention: bool = False,
        context_dim: Optional[int] = None,
        mode: str = "kernel",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            rope=rope,
            dropout=dropout,
            bias=bias,
            is_cross_attention=is_cross_attention,
            context_dim=context_dim,
        )
        if mode not in ("kernel", "varlen_ref", "batched_sdpa"):
            raise ValueError(f"Unsupported mode={mode!r}; expected 'kernel', 'varlen_ref', or 'batched_sdpa'")
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        offset: int = 0,
        offsets: Optional[torch.Tensor] = None,
        kv_cache=None,
        use_cache: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        decode: bool = False,
        mode: Optional[str] = None,
    ):
        active_mode = self.mode if mode is None else mode
        if active_mode == "kernel":
            return VarlenFlashGQA.forward(
                self,
                x=x,
                cu_seqlens_q=cu_seqlens_q,
                context=context,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                is_causal=is_causal,
                offsets=offsets,
                kv_cache=kv_cache,
                use_cache=use_cache,
                decode=decode,
            )
        if active_mode == "varlen_ref":
            return VarlenFlashGQABaselineCorrectness.forward(
                self,
                x=x,
                cu_seqlens_q=cu_seqlens_q,
                context=context,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                is_causal=is_causal,
                offsets=offsets,
                kv_cache=kv_cache,
                use_cache=use_cache,
                decode=decode,
            )
        if active_mode == "batched_sdpa":
            return BatchedFlashGQABaselineSpeed.forward(
                self,
                x=x,
                context=context,
                attn_mask=attn_mask,
                is_causal=is_causal,
                offset=offset,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
        raise ValueError(f"Unsupported mode={active_mode!r}")


__all__ = [
    "RotaryPositionalEmbedding",
    "VarlenKVCache",
    "varlen_attention_ref",
    "varlen_decode_ref",
    "VarlenFlashGQA",
    "VarlenFlashGQABaselineCorrectness",
    "BatchedFlashGQABaselineSpeed",
    "FlashGQA",
]
