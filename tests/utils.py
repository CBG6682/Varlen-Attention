"""Benchmarking, profiling, and tensor-packing utilities for Varlen Transformer."""

import math
import random
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """Set deterministic RNG seeds for Python and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_varlen_seqlens(
    batch_size: int,
    min_len: int,
    max_len: int,
    distribution: str = "uniform",
    device: Union[str, torch.device] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Generate variable sequence lengths and cumulative offsets `cu_seqlens`.

    Args:
        batch_size: Number of sequences in the batch.
        min_len: Minimum sequence length (>= 0).
        max_len: Maximum sequence length (>= min_len).
        distribution: 'uniform', 'skewed', or 'fixed'.
        device: Target device for output tensors.
    Returns:
        seqlens: (B,) int32 tensor of sequence lengths.
        cu_seqlens: (B + 1,) int32 cumulative sequence lengths starting with 0.
        max_seqlen: Python int maximum length in `seqlens`.
    """
    if min_len < 0 or max_len < min_len:
        raise ValueError(f"Invalid length bounds: min_len={min_len}, max_len={max_len}")

    if batch_size == 0:
        seqlens = torch.empty((0,), dtype=torch.int32, device=device)
        cu_seqlens = torch.zeros((1,), dtype=torch.int32, device=device)
        return seqlens, cu_seqlens, 0

    if distribution == "fixed" or min_len == max_len:
        seqlens = torch.full((batch_size,), max_len, dtype=torch.int32, device=device)
    elif distribution == "uniform":
        seqlens = torch.randint(
            min_len, max_len + 1, (batch_size,), dtype=torch.int32, device=device
        )
    elif distribution == "skewed":
        # 75% short sequences in [min_len, min_len + span // 4], 25% long sequences near max_len
        span = max_len - min_len
        short_hi = min_len + max(1, span // 4)
        long_lo = max(min_len, max_len - max(1, span // 8))
        is_long = torch.rand((batch_size,), device=device) < 0.25
        short_lens = torch.randint(min_len, short_hi + 1, (batch_size,), dtype=torch.int32, device=device)
        long_lens = torch.randint(long_lo, max_len + 1, (batch_size,), dtype=torch.int32, device=device)
        seqlens = torch.where(is_long, long_lens, short_lens)
    else:
        raise ValueError(f"Unknown distribution={distribution!r}; expected 'uniform', 'skewed', or 'fixed'")

    cu_seqlens = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
    torch.cumsum(seqlens, dim=0, out=cu_seqlens[1:])
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() > 0 else 0
    return seqlens, cu_seqlens, max_seqlen


def pack_varlen_tensor(padded_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Pack a padded batch tensor into contiguous packed varlen layout.

    Supports:
        - (B, L_max, D)    -> (Total_L, D)
        - (B, H, L_max, D) -> (H, Total_L, D)
    """
    batch_size = int(cu_seqlens.shape[0]) - 1
    if padded_tensor.dim() == 3:
        chunks = [
            padded_tensor[i, : int(cu_seqlens[i + 1].item() - cu_seqlens[i].item())]
            for i in range(batch_size)
        ]
        if not chunks:
            return padded_tensor.new_empty((0, padded_tensor.shape[-1]))
        return torch.cat(chunks, dim=0).contiguous()

    if padded_tensor.dim() == 4:
        chunks = [
            padded_tensor[i, :, : int(cu_seqlens[i + 1].item() - cu_seqlens[i].item())]
            for i in range(batch_size)
        ]
        if not chunks:
            return padded_tensor.new_empty((padded_tensor.shape[1], 0, padded_tensor.shape[-1]))
        return torch.cat(chunks, dim=1).contiguous()

    raise ValueError(f"Expected 3D or 4D padded_tensor, got shape {tuple(padded_tensor.shape)}")


def unpack_varlen_tensor(
    packed_tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """
    Unpack a packed varlen tensor back into zero-padded batched layout.

    Supports:
        - (Total_L, D)    -> (B, max_seqlen, D)
        - (H, Total_L, D) -> (B, H, max_seqlen, D)
    """
    batch_size = int(cu_seqlens.shape[0]) - 1
    if packed_tensor.dim() == 2:
        out = packed_tensor.new_zeros((batch_size, max_seqlen, packed_tensor.shape[-1]))
        for i in range(batch_size):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            length = end - start
            if length > 0:
                out[i, :length] = packed_tensor[start:end]
        return out

    if packed_tensor.dim() == 3:
        out = packed_tensor.new_zeros(
            (batch_size, packed_tensor.shape[0], max_seqlen, packed_tensor.shape[-1])
        )
        for i in range(batch_size):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            length = end - start
            if length > 0:
                out[i, :, :length] = packed_tensor[:, start:end]
        return out

    raise ValueError(f"Expected 2D or 3D packed_tensor, got shape {tuple(packed_tensor.shape)}")


def build_padding_mask(cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Build boolean attention padding mask `(B, 1, 1, max_seqlen)` (True = valid token)."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    positions = torch.arange(max_seqlen, device=cu_seqlens.device, dtype=seqlens.dtype)
    mask = positions.unsqueeze(0) < seqlens.unsqueeze(1)  # (B, max_seqlen)
    return mask.unsqueeze(1).unsqueeze(1)


def align_up(value: int, alignment: int = 8) -> int:
    """Round integer `value` up to a multiple of `alignment`."""
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def build_additive_padding_mask(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    num_heads_q: int = 1,
    seqlen_q: int = 1,
    dtype: torch.dtype = torch.float16,
    align_to: int = 8,
) -> Tuple[torch.Tensor, int]:
    """
    Build FP16 additive padding mask `(B, num_heads_q, seqlen_q, aligned_len)`
    (0.0 = valid, -inf = masked) with last dimension aligned to `align_to` (8)
    and explicit strides `(aligned_len, 0, 0, 1)` so all strides satisfy CUTLASS
    `SDPBackend.EFFICIENT_ATTENTION` 16-byte alignment checks even when B=1 or L_q=1.
    """
    aligned_len = max(align_to, align_up(max_seqlen, align_to))
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    B = int(seqlens.shape[0])
    positions = torch.arange(aligned_len, device=cu_seqlens.device, dtype=seqlens.dtype)
    valid = positions.unsqueeze(0) < seqlens.unsqueeze(1)  # (B, aligned_len)
    base_mask = torch.zeros((B, aligned_len), device=cu_seqlens.device, dtype=dtype)
    base_mask.masked_fill_(~valid, float("-inf"))
    mask = torch.as_strided(
        base_mask,
        size=(B, num_heads_q, seqlen_q, aligned_len),
        stride=(aligned_len, 0, 0, 1),
    )
    return mask, aligned_len


def cuda_benchmark(
    fn: Callable[[], Any],
    warmup: int = 20,
    rep: int = 100,
    clear_l2: bool = False,
) -> Tuple[float, float, float]:
    """
    Accurate GPU kernel timing using `torch.cuda.Event`.

    Args:
        fn: Zero-argument callable executing CUDA work.
        warmup: Warmup iterations before timing.
        rep: Timed iterations.
        clear_l2: If True, flush L2 cache outside event boundaries before each rep.
    Returns:
        (median_ms, mean_ms, min_ms)
    """
    l2_buffer: Optional[torch.Tensor] = None
    if clear_l2:
        # 16 MB buffer lớn hơn dung lượng L2 của T4 (4 MB) để xả cache giữa các lần đo
        l2_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.int8, device="cuda")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings_ms: List[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(rep):
        if l2_buffer is not None:
            l2_buffer.zero_()
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        timings_ms.append(start_evt.elapsed_time(end_evt))

    timings_t = torch.tensor(timings_ms, dtype=torch.float64)
    return (
        float(timings_t.median().item()),
        float(timings_t.mean().item()),
        float(timings_t.min().item()),
    )


def measure_peak_memory_mb(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Execute `fn()` and return `(output, peak_allocated_memory_mb)`."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return out, peak_mb


def compute_attention_flops(
    seqlens_q: Union[torch.Tensor, Sequence[int]],
    seqlens_k: Optional[Union[torch.Tensor, Sequence[int]]] = None,
    num_heads_q: int = 8,
    head_dim: int = 64,
    is_causal: bool = False,
    include_backward: bool = False,
) -> float:
    """
    Compute exact theoretical FlashAttention FLOPs (summed per variable-length sequence).
    Forward counts 4 * H_q * active_qk_pairs * head_dim; backward adds 2.5x forward (3.5x total).
    """
    sq = seqlens_q.tolist() if isinstance(seqlens_q, torch.Tensor) else list(seqlens_q)
    if seqlens_k is None:
        sk = sq
    else:
        sk = seqlens_k.tolist() if isinstance(seqlens_k, torch.Tensor) else list(seqlens_k)

    active_pairs = 0
    for lq_i, lk_i in zip(sq, sk):
        lq_i = int(lq_i)
        lk_i = int(lk_i)
        if lq_i <= 0 or lk_i <= 0:
            continue
        if not is_causal:
            active_pairs += lq_i * lk_i
        else:
            # Số ô tam giác hợp lệ với mặt nạ causal col <= row
            n_sq = min(lq_i, lk_i)
            tri_pairs = n_sq * (n_sq + 1) // 2
            tail_rows = max(0, lq_i - lk_i)
            active_pairs += tri_pairs + tail_rows * lk_i

    fwd_flops = 4.0 * num_heads_q * float(active_pairs) * head_dim
    return fwd_flops * 3.5 if include_backward else fwd_flops


def compute_tflops(flops: float, time_ms: float) -> float:
    """Convert FLOPs and execution time (ms) into TFLOP/s."""
    if time_ms <= 0.0:
        return 0.0
    return flops / (time_ms * 1e-3) / 1e12


def compute_decode_bandwidth_gbps(
    total_lk: int,
    num_heads_kv: int,
    num_heads_q: int,
    batch_size: int,
    head_dim: int,
    time_ms: float,
    elem_bytes: int = 2,
) -> float:
    """
    Compute effective HBM memory bandwidth (GB/s) for Split-K Flash Decoding (Lq = 1):
    Reads raw k_cache + v_cache + q (+ out_o write), neglecting small cos/sin L2-hit traffic.
    """
    if time_ms <= 0.0:
        return 0.0
    kv_bytes = 2.0 * num_heads_kv * total_lk * head_dim * elem_bytes
    qo_bytes = 2.0 * num_heads_q * batch_size * head_dim * elem_bytes
    return (kv_bytes + qo_bytes) / (time_ms * 1e-3) / 1e9


def check_allclose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Tuple[bool, float, float]:
    """Compare two tensors in float32 and return `(passed, max_abs_err, max_rel_err)`."""
    a_f32 = actual.detach().float()
    e_f32 = expected.detach().float()
    diff = (a_f32 - e_f32).abs()
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    denom = e_f32.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
    passed = bool(torch.allclose(a_f32, e_f32, atol=atol, rtol=rtol))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:<36} max_abs={max_abs:.6f}  max_rel={max_rel:.6f}")
    return passed, max_abs, max_rel


def build_packed_rope_positions(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Build 1D token position indices `[0..L_0-1, 0..L_1-1, ...]` for packed varlen tensors."""
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if not seqlens:
        return torch.empty((0,), dtype=torch.int64, device=cu_seqlens.device)
    chunks = [torch.arange(int(l), device=cu_seqlens.device, dtype=torch.int64) for l in seqlens]
    return torch.cat(chunks, dim=0)


def format_markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> str:
    """Format rows and headers into an aligned GitHub-flavored Markdown table."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(cells: Sequence[Any]) -> str:
        padded = [str(c).ljust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [_fmt_row(headers), sep]
    lines.extend(_fmt_row(row) for row in str_rows)
    return "\n".join(lines)


__all__ = [
    "seed_everything",
    "generate_varlen_seqlens",
    "pack_varlen_tensor",
    "unpack_varlen_tensor",
    "build_padding_mask",
    "align_up",
    "build_additive_padding_mask",
    "build_packed_rope_positions",
    "cuda_benchmark",
    "measure_peak_memory_mb",
    "compute_attention_flops",
    "compute_tflops",
    "compute_decode_bandwidth_gbps",
    "check_allclose",
    "format_markdown_table",
]
