"""Inference (Decode L=1 & End-to-End Prefill+Generation) speed and E2E VRAM benchmark with torch.compile baseline."""

import argparse
import gc
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:
    SDPBackend = None
    sdpa_kernel = None
    _HAS_SDPA_KERNEL = False

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tests.utils import (
    HAS_FLASH_ATTN_1,
    FlashAttention1VarlenBaseline,
    align_up,
    build_additive_padding_mask,
    build_packed_rope_positions,
    build_padding_mask,
    compute_decode_bandwidth_gbps,
    cuda_benchmark,
    fa1_unpadded_func,
    format_markdown_table,
    generate_varlen_seqlens,
    measure_peak_memory_mb,
    pack_varlen_tensor,
    seed_everything,
)
from varlen_transformer import (
    HAS_CUDA_EXT,
    BatchedFlashGQABaselineSpeed,
    RotaryPositionalEmbedding,
    VarlenFlashGQA,
    VarlenKVCache,
    varlen_flash_decode,
    varlen_kv_append,
)


def _repeat_kv_batched(x: torch.Tensor, rep: int) -> torch.Tensor:
    if rep == 1:
        return x
    B, H_kv, L, D = x.shape
    return x.unsqueeze(2).expand(B, H_kv, rep, L, D).reshape(B, H_kv * rep, L, D)


def _repeat_kv_packed(x: torch.Tensor, rep: int) -> torch.Tensor:
    """Convert `(H_kv, Total_L, D)` -> `(Total_L, H_q, D)` for FlashAttention-1."""
    H_kv, total_l, D = x.shape
    x_lh = x.transpose(0, 1).contiguous()
    if rep == 1:
        return x_lh
    return x_lh.unsqueeze(2).expand(total_l, H_kv, rep, D).reshape(total_l, H_kv * rep, D)


def _compile_or_warn(target: Any, label: str, dynamic: bool = True):
    """Compile target with TorchInductor; return None on failure."""
    try:
        return torch.compile(target, backend="inductor", dynamic=dynamic)
    except Exception as exc:
        print(f"[WARN] Could not torch.compile({label}): {exc}")
        return None


def _verify_efficient_attention_active(
    batch_size: int,
    num_heads_q: int,
    aligned_len: int,
    head_dim: int,
    scale: float,
    device: torch.device,
) -> bool:
    """Verify that `SDPBackend.EFFICIENT_ATTENTION` is active for Part A FP16 additive mask."""
    q_chk = torch.randn((batch_size, num_heads_q, 1, head_dim), device=device, dtype=torch.float16)
    k_chk = torch.randn((batch_size, num_heads_q, aligned_len, head_dim), device=device, dtype=torch.float16)
    v_chk = torch.randn_like(k_chk)
    cu_chk = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * aligned_len
    mask_chk, _ = build_additive_padding_mask(
        cu_chk, aligned_len, num_heads_q=num_heads_q, seqlen_q=1, dtype=torch.float16, align_to=8
    )
    try:
        choice = None
        if hasattr(torch.ops.aten, "_fused_sdp_choice"):
            choice = int(torch.ops.aten._fused_sdp_choice(q_chk, k_chk, v_chk, attn_mask=mask_chk, scale=scale))
        if _HAS_SDPA_KERNEL:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                _ = F.scaled_dot_product_attention(q_chk, k_chk, v_chk, attn_mask=mask_chk, scale=scale)
        print(f"[INFO] Part A SDPA Backend verified: SDPBackend.EFFICIENT_ATTENTION (choice={choice})")
        return True
    except Exception as exc:
        print(f"[WARN] SDPBackend.EFFICIENT_ATTENTION verification failed: {exc}")
        return False


def _save_inference_plot(
    decode_results: List[Dict[str, Any]],
    e2e_results: List[Dict[str, Any]],
    save_path: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping inference plot.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    skewed_rows = [r for r in decode_results if r.get("dist") == "skewed"] or decode_results

    fig, axes = plt.subplots(2, 2, figsize=(20, 11.5))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # -------------------------------------------------------------------------
    # Subplot 1: Part A Decode L_q=1 Latency (Baseline vs Direct vs Split-K vs Auto)
    # -------------------------------------------------------------------------
    labels = [f"B={r['B']}\nL={r['max_len']}" for r in skewed_rows]
    x_idx = list(range(len(labels)))
    has_comp_dec = all(
        r.get("batched_comp_ms") is not None and r.get("varlen_comp_ms") is not None
        for r in skewed_rows
    )

    has_fa1_dec = all(r.get("fa1_dec_ms") is not None for r in skewed_rows)
    if has_comp_dec:
        t_base = [r["batched_comp_ms"] for r in skewed_rows]
        t_dir = [r.get("varlen_direct_comp_ms") or r["varlen_direct_eager_ms"] for r in skewed_rows]
        t_spk = [r.get("varlen_splitk_comp_ms") or r["varlen_splitk_eager_ms"] for r in skewed_rows]
        t_auto = [r["varlen_comp_ms"] for r in skewed_rows]
        if has_fa1_dec:
            w = 0.165
            t_fa1 = [r["fa1_dec_ms"] for r in skewed_rows]
            ax1.bar([i - 2.0 * w for i in x_idx], t_base, w, label="Batched SDPA + Compile (Base)", color="#e63946")
            ax1.bar([i - 1.0 * w for i in x_idx], t_fa1, w, label="FlashAttention-1 (Varlen)", color="#8ecae6")
            ax1.bar(x_idx, t_dir, w, label="Direct 1-Block Decode", color="#457b9d")
            ax1.bar([i + 1.0 * w for i in x_idx], t_spk, w, label="Split-K Flash Decode", color="#e9c46a")
            ax1.bar([i + 2.0 * w for i in x_idx], t_auto, w, label="Auto-Dispatch (Ours)", color="#2a9d8f")
            auto_x_off = 2.0 * w
        else:
            w = 0.21
            ax1.bar([i - 1.5 * w for i in x_idx], t_base, w, label="Batched SDPA + Compile (Base)", color="#e63946")
            ax1.bar([i - 0.5 * w for i in x_idx], t_dir, w, label="Direct 1-Block Decode", color="#457b9d")
            ax1.bar([i + 0.5 * w for i in x_idx], t_spk, w, label="Split-K Flash Decode", color="#e9c46a")
            ax1.bar([i + 1.5 * w for i in x_idx], t_auto, w, label="Auto-Dispatch (Ours)", color="#2a9d8f")
            auto_x_off = 1.5 * w
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_comp", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + auto_x_off, t_auto[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                fontweight="bold",
            )
        ax1.set_title("Part A: Decode L_q=1 Latency vs Compiled Baseline (Lower is Better)")
    else:
        w = 0.21
        t_base = [r["batched_dec_ms"] for r in skewed_rows]
        t_dir = [r["varlen_direct_eager_ms"] for r in skewed_rows]
        t_spk = [r["varlen_splitk_eager_ms"] for r in skewed_rows]
        t_auto = [r["varlen_dec_ms"] for r in skewed_rows]
        ax1.bar([i - 1.5 * w for i in x_idx], t_base, w, label="Batched SDPA (Eager)", color="#6c757d")
        ax1.bar([i - 0.5 * w for i in x_idx], t_dir, w, label="Direct 1-Block Decode", color="#457b9d")
        ax1.bar([i + 0.5 * w for i in x_idx], t_spk, w, label="Split-K Flash Decode", color="#e9c46a")
        ax1.bar([i + 1.5 * w for i in x_idx], t_auto, w, label="Auto-Dispatch (Ours)", color="#0d6efd")
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_eager", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + 1.5 * w, t_auto[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                fontweight="bold",
            )
        ax1.set_title("Part A: Decode L_q=1 Latency (Lower is Better)")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------------------
    # Subplots 2, 3, 4: Part B End-to-End (Baseline vs Direct 1-Block vs Split-K)
    # -------------------------------------------------------------------------
    if e2e_results:
        e2e_labels = [f"B={r['B']}\nL={r['max_len']}" for r in e2e_results]
        e_idx = list(range(len(e2e_labels)))
        w3 = 0.26

        has_comp_e2e = all(
            r.get("batched_comp_tpot_ms") is not None
            and r.get("varlen_direct_comp_tpot_ms") is not None
            and r.get("varlen_splitk_comp_tpot_ms") is not None
            for r in e2e_results
        )

        # Subplot 2: E2E Generation TPOT (ms/token)
        if has_comp_e2e:
            b_tpot = [r["batched_comp_tpot_ms"] for r in e2e_results]
            d_tpot = [r["varlen_direct_comp_tpot_ms"] for r in e2e_results]
            s_tpot = [r["varlen_splitk_comp_tpot_ms"] for r in e2e_results]
            ax2.set_title("Part B: E2E Generation TPOT (ms/token) vs Compiled Baseline")
        else:
            b_tpot = [r["batched_tpot_ms"] for r in e2e_results]
            d_tpot = [r["varlen_direct_tpot_ms"] for r in e2e_results]
            s_tpot = [r["varlen_splitk_tpot_ms"] for r in e2e_results]
            ax2.set_title("Part B: E2E Generation TPOT (ms/token) [Eager]")

        ax2.bar([i - w3 for i in e_idx], b_tpot, w3, label="Baseline (Batched SDPA)", color="#e63946")
        ax2.bar(e_idx, d_tpot, w3, label="Direct 1-Block", color="#457b9d")
        ax2.bar([i + w3 for i in e_idx], s_tpot, w3, label="Flash Decoding (Split-K)", color="#2a9d8f")
        for i in e_idx:
            best_v = min(d_tpot[i], s_tpot[i])
            best_x = i if d_tpot[i] <= s_tpot[i] else i + w3
            sp = b_tpot[i] / max(best_v, 1e-6)
            ax2.annotate(
                f"{sp:.2f}x",
                xy=(best_x, best_v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                fontweight="bold",
            )
        ax2.set_ylabel("TPOT (ms / token)")
        ax2.set_xticks(e_idx)
        ax2.set_xticklabels(e2e_labels, rotation=25, ha="right", fontsize=7.5)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend(loc="upper left", fontsize=8)

        # Subplot 3: Total E2E Latency (Prefill + Generation)
        if has_comp_e2e:
            b_e2e = [r["batched_comp_e2e_ms"] for r in e2e_results]
            d_e2e = [r["varlen_direct_comp_e2e_ms"] for r in e2e_results]
            s_e2e = [r["varlen_splitk_comp_e2e_ms"] for r in e2e_results]
            ax3.set_title("Part B: Total E2E Latency (Prefill + Gen) vs Compiled Baseline")
        else:
            b_e2e = [r["batched_e2e_ms"] for r in e2e_results]
            d_e2e = [r["varlen_direct_e2e_ms"] for r in e2e_results]
            s_e2e = [r["varlen_splitk_e2e_ms"] for r in e2e_results]
            ax3.set_title("Part B: Total E2E Latency (Prefill + Gen) [Eager]")

        ax3.bar([i - w3 for i in e_idx], b_e2e, w3, label="Baseline (Batched SDPA)", color="#e63946")
        ax3.bar(e_idx, d_e2e, w3, label="Direct 1-Block", color="#457b9d")
        ax3.bar([i + w3 for i in e_idx], s_e2e, w3, label="Flash Decoding (Split-K)", color="#2a9d8f")
        for i in e_idx:
            best_v = min(d_e2e[i], s_e2e[i])
            best_x = i if d_e2e[i] <= s_e2e[i] else i + w3
            sp = b_e2e[i] / max(best_v, 1e-6)
            ax3.annotate(
                f"{sp:.2f}x",
                xy=(best_x, best_v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                fontweight="bold",
            )
        ax3.set_ylabel("Total E2E Latency (ms)")
        ax3.set_xticks(e_idx)
        ax3.set_xticklabels(e2e_labels, rotation=25, ha="right", fontsize=7.5)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        ax3.legend(loc="upper left", fontsize=8)

        # Subplot 4: E2E Peak Allocated VRAM (MB)
        b_vram = [r["batched_vram_mb"] for r in e2e_results]
        d_vram = [r["varlen_direct_vram_mb"] for r in e2e_results]
        s_vram = [r["varlen_splitk_vram_mb"] for r in e2e_results]
        ax4.bar([i - w3 for i in e_idx], b_vram, w3, label="Baseline (Batched Padded)", color="#f4a261")
        ax4.bar(e_idx, d_vram, w3, label="Direct 1-Block (Packed)", color="#457b9d")
        ax4.bar([i + w3 for i in e_idx], s_vram, w3, label="Flash Decoding Split-K (Packed)", color="#264653")
        for i in e_idx:
            save_pct = (1.0 - min(d_vram[i], s_vram[i]) / max(b_vram[i], 1e-6)) * 100.0
            ax4.annotate(
                f"-{save_pct:.0f}%",
                xy=(i, d_vram[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                fontweight="bold",
            )
        ax4.set_title("Part B: E2E Peak Allocated VRAM (Lower is Better)")
        ax4.set_ylabel("Peak Memory (MB)")
        ax4.set_xticks(e_idx)
        ax4.set_xticklabels(e2e_labels, rotation=25, ha="right", fontsize=7.5)
        ax4.grid(axis="y", linestyle="--", alpha=0.4)
        ax4.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved inference benchmark plot -> {save_path}")


def run_inference_benchmark(
    decode_batch_sizes: Sequence[int] = (1, 8, 32, 64),
    decode_max_lens: Sequence[int] = (128, 256, 1024, 2048, 4096),
    decode_distributions: Sequence[str] = ("skewed", "uniform"),
    e2e_configs: Optional[Sequence[Tuple[int, int]]] = None,
    num_heads_q: int = 32,
    num_heads_kv: int = 8,
    head_dim: int = 64,
    gen_steps: int = 32,
    use_rope: bool = True,
    include_compile: bool = True,
    clear_l2: bool = True,
    warmup: int = 15,
    rep: int = 50,
    quick_e2e: bool = False,
    save_plot: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run Decode L_q=1 kernel benchmark (speed/BW only) and End-to-End Prefill + Generation benchmark
    measuring Baseline (Batched SDPA), Direct 1-Block Decoding, and Split-K Flash Decoding separately
    alongside E2E Peak VRAM.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference benchmark.")
    if not HAS_CUDA_EXT:
        raise RuntimeError("varlen_transformer._C extension is not compiled.")

    seed_everything(seed)
    dev = torch.device("cuda")
    embed_dim = num_heads_q * head_dim
    rep_gqa = num_heads_q // num_heads_kv
    scale = 1.0 / math.sqrt(head_dim)

    all_lens = list(decode_max_lens)
    if e2e_configs:
        all_lens.extend([cfg[1] for cfg in e2e_configs])
    else:
        all_lens.extend([128, 256, 512, 1024, 2048])
    max_table_len = max(all_lens) + gen_steps + 128

    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_table_len, device=dev) if use_rope else None
    cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16) if rope is not None else (None, None)

    # =========================================================================
    # PART A: Microbenchmark Decode Attention L_q = 1 (Latency, BW, Tok/s only)
    # =========================================================================
    print("\n" + "=" * 95)
    print(" PART A: Microbenchmark Decode Attention L_q=1 (Cold L2 - Speed & Bandwidth Only)")
    print(f" Config: H_q={num_heads_q}, H_kv={num_heads_kv}, D={head_dim}, rope={use_rope}, clear_l2={clear_l2}")
    print(" Baseline #1 Speed: Batched SDPA (SDPBackend.EFFICIENT_ATTENTION) + torch.compile(dynamic=False)")
    print(f" Baseline #2 Speed: FlashAttention-1 ({'Enabled' if HAS_FLASH_ATTN_1 else 'Not installed'})")
    print(" Custom Kernel:     Direct 1-Block vs Split-K vs Adaptive Auto-Dispatch (+ torch.compile)")
    print("=" * 95)

    eff_attn_ok = _verify_efficient_attention_active(
        batch_size=min(decode_batch_sizes) if decode_batch_sizes else 1,
        num_heads_q=num_heads_q,
        aligned_len=128,
        head_dim=head_dim,
        scale=scale,
        device=dev,
    )

    compiled_varlen_auto = None
    compiled_varlen_direct = None
    compiled_varlen_splitk = None
    compiled_batched_dec = None
    if include_compile:
        def _c_varlen_auto(q_in, k_in, v_in, cu_in, max_k, c_in, s_in):
            return varlen_flash_decode(q_in, k_in, v_in, cu_in, max_seqlen_k=max_k, cos=c_in, sin=s_in)

        compiled_varlen_auto = _compile_or_warn(_c_varlen_auto, "varlen_flash_decode_auto")

        def _c_varlen_direct(q_in, k_in, v_in, cu_in, max_k, c_in, s_in):
            return varlen_flash_decode(
                q_in, k_in, v_in, cu_in, max_seqlen_k=max_k, cos=c_in, sin=s_in, use_splitk=False
            )

        compiled_varlen_direct = _compile_or_warn(_c_varlen_direct, "varlen_flash_decode_direct")

        def _c_varlen_splitk(q_in, k_in, v_in, cu_in, max_k, c_in, s_in):
            return varlen_flash_decode(
                q_in, k_in, v_in, cu_in, max_seqlen_k=max_k, cos=c_in, sin=s_in, use_splitk=True
            )

        compiled_varlen_splitk = _compile_or_warn(_c_varlen_splitk, "varlen_flash_decode_splitk")

        def _c_batched_dec(q_b, k_exp, v_exp, pad_m, sc_val):
            return F.scaled_dot_product_attention(q_b, k_exp, v_exp, attn_mask=pad_m, scale=sc_val)

        # Static shape compilation so Inductor proves pad_m.size(-1) % 8 == 0 and uses EFFICIENT_ATTENTION
        compiled_batched_dec = _compile_or_warn(_c_batched_dec, "batched_sdpa_decode", dynamic=False)

    decode_results: List[Dict[str, Any]] = []
    decode_rows: List[List[str]] = []

    for dist in decode_distributions:
        for B in decode_batch_sizes:
            for max_len in decode_max_lens:
                min_len = max(16, max_len // 16)
                seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
                    B, min_len, max_len, distribution=dist, device=dev
                )
                total_lk = int(cu_seqlens[-1].item())

                try:
                    # 1. Batched SDPA Decode Baseline (FP16 additive mask + L%8 aligned for EFFICIENT_ATTENTION)
                    pad_mask, aligned_max = build_additive_padding_mask(
                        cu_seqlens,
                        actual_max,
                        num_heads_q=num_heads_q,
                        seqlen_q=1,
                        dtype=torch.float16,
                        align_to=8,
                    )
                    q_batched = torch.randn((B, num_heads_q, 1, head_dim), device=dev, dtype=torch.float16)
                    k_batched = torch.randn((B, num_heads_kv, aligned_max, head_dim), device=dev, dtype=torch.float16)
                    v_batched = torch.randn((B, num_heads_kv, aligned_max, head_dim), device=dev, dtype=torch.float16)
                    q_batched_roped = rope(q_batched, offset=actual_max - 1) if rope is not None else q_batched
                    k_batched_exp = _repeat_kv_batched(k_batched, rep_gqa)
                    v_batched_exp = _repeat_kv_batched(v_batched, rep_gqa)

                    sdp_backends = (
                        SDPBackend.EFFICIENT_ATTENTION
                        if eff_attn_ok
                        else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
                    )

                    b_comp_ms: Optional[float] = None
                    if compiled_batched_dec is not None:
                        def _batched_dec_comp_fn():
                            with torch.no_grad():
                                if _HAS_SDPA_KERNEL:
                                    with sdpa_kernel(sdp_backends):
                                        return compiled_batched_dec(
                                            q_batched_roped, k_batched_exp, v_batched_exp, pad_mask, scale
                                        )
                                return compiled_batched_dec(
                                    q_batched_roped, k_batched_exp, v_batched_exp, pad_mask, scale
                                )

                        b_comp_ms, _, _ = cuda_benchmark(
                            _batched_dec_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                        )

                    def _batched_dec_eager_fn():
                        with torch.no_grad():
                            if _HAS_SDPA_KERNEL:
                                with sdpa_kernel(sdp_backends):
                                    return F.scaled_dot_product_attention(
                                        q_batched_roped, k_batched_exp, v_batched_exp, attn_mask=pad_mask, scale=scale
                                    )
                            return F.scaled_dot_product_attention(
                                q_batched_roped, k_batched_exp, v_batched_exp, attn_mask=pad_mask, scale=scale
                            )

                    b_eager_ms, _, _ = cuda_benchmark(
                        _batched_dec_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                    )

                    del q_batched, q_batched_roped, k_batched, v_batched, k_batched_exp, v_batched_exp, pad_mask
                    gc.collect()
                    torch.cuda.empty_cache()

                    # 2. Varlen Decode (Direct 1-Block, Split-K, Auto-Dispatch)
                    q_dec = torch.randn((num_heads_q, B, head_dim), device=dev, dtype=torch.float16)
                    k_packed = torch.randn((num_heads_kv, total_lk, head_dim), device=dev, dtype=torch.float16)
                    v_packed = torch.randn((num_heads_kv, total_lk, head_dim), device=dev, dtype=torch.float16)

                    v_direct_comp_ms: Optional[float] = None
                    v_splitk_comp_ms: Optional[float] = None
                    v_auto_comp_ms: Optional[float] = None

                    if compiled_varlen_direct is not None:
                        def _varlen_direct_comp_fn():
                            with torch.no_grad():
                                return compiled_varlen_direct(
                                    q_dec, k_packed, v_packed, cu_seqlens, actual_max, cos, sin
                                )

                        v_direct_comp_ms, _, _ = cuda_benchmark(
                            _varlen_direct_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                        )

                    if compiled_varlen_splitk is not None:
                        def _varlen_splitk_comp_fn():
                            with torch.no_grad():
                                return compiled_varlen_splitk(
                                    q_dec, k_packed, v_packed, cu_seqlens, actual_max, cos, sin
                                )

                        v_splitk_comp_ms, _, _ = cuda_benchmark(
                            _varlen_splitk_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                        )

                    if compiled_varlen_auto is not None:
                        def _varlen_auto_comp_fn():
                            with torch.no_grad():
                                return compiled_varlen_auto(
                                    q_dec, k_packed, v_packed, cu_seqlens, actual_max, cos, sin
                                )

                        v_auto_comp_ms, _, _ = cuda_benchmark(
                            _varlen_auto_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                        )

                    def _varlen_direct_eager_fn():
                        with torch.no_grad():
                            return varlen_flash_decode(
                                q_dec,
                                k_packed,
                                v_packed,
                                cu_seqlens,
                                max_seqlen_k=actual_max,
                                cos=cos,
                                sin=sin,
                                use_splitk=False,
                            )

                    def _varlen_splitk_eager_fn():
                        with torch.no_grad():
                            return varlen_flash_decode(
                                q_dec,
                                k_packed,
                                v_packed,
                                cu_seqlens,
                                max_seqlen_k=actual_max,
                                cos=cos,
                                sin=sin,
                                use_splitk=True,
                            )

                    def _varlen_auto_eager_fn():
                        with torch.no_grad():
                            return varlen_flash_decode(
                                q_dec, k_packed, v_packed, cu_seqlens, max_seqlen_k=actual_max, cos=cos, sin=sin
                            )

                    v_direct_eager_ms, _, _ = cuda_benchmark(
                        _varlen_direct_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                    )
                    v_splitk_eager_ms, _, _ = cuda_benchmark(
                        _varlen_splitk_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                    )
                    v_eager_ms, _, _ = cuda_benchmark(
                        _varlen_auto_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                    )

                    # 3. FlashAttention-1 Decode L_q=1 Baseline (if installed)
                    fa1_dec_ms: Optional[float] = None
                    if HAS_FLASH_ATTN_1 and fa1_unpadded_func is not None:
                        try:
                            q_fa1 = q_dec.transpose(0, 1).contiguous()  # (B, H_q, D)
                            k_fa1 = _repeat_kv_packed(k_packed, rep_gqa)  # (Total_Lk, H_q, D)
                            v_fa1 = _repeat_kv_packed(v_packed, rep_gqa)
                            cu_q_fa1 = torch.arange(B + 1, device=dev, dtype=torch.int32)

                            def _fa1_dec_fn():
                                with torch.no_grad():
                                    return fa1_unpadded_func(
                                        q_fa1,
                                        k_fa1,
                                        v_fa1,
                                        cu_q_fa1,
                                        cu_seqlens,
                                        1,
                                        actual_max,
                                        0.0,
                                        softmax_scale=scale,
                                        causal=False,
                                    )

                            fa1_dec_ms, _, _ = cuda_benchmark(
                                _fa1_dec_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                            )
                            del q_fa1, k_fa1, v_fa1, cu_q_fa1
                        except Exception as exc:
                            print(f"[WARN] FA1 Part A decode skipped for B={B}, MaxK={actual_max}: {exc}")

                    del q_dec, k_packed, v_packed
                    gc.collect()
                    torch.cuda.empty_cache()

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"[WARN] Part A skipped (OOM) for dist={dist}, B={B}, max_len={max_len}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    raise

                speedup_comp = (
                    b_comp_ms / max(v_auto_comp_ms, 1e-6)
                    if b_comp_ms is not None and v_auto_comp_ms is not None
                    else None
                )
                speedup_eager = b_eager_ms / max(v_eager_ms, 1e-6)

                best_v_ms = v_auto_comp_ms if v_auto_comp_ms is not None else v_eager_ms
                bw_gbps = compute_decode_bandwidth_gbps(
                    total_lk, num_heads_kv, num_heads_q, B, head_dim, best_v_ms
                )
                tok_per_s = B / (best_v_ms * 1e-3)

                decode_results.append(
                    {
                        "dist": dist,
                        "B": B,
                        "max_len": actual_max,
                        "total_lk": total_lk,
                        "batched_comp_ms": b_comp_ms,
                        "fa1_dec_ms": fa1_dec_ms,
                        "varlen_direct_comp_ms": v_direct_comp_ms,
                        "varlen_splitk_comp_ms": v_splitk_comp_ms,
                        "varlen_comp_ms": v_auto_comp_ms,
                        "speedup_comp": speedup_comp,
                        "batched_dec_ms": b_eager_ms,
                        "varlen_direct_eager_ms": v_direct_eager_ms,
                        "varlen_splitk_eager_ms": v_splitk_eager_ms,
                        "varlen_dec_ms": v_eager_ms,
                        "speedup_eager": speedup_eager,
                        "bw_gbps": bw_gbps,
                        "tok_per_s": tok_per_s,
                    }
                )

                b_comp_str = f"{b_comp_ms:.3f}" if b_comp_ms is not None else f"{b_eager_ms:.3f}"
                v_dir_str = f"{v_direct_comp_ms:.3f}" if v_direct_comp_ms is not None else f"{v_direct_eager_ms:.3f}"
                v_spk_str = f"{v_splitk_comp_ms:.3f}" if v_splitk_comp_ms is not None else f"{v_splitk_eager_ms:.3f}"
                v_auto_str = f"{v_auto_comp_ms:.3f}" if v_auto_comp_ms is not None else f"{v_eager_ms:.3f}"
                sp_comp_str = f"{speedup_comp:.2f}x" if speedup_comp is not None else f"{speedup_eager:.2f}x"

                row_dec = [
                    dist,
                    str(B),
                    str(actual_max),
                    str(total_lk),
                    b_comp_str,
                    v_dir_str,
                    v_spk_str,
                    v_auto_str,
                    sp_comp_str,
                    f"{bw_gbps:.1f}",
                    f"{tok_per_s:,.0f}",
                ]
                if HAS_FLASH_ATTN_1:
                    row_dec.insert(5, f"{fa1_dec_ms:.3f}" if fa1_dec_ms is not None else "N/A")
                decode_rows.append(row_dec)

    decode_headers = [
        "Dist",
        "B",
        "MaxK",
        "TotalLk",
        "Batched+Comp(ms) [Base]",
        "Direct 1-Blk(ms)",
        "Split-K(ms)",
        "Auto+Comp(ms) [Ours]",
        "Comp Spd",
        "BW (GB/s)",
        "Tok/s",
    ]
    if HAS_FLASH_ATTN_1:
        decode_headers.insert(5, "FA1(ms)")

    print("\n" + format_markdown_table(decode_headers, decode_rows))

    # =========================================================================
    # PART B: End-to-End Prefill (TTFT) + Generation (TPOT) + E2E Peak VRAM
    # Separate explicit comparison: Baseline vs Direct 1-Block vs Split-K Flash Decoding
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" PART B: End-to-End Prefill + {gen_steps}-Step Generation + Peak VRAM Benchmark")
    print(f" Config: H_q={num_heads_q}, H_kv={num_heads_kv}, embed_dim={embed_dim} (high SM & VRAM utilization)")
    print(" Comparing separately: Baseline (Batched SDPA) vs Direct 1-Block vs Split-K Flash Decoding")
    print("=" * 95)

    mod_varlen = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched = BatchedFlashGQABaselineSpeed(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched.load_state_dict(mod_varlen.state_dict())

    mod_fa1 = None
    if HAS_FLASH_ATTN_1:
        mod_fa1 = FlashAttention1VarlenBaseline(
            embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False
        ).to(device=dev, dtype=torch.float16)
        mod_fa1.load_state_dict(mod_varlen.state_dict())

    mod_varlen_comp = _compile_or_warn(mod_varlen, "VarlenFlashGQA") if include_compile else None
    mod_batched_comp = _compile_or_warn(mod_batched, "BatchedFlashGQA") if include_compile else None

    compiled_varlen_step_direct = None
    compiled_varlen_step_splitk = None
    compiled_batched_step_fn = None
    if include_compile:
        def _c_varlen_step_direct(x_step, k_cache, v_cache, cu_seqlens_k, max_k, cos_t, sin_t, offsets_t):
            q, new_k, new_v = mod_varlen._project_qkv_decode(x_step)
            k_next, v_next, cu_next = varlen_kv_append(k_cache, v_cache, cu_seqlens_k, new_k, new_v)
            attn_out = varlen_flash_decode(
                q,
                k_next,
                v_next,
                cu_next,
                max_seqlen_k=max_k,
                cos=cos_t,
                sin=sin_t,
                offsets=offsets_t,
                use_splitk=False,
            )
            out_flat = attn_out.transpose(0, 1).contiguous().view(x_step.shape[0], -1)
            return mod_varlen.out_proj(out_flat), k_next, v_next, cu_next

        compiled_varlen_step_direct = _compile_or_warn(_c_varlen_step_direct, "varlen_decode_step_direct")

        def _c_varlen_step_splitk(x_step, k_cache, v_cache, cu_seqlens_k, max_k, cos_t, sin_t, offsets_t):
            q, new_k, new_v = mod_varlen._project_qkv_decode(x_step)
            k_next, v_next, cu_next = varlen_kv_append(k_cache, v_cache, cu_seqlens_k, new_k, new_v)
            attn_out = varlen_flash_decode(
                q,
                k_next,
                v_next,
                cu_next,
                max_seqlen_k=max_k,
                cos=cos_t,
                sin=sin_t,
                offsets=offsets_t,
                use_splitk=True,
            )
            out_flat = attn_out.transpose(0, 1).contiguous().view(x_step.shape[0], -1)
            return mod_varlen.out_proj(out_flat), k_next, v_next, cu_next

        compiled_varlen_step_splitk = _compile_or_warn(_c_varlen_step_splitk, "varlen_decode_step_splitk")

        def _c_batched_step(x_step, past_k, past_v, step_mask, offset_t):
            qkv = mod_batched.qkv_proj(x_step)
            B_sz = x_step.shape[0]
            qkv = qkv.view(
                B_sz, 1, mod_batched.num_heads_q + 2 * mod_batched.num_heads_kv, mod_batched.head_dim
            ).transpose(1, 2)
            q, k, v = qkv.split(
                [mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1
            )
            if mod_batched.rope is not None:
                q = mod_batched.rope(q, offset=offset_t)
                k = mod_batched.rope(k, offset=offset_t)
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
            k_exp = _repeat_kv_batched(k_all, rep_gqa)
            v_exp = _repeat_kv_batched(v_all, rep_gqa)
            if _HAS_SDPA_KERNEL:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=step_mask, scale=scale)
            else:
                out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=step_mask, scale=scale)
            out_flat = out.transpose(1, 2).contiguous().view(B_sz, 1, -1)
            return mod_batched.out_proj(out_flat), k_all, v_all

        compiled_batched_step_fn = _compile_or_warn(_c_batched_step, "batched_decode_step")

    # Expanded E2E evaluation matrix: high batches (up to B=96 at short L) + long contexts (up to 2048)
    if e2e_configs is None:
        if quick_e2e:
            e2e_configs = [
                (1, 1024),
                (8, 1024),
                (32, 256),
                (64, 128),
                (64, 256),
            ]
        else:
            e2e_configs = [
                # Small-to-medium batch, long context (Split-K Flash Decoding territory)
                (1, 1024),
                (1, 2048),
                (4, 1024),
                (4, 2048),
                (8, 512),
                (8, 1024),
                # High-concurrency / short-to-medium L (saturate SMs & maximize VRAM usage)
                (16, 256),
                (16, 512),
                (16, 1024),
                (32, 128),
                (32, 256),
                (32, 512),
                (64, 128),
                (64, 256),
                (64, 512),
                (96, 128),
                (96, 256),
            ]

    e2e_results: List[Dict[str, Any]] = []
    e2e_rows: List[List[str]] = []

    for B, max_len in e2e_configs:
        min_len = max(16, max_len // 16)
        seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
            B, min_len, max_len, distribution="skewed", device=dev
        )
        total_tokens = int(cu_seqlens[-1].item())
        pad_waste_pct = (1.0 - float(total_tokens) / float(B * actual_max)) * 100.0

        try:
            # Allocate padded inputs FIRST (isolated Phase 1 for clean Baseline E2E VRAM)
            x_prefill_b = torch.randn((B, actual_max, embed_dim), device=dev, dtype=torch.float16)
            pos_prefill = torch.arange(actual_max, device=dev, dtype=torch.int32).unsqueeze(0)
            valid_prefill_tok = pos_prefill < seqlens.unsqueeze(1)
            x_prefill_b = x_prefill_b * valid_prefill_tok.unsqueeze(-1).to(dtype=torch.float16)

            x_step_b = torch.randn((B, 1, embed_dim), device=dev, dtype=torch.float16)
            x_prefill_p_cpu = pack_varlen_tensor(x_prefill_b, cu_seqlens).cpu()
            x_step_p_cpu = x_step_b.squeeze(1).contiguous().cpu()

            total_kv_cap = actual_max + gen_steps
            total_kv_cap_aligned = align_up(total_kv_cap, 8)
            pos_dec = torch.arange(total_kv_cap_aligned, device=dev, dtype=torch.int32).unsqueeze(0)
            valid_dec = (pos_dec < seqlens.unsqueeze(1)) | (pos_dec >= actual_max)
            base_dec_mask = torch.zeros((B, total_kv_cap_aligned), device=dev, dtype=torch.float16)
            base_dec_mask.masked_fill_(~valid_dec, float("-inf"))
            full_dec_mask = torch.as_strided(
                base_dec_mask,
                size=(B, num_heads_q, 1, total_kv_cap_aligned),
                stride=(total_kv_cap_aligned, 0, 0, 1),
            )
            step_masks = [full_dec_mask[:, :, :, : actual_max + step + 1] for step in range(gen_steps)]
            offset_tensors = [
                torch.tensor(actual_max + step, device=dev, dtype=torch.int64) for step in range(gen_steps)
            ]
            del pos_prefill, valid_prefill_tok, pos_dec, valid_dec
            gc.collect()
            torch.cuda.empty_cache()

            # -----------------------------------------------------------------
            # PHASE 1: BASELINE (Batched SDPA) E2E
            # -----------------------------------------------------------------
            b_comp_ttft_ms: Optional[float] = None
            b_comp_tpot_ms: Optional[float] = None
            b_comp_e2e_ms: Optional[float] = None
            has_batched_comp = mod_batched_comp is not None and compiled_batched_step_fn is not None

            if has_batched_comp:
                def _batched_comp_ttft_fn():
                    with torch.no_grad():
                        return mod_batched_comp(x_prefill_b, is_causal=True, use_cache=True)

                def _batched_comp_e2e_fn():
                    with torch.no_grad():
                        _, (past_k, past_v) = mod_batched_comp(x_prefill_b, is_causal=True, use_cache=True)
                        for step in range(gen_steps):
                            _, past_k, past_v = compiled_batched_step_fn(
                                x_step_b, past_k, past_v, step_masks[step], offset_tensors[step]
                            )
                        return past_k, past_v

                b_comp_ttft_ms, _, _ = cuda_benchmark(
                    _batched_comp_ttft_fn, warmup=max(4, warmup // 3), rep=min(20, rep)
                )
                b_comp_e2e_ms, _, _ = cuda_benchmark(
                    _batched_comp_e2e_fn, warmup=max(4, warmup // 3), rep=min(15, rep), clear_l2=clear_l2
                )
                torch.cuda.empty_cache()
                _, b_vram_mb = measure_peak_memory_mb(_batched_comp_e2e_fn)

                with torch.no_grad():
                    _, (init_b_k, init_b_v) = mod_batched_comp(x_prefill_b, is_causal=True, use_cache=True)

                def _batched_comp_gen_fn():
                    with torch.no_grad():
                        past_k, past_v = init_b_k, init_b_v
                        for step in range(gen_steps):
                            _, past_k, past_v = compiled_batched_step_fn(
                                x_step_b, past_k, past_v, step_masks[step], offset_tensors[step]
                            )
                        return past_k, past_v

                b_comp_gen_ms, _, _ = cuda_benchmark(
                    _batched_comp_gen_fn, warmup=max(4, warmup // 3), rep=min(20, rep), clear_l2=clear_l2
                )
                b_comp_tpot_ms = b_comp_gen_ms / float(gen_steps)
                del init_b_k, init_b_v

            def _batched_eager_ttft_fn():
                with torch.no_grad():
                    return mod_batched(x_prefill_b, is_causal=True, use_cache=True)

            def _batched_eager_e2e_fn():
                with torch.no_grad():
                    _, b_cache = mod_batched(x_prefill_b, is_causal=True, use_cache=True)
                    for step in range(gen_steps):
                        _, b_cache = mod_batched(
                            x_step_b,
                            attn_mask=step_masks[step],
                            kv_cache=b_cache,
                            use_cache=True,
                            offset=offset_tensors[step],
                        )
                    return b_cache

            b_eager_ttft_ms, _, _ = cuda_benchmark(_batched_eager_ttft_fn, warmup=warmup // 2, rep=min(20, rep))
            b_eager_e2e_ms, _, _ = cuda_benchmark(
                _batched_eager_e2e_fn, warmup=4, rep=min(15, rep), clear_l2=clear_l2
            )
            if not has_batched_comp:
                torch.cuda.empty_cache()
                _, b_vram_mb = measure_peak_memory_mb(_batched_eager_e2e_fn)

            with torch.no_grad():
                _, init_b_eager_cache = mod_batched(x_prefill_b, is_causal=True, use_cache=True)

            def _batched_eager_gen_fn():
                with torch.no_grad():
                    b_cache = init_b_eager_cache
                    for step in range(gen_steps):
                        _, b_cache = mod_batched(
                            x_step_b,
                            attn_mask=step_masks[step],
                            kv_cache=b_cache,
                            use_cache=True,
                            offset=offset_tensors[step],
                        )
                    return b_cache

            b_eager_gen_ms, _, _ = cuda_benchmark(
                _batched_eager_gen_fn, warmup=4, rep=min(20, rep), clear_l2=clear_l2
            )
            b_eager_tpot_ms = b_eager_gen_ms / float(gen_steps)

            del x_prefill_b, x_step_b, full_dec_mask, step_masks, offset_tensors, init_b_eager_cache
            gc.collect()
            torch.cuda.empty_cache()

            # -----------------------------------------------------------------
            # PHASE 2: VARLEN PREFILL + DIRECT 1-BLOCK vs SPLIT-K FLASH DECODING
            # -----------------------------------------------------------------
            x_prefill_p = x_prefill_p_cpu.to(device=dev)
            x_step_p = x_step_p_cpu.to(device=dev)
            del x_prefill_p_cpu, x_step_p_cpu

            cos_t, sin_t = mod_varlen._resolve_rope_tables(actual_max + gen_steps + 32, dev, torch.float16)

            v_comp_ttft_ms: Optional[float] = None
            v_dir_comp_tpot_ms: Optional[float] = None
            v_spk_comp_tpot_ms: Optional[float] = None
            v_dir_comp_e2e_ms: Optional[float] = None
            v_spk_comp_e2e_ms: Optional[float] = None
            v_dir_vram_mb: float = 0.0
            v_spk_vram_mb: float = 0.0

            has_varlen_comp = (
                mod_varlen_comp is not None
                and compiled_varlen_step_direct is not None
                and compiled_varlen_step_splitk is not None
            )

            if has_varlen_comp:
                def _varlen_comp_ttft_fn():
                    with torch.no_grad():
                        return mod_varlen_comp(
                            x_prefill_p,
                            cu_seqlens_q=cu_seqlens,
                            max_seqlen_q=actual_max,
                            max_seqlen_k=actual_max,
                            is_causal=True,
                            use_cache=True,
                        )

                def _make_varlen_comp_e2e_fn(step_fn):
                    def _e2e_fn():
                        with torch.no_grad():
                            _, v_cache = mod_varlen_comp(
                                x_prefill_p,
                                cu_seqlens_q=cu_seqlens,
                                max_seqlen_q=actual_max,
                                max_seqlen_k=actual_max,
                                is_causal=True,
                                use_cache=True,
                            )
                            cur_k = v_cache.k_cache
                            cur_v = v_cache.v_cache
                            cur_cu = v_cache.cu_seqlens_k
                            for step in range(gen_steps):
                                _, cur_k, cur_v, cur_cu = step_fn(
                                    x_step_p,
                                    cur_k,
                                    cur_v,
                                    cur_cu,
                                    actual_max + step + 1,
                                    cos_t,
                                    sin_t,
                                    v_cache.offsets,
                                )
                            return cur_k, cur_v, cur_cu

                    return _e2e_fn

                v_comp_ttft_ms, _, _ = cuda_benchmark(
                    _varlen_comp_ttft_fn, warmup=max(4, warmup // 3), rep=min(20, rep)
                )

                # 2A. Direct 1-Block E2E + Peak VRAM
                _varlen_dir_e2e_fn = _make_varlen_comp_e2e_fn(compiled_varlen_step_direct)
                v_dir_comp_e2e_ms, _, _ = cuda_benchmark(
                    _varlen_dir_e2e_fn, warmup=max(4, warmup // 3), rep=min(15, rep), clear_l2=clear_l2
                )
                torch.cuda.empty_cache()
                _, v_dir_vram_mb = measure_peak_memory_mb(_varlen_dir_e2e_fn)

                # 2B. Split-K Flash Decoding E2E + Peak VRAM
                _varlen_spk_e2e_fn = _make_varlen_comp_e2e_fn(compiled_varlen_step_splitk)
                v_spk_comp_e2e_ms, _, _ = cuda_benchmark(
                    _varlen_spk_e2e_fn, warmup=max(4, warmup // 3), rep=min(15, rep), clear_l2=clear_l2
                )
                torch.cuda.empty_cache()
                _, v_spk_vram_mb = measure_peak_memory_mb(_varlen_spk_e2e_fn)

                # Shared prefill cache for measuring isolated generation TPOT
                with torch.no_grad():
                    _, init_v_cache = mod_varlen_comp(
                        x_prefill_p,
                        cu_seqlens_q=cu_seqlens,
                        max_seqlen_q=actual_max,
                        max_seqlen_k=actual_max,
                        is_causal=True,
                        use_cache=True,
                    )

                def _make_varlen_comp_gen_fn(step_fn):
                    def _gen_fn():
                        with torch.no_grad():
                            cur_k = init_v_cache.k_cache
                            cur_v = init_v_cache.v_cache
                            cur_cu = init_v_cache.cu_seqlens_k
                            for step in range(gen_steps):
                                _, cur_k, cur_v, cur_cu = step_fn(
                                    x_step_p,
                                    cur_k,
                                    cur_v,
                                    cur_cu,
                                    actual_max + step + 1,
                                    cos_t,
                                    sin_t,
                                    init_v_cache.offsets,
                                )
                            return cur_k, cur_v, cur_cu

                    return _gen_fn

                v_dir_comp_gen_ms, _, _ = cuda_benchmark(
                    _make_varlen_comp_gen_fn(compiled_varlen_step_direct),
                    warmup=max(4, warmup // 3),
                    rep=min(20, rep),
                    clear_l2=clear_l2,
                )
                v_spk_comp_gen_ms, _, _ = cuda_benchmark(
                    _make_varlen_comp_gen_fn(compiled_varlen_step_splitk),
                    warmup=max(4, warmup // 3),
                    rep=min(20, rep),
                    clear_l2=clear_l2,
                )
                v_dir_comp_tpot_ms = v_dir_comp_gen_ms / float(gen_steps)
                v_spk_comp_tpot_ms = v_spk_comp_gen_ms / float(gen_steps)
                del init_v_cache

            # Eager Varlen (Prefill + Direct 1-Block + Split-K)
            def _varlen_eager_ttft_fn():
                with torch.no_grad():
                    return mod_varlen(
                        x_prefill_p,
                        cu_seqlens_q=cu_seqlens,
                        max_seqlen_q=actual_max,
                        max_seqlen_k=actual_max,
                        is_causal=True,
                        use_cache=True,
                    )

            def _make_varlen_eager_e2e_fn(use_splitk_flag: bool):
                def _e2e_fn():
                    with torch.no_grad():
                        _, v_cache = mod_varlen(
                            x_prefill_p,
                            cu_seqlens_q=cu_seqlens,
                            max_seqlen_q=actual_max,
                            max_seqlen_k=actual_max,
                            is_causal=True,
                            use_cache=True,
                        )
                        for _ in range(gen_steps):
                            _, v_cache = mod_varlen(
                                x_step_p,
                                kv_cache=v_cache,
                                use_cache=True,
                                decode=True,
                                use_splitk=use_splitk_flag,
                            )
                        return v_cache

                return _e2e_fn

            v_eager_ttft_ms, _, _ = cuda_benchmark(_varlen_eager_ttft_fn, warmup=warmup // 2, rep=min(20, rep))
            _varlen_dir_eager_e2e_fn = _make_varlen_eager_e2e_fn(False)
            _varlen_spk_eager_e2e_fn = _make_varlen_eager_e2e_fn(True)
            v_dir_eager_e2e_ms, _, _ = cuda_benchmark(
                _varlen_dir_eager_e2e_fn, warmup=4, rep=min(15, rep), clear_l2=clear_l2
            )
            v_spk_eager_e2e_ms, _, _ = cuda_benchmark(
                _varlen_spk_eager_e2e_fn, warmup=4, rep=min(15, rep), clear_l2=clear_l2
            )
            if not has_varlen_comp:
                torch.cuda.empty_cache()
                _, v_dir_vram_mb = measure_peak_memory_mb(_varlen_dir_eager_e2e_fn)
                torch.cuda.empty_cache()
                _, v_spk_vram_mb = measure_peak_memory_mb(_varlen_spk_eager_e2e_fn)

            with torch.no_grad():
                _, init_v_eager_cache = mod_varlen(
                    x_prefill_p,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=actual_max,
                    max_seqlen_k=actual_max,
                    is_causal=True,
                    use_cache=True,
                )

            def _make_varlen_eager_gen_fn(use_splitk_flag: bool):
                def _gen_fn():
                    with torch.no_grad():
                        v_cache = VarlenKVCache(
                            init_v_eager_cache.k_cache,
                            init_v_eager_cache.v_cache,
                            init_v_eager_cache.cu_seqlens_k,
                            max_seqlen_k=init_v_eager_cache.max_seqlen_k,
                            offsets=init_v_eager_cache.offsets,
                        )
                        for _ in range(gen_steps):
                            _, v_cache = mod_varlen(
                                x_step_p,
                                kv_cache=v_cache,
                                use_cache=True,
                                decode=True,
                                use_splitk=use_splitk_flag,
                            )
                        return v_cache

                return _gen_fn

            v_dir_eager_gen_ms, _, _ = cuda_benchmark(
                _make_varlen_eager_gen_fn(False), warmup=4, rep=min(20, rep), clear_l2=clear_l2
            )
            v_spk_eager_gen_ms, _, _ = cuda_benchmark(
                _make_varlen_eager_gen_fn(True), warmup=4, rep=min(20, rep), clear_l2=clear_l2
            )
            v_dir_eager_tpot_ms = v_dir_eager_gen_ms / float(gen_steps)
            v_spk_eager_tpot_ms = v_spk_eager_gen_ms / float(gen_steps)

            fa1_ttft_ms: Optional[float] = None
            if mod_fa1 is not None:
                fa1_pos_ids = build_packed_rope_positions(cu_seqlens) if use_rope else None

                def _fa1_ttft_fn():
                    with torch.no_grad():
                        return mod_fa1(
                            x_prefill_p,
                            cu_seqlens_q=cu_seqlens,
                            max_seqlen_q=actual_max,
                            max_seqlen_k=actual_max,
                            is_causal=True,
                            pos_ids=fa1_pos_ids,
                        )

                fa1_ttft_ms, _, _ = cuda_benchmark(_fa1_ttft_fn, warmup=warmup // 2, rep=min(20, rep))
                del fa1_pos_ids

            del init_v_eager_cache, x_prefill_p, x_step_p
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WARN] Part B E2E skipped (OOM) for B={B}, max_len={max_len}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        # Metrics & Speedups
        has_comp = (
            b_comp_ttft_ms is not None
            and v_comp_ttft_ms is not None
            and b_comp_e2e_ms is not None
            and v_dir_comp_e2e_ms is not None
            and v_spk_comp_e2e_ms is not None
        )

        ref_b_ttft = b_comp_ttft_ms if has_comp else b_eager_ttft_ms
        ref_v_ttft = v_comp_ttft_ms if has_comp else v_eager_ttft_ms
        ref_b_tpot = b_comp_tpot_ms if has_comp else b_eager_tpot_ms
        ref_d_tpot = v_dir_comp_tpot_ms if has_comp else v_dir_eager_tpot_ms
        ref_s_tpot = v_spk_comp_tpot_ms if has_comp else v_spk_eager_tpot_ms
        ref_b_e2e = b_comp_e2e_ms if has_comp else b_eager_e2e_ms
        ref_d_e2e = v_dir_comp_e2e_ms if has_comp else v_dir_eager_e2e_ms
        ref_s_e2e = v_spk_comp_e2e_ms if has_comp else v_spk_eager_e2e_ms

        ttft_spd = ref_b_ttft / max(ref_v_ttft, 1e-6)
        tpot_spd_dir = ref_b_tpot / max(ref_d_tpot, 1e-6)
        tpot_spd_spk = ref_b_tpot / max(ref_s_tpot, 1e-6)
        e2e_spd_dir = ref_b_e2e / max(ref_d_e2e, 1e-6)
        e2e_spd_spk = ref_b_e2e / max(ref_s_e2e, 1e-6)

        e2e_results.append(
            {
                "B": B,
                "max_len": actual_max,
                "pad_waste_pct": pad_waste_pct,
                "batched_comp_ttft_ms": b_comp_ttft_ms,
                "varlen_comp_ttft_ms": v_comp_ttft_ms,
                "batched_comp_tpot_ms": b_comp_tpot_ms,
                "varlen_direct_comp_tpot_ms": v_dir_comp_tpot_ms,
                "varlen_splitk_comp_tpot_ms": v_spk_comp_tpot_ms,
                "batched_comp_e2e_ms": b_comp_e2e_ms,
                "varlen_direct_comp_e2e_ms": v_dir_comp_e2e_ms,
                "varlen_splitk_comp_e2e_ms": v_spk_comp_e2e_ms,
                "batched_ttft_ms": b_eager_ttft_ms,
                "varlen_ttft_ms": v_eager_ttft_ms,
                "batched_tpot_ms": b_eager_tpot_ms,
                "varlen_direct_tpot_ms": v_dir_eager_tpot_ms,
                "varlen_splitk_tpot_ms": v_spk_eager_tpot_ms,
                "batched_e2e_ms": b_eager_e2e_ms,
                "varlen_direct_e2e_ms": v_dir_eager_e2e_ms,
                "varlen_splitk_e2e_ms": v_spk_eager_e2e_ms,
                "varlen_direct_vram_mb": v_dir_vram_mb,
                "varlen_splitk_vram_mb": v_spk_vram_mb,
                "batched_vram_mb": b_vram_mb,
                "fa1_ttft_ms": fa1_ttft_ms,
            }
        )

        e2e_row = [
            str(B),
            str(actual_max),
            f"{pad_waste_pct:.1f}%",
            f"{ref_b_ttft:.2f}",
            f"{ref_v_ttft:.2f} ({ttft_spd:.2f}x)",
            f"{ref_b_tpot:.3f}",
            f"{ref_d_tpot:.3f} ({tpot_spd_dir:.2f}x)",
            f"{ref_s_tpot:.3f} ({tpot_spd_spk:.2f}x)",
            f"{e2e_spd_dir:.2f}x / {e2e_spd_spk:.2f}x",
            f"{v_dir_vram_mb:.1f} / {v_spk_vram_mb:.1f} / {b_vram_mb:.1f}",
        ]
        if HAS_FLASH_ATTN_1:
            e2e_row.insert(5, f"{fa1_ttft_ms:.2f}" if fa1_ttft_ms is not None else "N/A")
        e2e_rows.append(e2e_row)

    e2e_headers = [
        "B",
        "MaxL",
        "Pad%",
        "Base TTFT(ms)",
        "Ours TTFT(ms)",
        "Base TPOT(ms)",
        "Direct TPOT(ms)",
        "SplitK TPOT(ms)",
        "E2E Spd (Dir/SpK)",
        "E2E VRAM MB (Dir/SpK/Base)",
    ]
    if HAS_FLASH_ATTN_1:
        e2e_headers.insert(5, "FA1 TTFT(ms)")

    print("\n" + format_markdown_table(e2e_headers, e2e_rows))

    if save_plot:
        _save_inference_plot(decode_results, e2e_results, save_plot)

    return {"decode": decode_results, "e2e": e2e_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark for VarlenFlashGQA vs torch.compile Baseline.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--max-lens", type=int, nargs="+", default=[128, 256, 1024, 2048, 4096])
    parser.add_argument("--distributions", nargs="+", default=["skewed", "uniform"])
    parser.add_argument("--e2e-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--e2e-max-lens", type=int, nargs="+", default=None)
    parser.add_argument("--e2e-quick", action="store_true", help="Run a quick 5-case subset for Part B E2E.")
    parser.add_argument("--num-heads-q", type=int, default=32)
    parser.add_argument("--num-heads-kv", type=int, default=8)
    parser.add_argument("--gen-steps", type=int, default=32)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-clear-l2", action="store_true")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    custom_e2e_configs = None
    if args.e2e_batch_sizes is not None and args.e2e_max_lens is not None:
        custom_e2e_configs = [
            (b, l) for b in args.e2e_batch_sizes for l in args.e2e_max_lens
        ]

    run_inference_benchmark(
        decode_batch_sizes=args.batch_sizes,
        decode_max_lens=args.max_lens,
        decode_distributions=args.distributions,
        e2e_configs=custom_e2e_configs,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        gen_steps=args.gen_steps,
        use_rope=not args.no_rope,
        include_compile=not args.no_compile,
        clear_l2=not args.no_clear_l2,
        warmup=args.warmup,
        rep=args.rep,
        quick_e2e=args.e2e_quick,
        save_plot=args.save_plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
