"""Inference (Paged Decode L=1 & End-to-End Prefill+Generation) speed and E2E VRAM benchmark (v2 Eager vs torch.compile Baseline)."""

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
    align_up,
    build_additive_padding_mask,
    compute_decode_bandwidth_gbps,
    cuda_benchmark,
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
)


def _repeat_kv_batched(x: torch.Tensor, rep: int) -> torch.Tensor:
    if rep == 1:
        return x
    B, H_kv, L, D = x.shape
    return x.unsqueeze(2).expand(B, H_kv, rep, L, D).reshape(B, H_kv * rep, L, D)


def _compile_or_warn(target: Any, label: str, dynamic: bool = True):
    """Compile baseline target with TorchInductor; return None on failure."""
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

    # Subplot 1: Part A Decode L_q=1 Latency
    labels = [f"B={r['B']}\nL={r['max_len']}" for r in skewed_rows]
    x_idx = list(range(len(labels)))
    has_comp_dec = all(r.get("batched_comp_ms") is not None for r in skewed_rows)

    w2 = 0.36
    t_var = [r["varlen_dec_ms"] for r in skewed_rows]
    if has_comp_dec:
        t_base = [r["batched_comp_ms"] for r in skewed_rows]
        ax1.bar([i - w2 / 2 for i in x_idx], t_base, w2, label="Batched SDPA + Compile (Base)", color="#e63946")
        ax1.bar([i + w2 / 2 for i in x_idx], t_var, w2, label="Varlen Paged Eager (Ours v2)", color="#2a9d8f")
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_vs_comp", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + w2 / 2, t_var[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
            )
        ax1.set_title("Part A: Paged Decode L_q=1 Latency vs Compiled Baseline (Lower is Better)")
    else:
        t_base = [r["batched_dec_ms"] for r in skewed_rows]
        ax1.bar([i - w2 / 2 for i in x_idx], t_base, w2, label="Batched SDPA (Eager)", color="#6c757d")
        ax1.bar([i + w2 / 2 for i in x_idx], t_var, w2, label="Varlen Paged Eager (Ours v2)", color="#0d6efd")
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_eager", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + w2 / 2, t_var[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
            )
        ax1.set_title("Part A: Paged Decode L_q=1 Latency (Lower is Better)")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8)

    # Subplots 2, 3, 4: Part B End-to-End (Compiled Batched Baseline vs Pure Eager Varlen)
    if e2e_results:
        e2e_labels = [f"B={r['B']}\nL={r['max_len']}" for r in e2e_results]
        e_idx = list(range(len(e2e_labels)))
        has_comp_e2e = all(r.get("batched_comp_tpot_ms") is not None for r in e2e_results)

        v_tpot = [r["varlen_tpot_ms"] for r in e2e_results]
        if has_comp_e2e:
            b_tpot = [r["batched_comp_tpot_ms"] for r in e2e_results]
            ax2.set_title("Part B: E2E Generation TPOT (ms/token) vs Compiled Baseline")
        else:
            b_tpot = [r["batched_tpot_ms"] for r in e2e_results]
            ax2.set_title("Part B: E2E Generation TPOT (ms/token) [Eager]")

        ax2.bar([i - w2 / 2 for i in e_idx], b_tpot, w2, label="Baseline (Batched SDPA)", color="#e63946")
        ax2.bar([i + w2 / 2 for i in e_idx], v_tpot, w2, label="Varlen Fused Eager (Ours v2)", color="#2a9d8f")
        for i in e_idx:
            sp = b_tpot[i] / max(v_tpot[i], 1e-6)
            ax2.annotate(
                f"{sp:.2f}x",
                xy=(i + w2 / 2, v_tpot[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
            )
        ax2.set_ylabel("TPOT (ms / token)")
        ax2.set_xticks(e_idx)
        ax2.set_xticklabels(e2e_labels, rotation=25, ha="right", fontsize=7.5)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend(loc="upper left", fontsize=8)

        v_e2e = [r["varlen_e2e_ms"] for r in e2e_results]
        if has_comp_e2e:
            b_e2e = [r["batched_comp_e2e_ms"] for r in e2e_results]
            ax3.set_title("Part B: Total E2E Latency (Prefill + Gen) vs Compiled Baseline")
        else:
            b_e2e = [r["batched_e2e_ms"] for r in e2e_results]
            ax3.set_title("Part B: Total E2E Latency (Prefill + Gen) [Eager]")

        ax3.bar([i - w2 / 2 for i in e_idx], b_e2e, w2, label="Baseline (Batched SDPA)", color="#e63946")
        ax3.bar([i + w2 / 2 for i in e_idx], v_e2e, w2, label="Varlen Fused Eager (Ours v2)", color="#2a9d8f")
        for i in e_idx:
            sp = b_e2e[i] / max(v_e2e[i], 1e-6)
            ax3.annotate(
                f"{sp:.2f}x",
                xy=(i + w2 / 2, v_e2e[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
            )
        ax3.set_ylabel("Total E2E Latency (ms)")
        ax3.set_xticks(e_idx)
        ax3.set_xticklabels(e2e_labels, rotation=25, ha="right", fontsize=7.5)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        ax3.legend(loc="upper left", fontsize=8)

        b_vram = [r["batched_vram_mb"] for r in e2e_results]
        v_vram = [r["varlen_vram_mb"] for r in e2e_results]
        ax4.bar([i - w2 / 2 for i in e_idx], b_vram, w2, label="Baseline (Batched Padded)", color="#f4a261")
        ax4.bar([i + w2 / 2 for i in e_idx], v_vram, w2, label="Varlen Paged Eager (Ours v2)", color="#264653")
        for i in e_idx:
            save_pct = (1.0 - v_vram[i] / max(b_vram[i], 1e-6)) * 100.0
            ax4.annotate(
                f"-{save_pct:.0f}%",
                xy=(i + w2 / 2, v_vram[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
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
    Run v2 Paged Decode L_q=1 microbenchmark and End-to-End Prefill + Generation benchmark:
    Pure Eager C++ Fused Varlen (THD + Paged KV) vs Batched SDPA (+ torch.compile baseline).
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
    # PART A: Microbenchmark Paged Decode Attention L_q = 1 (Latency, BW, Tok/s)
    # =========================================================================
    print("\n" + "=" * 95)
    print(" PART A: Microbenchmark Paged Decode Attention L_q=1 (Cold L2 - Speed & Bandwidth)")
    print(f" Config: H_q={num_heads_q}, H_kv={num_heads_kv}, D={head_dim}, rope={use_rope}, clear_l2={clear_l2}")
    print(" Baseline Speed: Batched SDPA (SDPBackend.EFFICIENT_ATTENTION) + torch.compile(dynamic=False)")
    print(" Custom Kernel:  Varlen Paged Flash Decoding (Pure Eager, PAGE_SIZE=16)")
    print("=" * 95)

    eff_attn_ok = _verify_efficient_attention_active(
        batch_size=min(decode_batch_sizes) if decode_batch_sizes else 1,
        num_heads_q=num_heads_q,
        aligned_len=128,
        head_dim=head_dim,
        scale=scale,
        device=dev,
    )

    compiled_batched_dec = None
    if include_compile:
        def _c_batched_dec(q_b, k_exp, v_exp, pad_m, sc_val):
            return F.scaled_dot_product_attention(q_b, k_exp, v_exp, attn_mask=pad_m, scale=sc_val)

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
                    # 1. Batched SDPA Decode Baseline
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

                    # 2. Varlen Paged Decode (Pure Eager, (B, H_q, 64))
                    q_dec = torch.randn((B, num_heads_q, head_dim), device=dev, dtype=torch.float16)
                    k_raw = torch.randn((total_lk, num_heads_kv, head_dim), device=dev, dtype=torch.float16)
                    v_raw = torch.randn((total_lk, num_heads_kv, head_dim), device=dev, dtype=torch.float16)
                    kv_cache = VarlenKVCache.from_packed(
                        k_raw, v_raw, cu_seqlens, max_seqlen_k=actual_max, extra_pages_per_seq=0
                    )
                    del k_raw, v_raw

                    def _varlen_auto_eager_fn():
                        with torch.no_grad():
                            return varlen_flash_decode(
                                q_dec,
                                kv_cache.k_cache,
                                kv_cache.v_cache,
                                kv_cache.block_table,
                                kv_cache.cache_seqlens,
                                max_seqlen_k=actual_max,
                                cos=cos,
                                sin=sin,
                            )

                    v_eager_ms, _, _ = cuda_benchmark(
                        _varlen_auto_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                    )

                    del q_dec, kv_cache
                    gc.collect()
                    torch.cuda.empty_cache()

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"[WARN] Part A skipped (OOM) for dist={dist}, B={B}, max_len={max_len}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    raise

                speedup_vs_comp = b_comp_ms / max(v_eager_ms, 1e-6) if b_comp_ms is not None else None
                speedup_eager = b_eager_ms / max(v_eager_ms, 1e-6)

                bw_gbps = compute_decode_bandwidth_gbps(
                    total_lk, num_heads_kv, num_heads_q, B, head_dim, v_eager_ms
                )
                tok_per_s = B / (v_eager_ms * 1e-3)

                decode_results.append(
                    {
                        "dist": dist,
                        "B": B,
                        "max_len": actual_max,
                        "total_lk": total_lk,
                        "batched_comp_ms": b_comp_ms,
                        "speedup_vs_comp": speedup_vs_comp,
                        "batched_dec_ms": b_eager_ms,
                        "varlen_dec_ms": v_eager_ms,
                        "speedup_eager": speedup_eager,
                        "bw_gbps": bw_gbps,
                        "tok_per_s": tok_per_s,
                    }
                )

                b_comp_str = f"{b_comp_ms:.3f}" if b_comp_ms is not None else f"{b_eager_ms:.3f}"
                sp_comp_str = f"{speedup_vs_comp:.2f}x" if speedup_vs_comp is not None else f"{speedup_eager:.2f}x"

                decode_rows.append(
                    [
                        dist,
                        str(B),
                        str(actual_max),
                        str(total_lk),
                        b_comp_str,
                        f"{v_eager_ms:.3f}",
                        sp_comp_str,
                        f"{b_eager_ms:.3f}",
                        f"{speedup_eager:.2f}x",
                        f"{bw_gbps:.1f}",
                        f"{tok_per_s:,.0f}",
                    ]
                )

    decode_headers = [
        "Dist",
        "B",
        "MaxK",
        "TotalLk",
        "Batched+Comp(ms) [Base]",
        "Paged Eager(ms) [Ours]",
        "vs Comp",
        "Batched Eager",
        "vs Eager",
        "BW (GB/s)",
        "Tok/s",
    ]

    print("\n" + format_markdown_table(decode_headers, decode_rows))

    # =========================================================================
    # PART B: End-to-End Prefill (TTFT) + Generation (TPOT) + E2E Peak VRAM
    # Comparison: Baseline (Batched SDPA + Compile) vs Varlen Fused Eager (v2)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" PART B: End-to-End Prefill + {gen_steps}-Step Generation + Peak VRAM Benchmark (v2)")
    print(f" Config: H_q={num_heads_q}, H_kv={num_heads_kv}, embed_dim={embed_dim}")
    print(" Comparing: Baseline (Batched SDPA + Compile) vs Varlen Fused Eager (C++ Linear + Paged KV)")
    print("=" * 95)

    mod_varlen = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched = BatchedFlashGQABaselineSpeed(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched.load_state_dict(mod_varlen.state_dict())

    mod_batched_comp = _compile_or_warn(mod_batched, "BatchedFlashGQA") if include_compile else None

    compiled_batched_step_fn = None
    if include_compile:
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
                (1, 1024),
                (1, 2048),
                (4, 1024),
                (4, 2048),
                (8, 512),
                (8, 1024),
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

            # PHASE 1: BASELINE (Batched SDPA) E2E
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

            # PHASE 2: VARLEN FUSED EAGER PREFILL + PAGED KV DECODE (v2)
            x_prefill_p = x_prefill_p_cpu.to(device=dev)
            x_step_p = x_step_p_cpu.to(device=dev)
            del x_prefill_p_cpu, x_step_p_cpu

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

            def _varlen_eager_e2e_fn():
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
                        )
                    return v_cache

            v_eager_ttft_ms, _, _ = cuda_benchmark(_varlen_eager_ttft_fn, warmup=warmup // 2, rep=min(20, rep))
            v_eager_e2e_ms, _, _ = cuda_benchmark(
                _varlen_eager_e2e_fn, warmup=4, rep=min(15, rep), clear_l2=clear_l2
            )
            torch.cuda.empty_cache()
            _, v_vram_mb = measure_peak_memory_mb(_varlen_eager_e2e_fn)

            with torch.no_grad():
                _, init_v_eager_cache = mod_varlen(
                    x_prefill_p,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=actual_max,
                    max_seqlen_k=actual_max,
                    is_causal=True,
                    use_cache=True,
                )
                # Ensure pool headroom covers gen_steps without resizing inside timed iterations
                while (
                    init_v_eager_cache.max_seqlen_k + gen_steps
                    > init_v_eager_cache.max_pages_per_seq * init_v_eager_cache.PAGE_SIZE
                ):
                    saved_max = init_v_eager_cache.max_seqlen_k
                    init_v_eager_cache.max_seqlen_k = (
                        init_v_eager_cache.max_pages_per_seq * init_v_eager_cache.PAGE_SIZE
                    )
                    init_v_eager_cache.ensure_capacity_for_append()
                    init_v_eager_cache.max_seqlen_k = saved_max

                init_seqlens_gpu = init_v_eager_cache.cache_seqlens.clone()
                init_seqlens_cpu = init_v_eager_cache.seqlens_cpu
                init_max_seqlen_k = init_v_eager_cache.max_seqlen_k

            def _varlen_eager_gen_fn():
                with torch.no_grad():
                    # Reset logical length counters in-place so every rep measures the exact [L..L+gen_steps] window
                    init_v_eager_cache.cache_seqlens.copy_(init_seqlens_gpu)
                    init_v_eager_cache._seqlens_cpu = list(init_seqlens_cpu)
                    init_v_eager_cache.max_seqlen_k = init_max_seqlen_k
                    for _ in range(gen_steps):
                        mod_varlen(
                            x_step_p,
                            kv_cache=init_v_eager_cache,
                            decode=True,
                        )
                    return init_v_eager_cache

            v_eager_gen_ms, _, _ = cuda_benchmark(
                _varlen_eager_gen_fn, warmup=4, rep=min(20, rep), clear_l2=clear_l2
            )
            v_eager_tpot_ms = v_eager_gen_ms / float(gen_steps)

            del init_v_eager_cache, init_seqlens_gpu, x_prefill_p, x_step_p
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WARN] Part B E2E skipped (OOM) for B={B}, max_len={max_len}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        has_comp = b_comp_ttft_ms is not None and b_comp_tpot_ms is not None and b_comp_e2e_ms is not None
        ref_b_ttft = b_comp_ttft_ms if has_comp else b_eager_ttft_ms
        ref_b_tpot = b_comp_tpot_ms if has_comp else b_eager_tpot_ms
        ref_b_e2e = b_comp_e2e_ms if has_comp else b_eager_e2e_ms

        ttft_spd = ref_b_ttft / max(v_eager_ttft_ms, 1e-6)
        tpot_spd = ref_b_tpot / max(v_eager_tpot_ms, 1e-6)
        e2e_spd = ref_b_e2e / max(v_eager_e2e_ms, 1e-6)
        vram_save_pct = (1.0 - v_vram_mb / max(b_vram_mb, 1e-6)) * 100.0

        e2e_results.append(
            {
                "B": B,
                "max_len": actual_max,
                "pad_waste_pct": pad_waste_pct,
                "batched_comp_ttft_ms": b_comp_ttft_ms,
                "batched_comp_tpot_ms": b_comp_tpot_ms,
                "batched_comp_e2e_ms": b_comp_e2e_ms,
                "batched_ttft_ms": b_eager_ttft_ms,
                "varlen_ttft_ms": v_eager_ttft_ms,
                "batched_tpot_ms": b_eager_tpot_ms,
                "varlen_tpot_ms": v_eager_tpot_ms,
                "batched_e2e_ms": b_eager_e2e_ms,
                "varlen_e2e_ms": v_eager_e2e_ms,
                "varlen_vram_mb": v_vram_mb,
                "batched_vram_mb": b_vram_mb,
            }
        )

        e2e_rows.append(
            [
                str(B),
                str(actual_max),
                f"{pad_waste_pct:.1f}%",
                f"{ref_b_ttft:.2f}",
                f"{v_eager_ttft_ms:.2f} ({ttft_spd:.2f}x)",
                f"{ref_b_tpot:.3f}",
                f"{v_eager_tpot_ms:.3f} ({tpot_spd:.2f}x)",
                f"{ref_b_e2e:.2f}",
                f"{v_eager_e2e_ms:.2f} ({e2e_spd:.2f}x)",
                f"{v_vram_mb:.1f} / {b_vram_mb:.1f} (-{vram_save_pct:.0f}%)",
            ]
        )

    e2e_headers = [
        "B",
        "MaxL",
        "Pad%",
        "Base TTFT(ms)",
        "Ours TTFT(ms)",
        "Base TPOT(ms)",
        "Ours TPOT(ms)",
        "Base E2E(ms)",
        "Ours E2E(ms)",
        "E2E VRAM MB (Ours/Base)",
    ]

    print("\n" + format_markdown_table(e2e_headers, e2e_rows))

    if save_plot:
        _save_inference_plot(decode_results, e2e_results, save_plot)

    return {"decode": decode_results, "e2e": e2e_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark for Pure Eager v2 VarlenFlashGQA vs torch.compile Baseline.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--max-lens", type=int, nargs="+", default=[128, 256, 1024, 2048, 4096])
    parser.add_argument("--distributions", nargs="+", default=["skewed", "uniform"])
    parser.add_argument("--e2e-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--e2e-max-lens", type=int, nargs="+", default=None)
    parser.add_argument("--e2e-quick", action="store_true", help="Run a quick 5-config E2E subset for Part B E2E.")
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
