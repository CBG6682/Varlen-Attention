"""Inference (Decode L=1 & End-to-End Prefill+Generation) speed and VRAM benchmark with torch.compile baseline."""

import argparse
import gc
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tests.utils import (
    build_padding_mask,
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


def _compile_or_warn(target: Any, label: str):
    """Compile target with TorchInductor; return None on failure."""
    try:
        return torch.compile(target, backend="inductor", dynamic=True)
    except Exception as exc:
        print(f"[WARN] Could not torch.compile({label}): {exc}")
        return None


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    labels = [f"B={r['B']}\nL={r['max_len']}" for r in skewed_rows]
    x_idx = list(range(len(labels)))
    width = 0.35

    has_comp_dec = all(
        r.get("batched_comp_ms") is not None and r.get("varlen_comp_ms") is not None
        for r in skewed_rows
    )

    if has_comp_dec:
        t_base = [r["batched_comp_ms"] for r in skewed_rows]
        t_varlen = [r["varlen_comp_ms"] for r in skewed_rows]
        ax1.bar(
            [i - width / 2 for i in x_idx],
            t_base,
            width,
            label="Batched SDPA Decode + torch.compile (Baseline)",
            color="#e63946",
        )
        ax1.bar(
            [i + width / 2 for i in x_idx],
            t_varlen,
            width,
            label="Varlen Split-K Decode + torch.compile (Ours)",
            color="#2a9d8f",
        )
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_comp", 1.0)
            ax1.annotate(
                f"{sp:.2f}x\n({r['bw_gbps']:.0f} GB/s)",
                xy=(i + width / 2, t_varlen[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
        ax1.set_title("Decode L_q=1 Latency vs torch.compile Baseline (Lower is Better)")
    else:
        t_base = [r["batched_dec_ms"] for r in skewed_rows]
        t_varlen = [r["varlen_dec_ms"] for r in skewed_rows]
        ax1.bar([i - width / 2 for i in x_idx], t_base, width, label="Batched SDPA Decode (Eager)", color="#6c757d")
        ax1.bar([i + width / 2 for i in x_idx], t_varlen, width, label="Varlen Split-K Decode (Eager)", color="#0d6efd")
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_eager", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + width / 2, t_varlen[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
        ax1.set_title("Decode L_q=1 Latency (Lower is Better)")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=0, ha="center")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Plot E2E TPOT
    if e2e_results:
        # Select representative cases (max 10 to fit cleanly)
        selected_e2e = e2e_results[:10] if len(e2e_results) > 10 else e2e_results
        e2e_labels = [f"B={r['B']}\nL={r['max_len']}" for r in selected_e2e]
        e_idx = list(range(len(e2e_labels)))

        has_comp_e2e = all(
            r.get("batched_comp_tpot_ms") is not None and r.get("varlen_comp_tpot_ms") is not None
            for r in selected_e2e
        )

        if has_comp_e2e:
            b_tpot = [r["batched_comp_tpot_ms"] for r in selected_e2e]
            v_tpot = [r["varlen_comp_tpot_ms"] for r in selected_e2e]
            ax2.bar(
                [i - width / 2 for i in e_idx],
                b_tpot,
                width,
                label="Batched SDPA TPOT + torch.compile (Baseline)",
                color="#f4a261",
            )
            ax2.bar(
                [i + width / 2 for i in e_idx],
                v_tpot,
                width,
                label="Varlen GQA TPOT + torch.compile (Ours)",
                color="#264653",
            )
            for i, r in enumerate(selected_e2e):
                sp = r.get("tpot_speedup_comp", 1.0)
                ax2.annotate(
                    f"{sp:.2f}x",
                    xy=(i + width / 2, v_tpot[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )
            ax2.set_title("E2E Generation TPOT vs torch.compile Baseline (Lower is Better)")
        else:
            b_tpot = [r["batched_tpot_ms"] for r in selected_e2e]
            v_tpot = [r["varlen_tpot_ms"] for r in selected_e2e]
            ax2.bar([i - width / 2 for i in e_idx], b_tpot, width, label="Batched SDPA TPOT (Eager)", color="#fd7e14")
            ax2.bar([i + width / 2 for i in e_idx], v_tpot, width, label="Varlen GQA TPOT (Eager)", color="#20c997")
            ax2.set_title("E2E Generation TPOT (ms/token)")

        ax2.set_ylabel("Time per Output Token (ms)")
        ax2.set_xticks(e_idx)
        ax2.set_xticklabels(e2e_labels, rotation=0, ha="center")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved inference benchmark plot -> {save_path}")


def run_inference_benchmark(
    decode_batch_sizes: Sequence[int] = (1, 8, 32),
    decode_max_lens: Sequence[int] = (256, 1024, 2048, 4096),
    decode_distributions: Sequence[str] = ("skewed", "uniform"),
    e2e_configs: Optional[Sequence[Tuple[int, int]]] = None,
    num_heads_q: int = 16,
    num_heads_kv: int = 4,
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
    Run Decode L_q=1 kernel benchmark and expanded End-to-End Prefill + Multi-Step Generation benchmark.

    Establishes `torch.compile` (TorchInductor) as the primary speed baseline and compares
    `VarlenFlashGQA` + `torch.compile` against compiled batched SDPA.
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

    # Calculate max possible seq len for RoPE tables
    all_lens = list(decode_max_lens)
    if e2e_configs:
        all_lens.extend([cfg[1] for cfg in e2e_configs])
    else:
        all_lens.extend([256, 512, 1024, 2048])
    max_table_len = max(all_lens) + gen_steps + 128

    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_table_len, device=dev) if use_rope else None
    cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16) if rope is not None else (None, None)

    # =========================================================================
    # PART A: Microbenchmark Decode Attention L_q = 1 (Varlen vs Batched SDPA)
    # =========================================================================
    print("\n" + "=" * 90)
    print(" PART A: Microbenchmark Decode Attention L_q=1 (Cold L2)")
    print(f" Config: H_q={num_heads_q}, H_kv={num_heads_kv}, D={head_dim}, rope={use_rope}, clear_l2={clear_l2}")
    print(f" Baseline Speed: Batched SDPA Decode + torch.compile(backend='inductor', dynamic=True)")
    print(f" Custom Kernel:  Varlen Split-K Decode + torch.compile(backend='inductor', dynamic=True)")
    print("=" * 90)

    # Compile Part A ops
    compiled_varlen_dec = None
    compiled_batched_dec = None
    if include_compile:
        @_compile_or_warn(
            lambda q, k, v, cu, mx, c, s: varlen_flash_decode(q, k, v, cu, max_seqlen_k=mx, cos=c, sin=s),
            "varlen_flash_decode",
        )
        def _c_varlen_dec(q_in, k_in, v_in, cu_in, max_k, c_in, s_in):
            return varlen_flash_decode(q_in, k_in, v_in, cu_in, max_seqlen_k=max_k, cos=c_in, sin=s_in)

        compiled_varlen_dec = _c_varlen_dec

        @_compile_or_warn(
            lambda q, k, v, mask, sc, rep_k: F.scaled_dot_product_attention(
                q, _repeat_kv_batched(k, rep_k), _repeat_kv_batched(v, rep_k), attn_mask=mask, scale=sc
            ),
            "batched_sdpa_decode",
        )
        def _c_batched_dec(q_b, k_b, v_b, pad_m, sc_val, rep_val):
            k_exp = _repeat_kv_batched(k_b, rep_val)
            v_exp = _repeat_kv_batched(v_b, rep_val)
            return F.scaled_dot_product_attention(q_b, k_exp, v_exp, attn_mask=pad_m, scale=sc_val)

        compiled_batched_dec = _c_batched_dec

    decode_results: List[Dict[str, Any]] = []
    decode_rows: List[List[str]] = []

    for dist in decode_distributions:
        for B in decode_batch_sizes:
            for max_len in decode_max_lens:
                min_len = max(32, max_len // 16)
                seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
                    B, min_len, max_len, distribution=dist, device=dev
                )
                total_lk = int(cu_seqlens[-1].item())

                q_dec = torch.randn((num_heads_q, B, head_dim), device=dev, dtype=torch.float16)
                k_packed = torch.randn((num_heads_kv, total_lk, head_dim), device=dev, dtype=torch.float16)
                v_packed = torch.randn((num_heads_kv, total_lk, head_dim), device=dev, dtype=torch.float16)

                q_batched = q_dec.transpose(0, 1).unsqueeze(2).contiguous()  # (B, H_q, 1, D)
                k_batched = torch.randn((B, num_heads_kv, actual_max, head_dim), device=dev, dtype=torch.float16)
                v_batched = torch.randn((B, num_heads_kv, actual_max, head_dim), device=dev, dtype=torch.float16)
                pad_mask = build_padding_mask(cu_seqlens, actual_max)

                # Pre-apply RoPE to batched query for fairness
                q_batched_roped = rope(q_batched, offset=actual_max - 1) if rope is not None else q_batched

                # 1. Batched SDPA Compiled (Primary Baseline Speed)
                b_comp_ms: Optional[float] = None
                if compiled_batched_dec is not None:
                    def _batched_dec_comp_fn():
                        with torch.no_grad():
                            return compiled_batched_dec(
                                q_batched_roped, k_batched, v_batched, pad_mask, scale, rep_gqa
                            )

                    b_comp_ms, _, _ = cuda_benchmark(
                        _batched_dec_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                    )

                # 2. Varlen Decode Compiled (Custom Kernel)
                v_comp_ms: Optional[float] = None
                if compiled_varlen_dec is not None:
                    def _varlen_dec_comp_fn():
                        with torch.no_grad():
                            return compiled_varlen_dec(
                                q_dec, k_packed, v_packed, cu_seqlens, actual_max, cos, sin
                            )

                    v_comp_ms, _, _ = cuda_benchmark(
                        _varlen_dec_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                    )

                # 3. Batched SDPA Eager
                def _batched_dec_eager_fn():
                    with torch.no_grad():
                        k_exp = _repeat_kv_batched(k_batched, rep_gqa)
                        v_exp = _repeat_kv_batched(v_batched, rep_gqa)
                        return F.scaled_dot_product_attention(
                            q_batched_roped, k_exp, v_exp, attn_mask=pad_mask, scale=scale
                        )

                b_eager_ms, _, _ = cuda_benchmark(
                    _batched_dec_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                )

                # 4. Varlen Decode Eager
                def _varlen_dec_eager_fn():
                    with torch.no_grad():
                        return varlen_flash_decode(
                            q_dec, k_packed, v_packed, cu_seqlens, max_seqlen_k=actual_max, cos=cos, sin=sin
                        )

                v_eager_ms, _, _ = cuda_benchmark(
                    _varlen_dec_eager_fn, warmup=warmup, rep=rep, clear_l2=clear_l2
                )

                # Speedups
                speedup_comp = (
                    b_comp_ms / max(v_comp_ms, 1e-6)
                    if b_comp_ms is not None and v_comp_ms is not None
                    else None
                )
                speedup_eager = b_eager_ms / max(v_eager_ms, 1e-6)

                best_v_ms = v_comp_ms if v_comp_ms is not None else v_eager_ms
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
                        "varlen_comp_ms": v_comp_ms,
                        "speedup_comp": speedup_comp,
                        "batched_dec_ms": b_eager_ms,
                        "varlen_dec_ms": v_eager_ms,
                        "speedup_eager": speedup_eager,
                        "bw_gbps": bw_gbps,
                        "tok_per_s": tok_per_s,
                    }
                )

                b_comp_str = f"{b_comp_ms:.3f}" if b_comp_ms is not None else "N/A"
                v_comp_str = f"{v_comp_ms:.3f}" if v_comp_ms is not None else "N/A"
                sp_comp_str = f"{speedup_comp:.2f}x" if speedup_comp is not None else "N/A"

                decode_rows.append(
                    [
                        dist,
                        str(B),
                        str(actual_max),
                        str(total_lk),
                        b_comp_str,
                        v_comp_str,
                        sp_comp_str,
                        f"{b_eager_ms:.3f}",
                        f"{v_eager_ms:.3f}",
                        f"{speedup_eager:.2f}x",
                        f"{bw_gbps:.1f}",
                        f"{tok_per_s:,.0f}",
                    ]
                )

                del q_dec, k_packed, v_packed, q_batched, k_batched, v_batched, pad_mask
                gc.collect()
                torch.cuda.empty_cache()

    print(
        "\n"
        + format_markdown_table(
            [
                "Dist",
                "B",
                "MaxK",
                "TotalLk",
                "Batched+Comp(ms) [Baseline]",
                "Varlen+Comp(ms) [Ours]",
                "Comp Speedup",
                "Batched Eager",
                "Varlen Eager",
                "Eager Speedup",
                "BW (GB/s)",
                "Tok/s",
            ],
            decode_rows,
        )
    )

    # =========================================================================
    # PART B: End-to-End Prefill (TTFT) + Multi-Step Generation (TPOT & VRAM)
    # =========================================================================
    print("\n" + "=" * 90)
    print(f" PART B: End-to-End Prefill + {gen_steps}-Step Generation Benchmark (Expanded Evaluation Matrix)")
    print(f" Baseline Speed: BatchedFlashGQABaselineSpeed + torch.compile")
    print(f" Custom Kernel:  VarlenFlashGQA + torch.compile")
    print("=" * 90)

    mod_varlen = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched = BatchedFlashGQABaselineSpeed(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched.load_state_dict(mod_varlen.state_dict())

    # Compile Prefill modules
    mod_varlen_comp = _compile_or_warn(mod_varlen, "VarlenFlashGQA") if include_compile else None
    mod_batched_comp = _compile_or_warn(mod_batched, "BatchedFlashGQA") if include_compile else None

    # Compile single-step decode execution for fair E2E generation loop comparison
    compiled_varlen_step_fn = None
    compiled_batched_step_fn = None
    if include_compile:
        @_compile_or_warn(
            lambda x_st, k_c, v_c, cu_c, mx_k, c_t, s_t, off_t: mod_varlen.out_proj(
                varlen_flash_decode(
                    mod_varlen._project_qkv_decode(x_st)[0],
                    k_c,
                    v_c,
                    cu_c,
                    max_seqlen_k=mx_k,
                    cos=c_t,
                    sin=s_t,
                    offsets=off_t,
                ).transpose(0, 1).contiguous().view(x_st.shape[0], -1)
            ),
            "varlen_decode_step",
        )
        def _c_varlen_step(x_step, k_cache, v_cache, cu_seqlens_k, max_k, cos_t, sin_t, offsets_t):
            q, new_k, new_v = mod_varlen._project_qkv_decode(x_step)
            attn_out = varlen_flash_decode(
                q, k_cache, v_cache, cu_seqlens_k, max_seqlen_k=max_k, cos=cos_t, sin=sin_t, offsets=offsets_t
            )
            out_flat = attn_out.transpose(0, 1).contiguous().view(x_step.shape[0], -1)
            return mod_varlen.out_proj(out_flat), new_k, new_v

        compiled_varlen_step_fn = _c_varlen_step

        @_compile_or_warn(
            lambda x_st, p_k, p_v, off_v: mod_batched.out_proj(
                F.scaled_dot_product_attention(
                    (mod_batched.rope(
                        mod_batched.qkv_proj(x_st).view(x_st.shape[0], 1, -1, mod_batched.head_dim).transpose(1, 2).split([mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1)[0],
                        offset=off_v
                    ) if mod_batched.rope is not None else mod_batched.qkv_proj(x_st).view(x_st.shape[0], 1, -1, mod_batched.head_dim).transpose(1, 2).split([mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1)[0]),
                    _repeat_kv_batched(torch.cat([p_k, mod_batched.qkv_proj(x_st).view(x_st.shape[0], 1, -1, mod_batched.head_dim).transpose(1, 2).split([mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1)[1]], dim=2), rep_gqa),
                    _repeat_kv_batched(torch.cat([p_v, mod_batched.qkv_proj(x_st).view(x_st.shape[0], 1, -1, mod_batched.head_dim).transpose(1, 2).split([mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1)[2]], dim=2), rep_gqa),
                    scale=scale,
                ).transpose(1, 2).contiguous().view(x_st.shape[0], 1, -1)
            ),
            "batched_decode_step",
        )
        def _c_batched_step(x_step, past_k, past_v, offset_val):
            qkv = mod_batched.qkv_proj(x_step)
            B_sz = x_step.shape[0]
            qkv = qkv.view(B_sz, 1, mod_batched.num_heads_q + 2 * mod_batched.num_heads_kv, mod_batched.head_dim).transpose(1, 2)
            q, k, v = qkv.split([mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1)
            if mod_batched.rope is not None:
                q = mod_batched.rope(q, offset=offset_val)
                k = mod_batched.rope(k, offset=offset_val)
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
            k_exp = _repeat_kv_batched(k_all, rep_gqa)
            v_exp = _repeat_kv_batched(v_all, rep_gqa)
            out = F.scaled_dot_product_attention(q, k_exp, v_exp, scale=scale)
            out_flat = out.transpose(1, 2).contiguous().view(B_sz, 1, -1)
            return mod_batched.out_proj(out_flat), k_all, v_all

        compiled_batched_step_fn = _c_batched_step

    # Default expanded evaluation matrix covering small/medium/large batches and contexts
    if e2e_configs is None:
        if quick_e2e:
            e2e_configs = [
                (1, 256),
                (1, 1024),
                (4, 1024),
                (8, 1024),
                (16, 512),
            ]
        else:
            e2e_configs = [
                # Single-stream interactive serving (B=1)
                (1, 256),
                (1, 512),
                (1, 1024),
                (1, 2048),
                # Small concurrent batch (B=4)
                (4, 256),
                (4, 512),
                (4, 1024),
                (4, 2048),
                # Medium enterprise serving (B=8)
                (8, 256),
                (8, 512),
                (8, 1024),
                (8, 2048),
                # High throughput serving (B=16)
                (16, 256),
                (16, 512),
                (16, 1024),
                # Maximum concurrency (B=32)
                (32, 256),
                (32, 512),
            ]

    e2e_results: List[Dict[str, Any]] = []
    e2e_rows: List[List[str]] = []

    for B, max_len in e2e_configs:
        min_len = max(32, max_len // 16)
        seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
            B, min_len, max_len, distribution="skewed", device=dev
        )
        total_tokens = int(cu_seqlens[-1].item())
        pad_waste_pct = (1.0 - float(total_tokens) / float(B * actual_max)) * 100.0

        x_prefill_b = torch.randn((B, actual_max, embed_dim), device=dev, dtype=torch.float16)
        x_prefill_p = pack_varlen_tensor(x_prefill_b, cu_seqlens)
        x_step_p = torch.randn((B, embed_dim), device=dev, dtype=torch.float16)
        x_step_b = x_step_p.unsqueeze(1)  # (B, 1, embed_dim)

        cos_t, sin_t = mod_varlen._resolve_rope_tables(actual_max + gen_steps + 32, dev, torch.float16)

        # ---------------------------------------------------------------------
        # 1. COMPILED E2E (Batched Baseline vs Varlen Custom Kernel)
        # ---------------------------------------------------------------------
        b_comp_ttft_ms: Optional[float] = None
        b_comp_tpot_ms: Optional[float] = None
        b_comp_e2e_ms: Optional[float] = None
        v_comp_ttft_ms: Optional[float] = None
        v_comp_tpot_ms: Optional[float] = None
        v_comp_e2e_ms: Optional[float] = None

        if mod_batched_comp is not None and compiled_batched_step_fn is not None:
            def _batched_comp_ttft_fn():
                with torch.no_grad():
                    return mod_batched_comp(x_prefill_b, is_causal=True)

            def _batched_comp_e2e_fn():
                with torch.no_grad():
                    qkv = mod_batched.qkv_proj(x_prefill_b)
                    B_sz, L_sz, _ = x_prefill_b.shape
                    qkv = qkv.view(B_sz, L_sz, mod_batched.num_heads_q + 2 * mod_batched.num_heads_kv, mod_batched.head_dim).transpose(1, 2)
                    q, k, v = qkv.split([mod_batched.num_heads_q, mod_batched.num_heads_kv, mod_batched.num_heads_kv], dim=1)
                    if mod_batched.rope is not None:
                        q = mod_batched.rope(q, offset=0)
                        k = mod_batched.rope(k, offset=0)
                    past_k, past_v = k, v
                    for step in range(gen_steps):
                        _, past_k, past_v = compiled_batched_step_fn(x_step_b, past_k, past_v, actual_max + step)
                    return past_k, past_v

            b_comp_ttft_ms, _, _ = cuda_benchmark(_batched_comp_ttft_fn, warmup=max(4, warmup // 3), rep=min(25, rep))
            b_comp_e2e_ms, _, _ = cuda_benchmark(_batched_comp_e2e_fn, warmup=max(4, warmup // 3), rep=min(20, rep), clear_l2=clear_l2)
            b_comp_tpot_ms = max(0.0, b_comp_e2e_ms - b_comp_ttft_ms) / float(gen_steps)

        if mod_varlen_comp is not None and compiled_varlen_step_fn is not None:
            def _varlen_comp_ttft_fn():
                with torch.no_grad():
                    return mod_varlen_comp(
                        x_prefill_p,
                        cu_seqlens_q=cu_seqlens,
                        max_seqlen_q=actual_max,
                        max_seqlen_k=actual_max,
                        is_causal=True,
                    )

            def _varlen_comp_e2e_fn():
                with torch.no_grad():
                    q, k, v, cu_k, _ = mod_varlen._project_qkv_packed(x_prefill_p, cu_seqlens)
                    v_cache = VarlenKVCache.from_packed(k, v, cu_k, actual_max)
                    for step in range(gen_steps):
                        _, new_k, new_v = compiled_varlen_step_fn(
                            x_step_p,
                            v_cache.k_cache,
                            v_cache.v_cache,
                            v_cache.cu_seqlens_k,
                            v_cache.max_seqlen_k,
                            cos_t,
                            sin_t,
                            v_cache.offsets,
                        )
                        v_cache.append(new_k, new_v)
                    return v_cache

            v_comp_ttft_ms, _, _ = cuda_benchmark(_varlen_comp_ttft_fn, warmup=max(4, warmup // 3), rep=min(25, rep))
            v_comp_e2e_ms, _, _ = cuda_benchmark(_varlen_comp_e2e_fn, warmup=max(4, warmup // 3), rep=min(20, rep), clear_l2=clear_l2)
            v_comp_tpot_ms = max(0.0, v_comp_e2e_ms - v_comp_ttft_ms) / float(gen_steps)

        # ---------------------------------------------------------------------
        # 2. EAGER E2E (Batched Baseline vs Varlen Custom Kernel)
        # ---------------------------------------------------------------------
        def _batched_eager_ttft_fn():
            with torch.no_grad():
                return mod_batched(x_prefill_b, is_causal=True, use_cache=True)

        def _batched_eager_e2e_fn():
            with torch.no_grad():
                _, b_cache = mod_batched(x_prefill_b, is_causal=True, use_cache=True)
                for step in range(gen_steps):
                    _, b_cache = mod_batched(
                        x_step_b, kv_cache=b_cache, use_cache=True, offset=actual_max + step
                    )
                return b_cache

        b_eager_ttft_ms, _, _ = cuda_benchmark(_batched_eager_ttft_fn, warmup=warmup // 2, rep=min(25, rep))
        _, b_vram_mb = measure_peak_memory_mb(_batched_eager_e2e_fn)
        b_eager_e2e_ms, _, _ = cuda_benchmark(_batched_eager_e2e_fn, warmup=4, rep=min(20, rep), clear_l2=clear_l2)
        b_eager_tpot_ms = max(0.0, b_eager_e2e_ms - b_eager_ttft_ms) / float(gen_steps)

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
                    _, v_cache = mod_varlen(x_step_p, kv_cache=v_cache, use_cache=True, decode=True)
                return v_cache

        v_eager_ttft_ms, _, _ = cuda_benchmark(_varlen_eager_ttft_fn, warmup=warmup // 2, rep=min(25, rep))
        _, v_vram_mb = measure_peak_memory_mb(_varlen_eager_e2e_fn)
        v_eager_e2e_ms, _, _ = cuda_benchmark(_varlen_eager_e2e_fn, warmup=4, rep=min(20, rep), clear_l2=clear_l2)
        v_eager_tpot_ms = max(0.0, v_eager_e2e_ms - v_eager_ttft_ms) / float(gen_steps)

        # Speedups
        has_comp = (
            b_comp_ttft_ms is not None
            and v_comp_ttft_ms is not None
            and b_comp_e2e_ms is not None
            and v_comp_e2e_ms is not None
        )
        ttft_sp_comp = b_comp_ttft_ms / max(v_comp_ttft_ms, 1e-6) if has_comp else None
        tpot_sp_comp = b_comp_tpot_ms / max(v_comp_tpot_ms, 1e-6) if has_comp else None
        e2e_sp_comp = b_comp_e2e_ms / max(v_comp_e2e_ms, 1e-6) if has_comp else None

        ttft_sp_eager = b_eager_ttft_ms / max(v_eager_ttft_ms, 1e-6)
        tpot_sp_eager = b_eager_tpot_ms / max(v_eager_tpot_ms, 1e-6)
        e2e_sp_eager = b_eager_e2e_ms / max(v_eager_e2e_ms, 1e-6)

        e2e_results.append(
            {
                "B": B,
                "max_len": actual_max,
                "pad_waste_pct": pad_waste_pct,
                "batched_comp_ttft_ms": b_comp_ttft_ms,
                "varlen_comp_ttft_ms": v_comp_ttft_ms,
                "ttft_speedup_comp": ttft_sp_comp,
                "batched_comp_tpot_ms": b_comp_tpot_ms,
                "varlen_comp_tpot_ms": v_comp_tpot_ms,
                "tpot_speedup_comp": tpot_sp_comp,
                "batched_comp_e2e_ms": b_comp_e2e_ms,
                "varlen_comp_e2e_ms": v_comp_e2e_ms,
                "e2e_speedup_comp": e2e_sp_comp,
                "batched_ttft_ms": b_eager_ttft_ms,
                "varlen_ttft_ms": v_eager_ttft_ms,
                "batched_tpot_ms": b_eager_tpot_ms,
                "varlen_tpot_ms": v_eager_tpot_ms,
                "batched_e2e_ms": b_eager_e2e_ms,
                "varlen_e2e_ms": v_eager_e2e_ms,
                "e2e_speedup_eager": e2e_sp_eager,
                "varlen_vram_mb": v_vram_mb,
                "batched_vram_mb": b_vram_mb,
            }
        )

        b_ttft_str = f"{b_comp_ttft_ms:.2f}" if b_comp_ttft_ms is not None else f"{b_eager_ttft_ms:.2f}"
        v_ttft_str = f"{v_comp_ttft_ms:.2f}" if v_comp_ttft_ms is not None else f"{v_eager_ttft_ms:.2f}"
        ttft_sp_str = f"{ttft_sp_comp:.2f}x" if ttft_sp_comp is not None else f"{ttft_sp_eager:.2f}x"

        b_tpot_str = f"{b_comp_tpot_ms:.3f}" if b_comp_tpot_ms is not None else f"{b_eager_tpot_ms:.3f}"
        v_tpot_str = f"{v_comp_tpot_ms:.3f}" if v_comp_tpot_ms is not None else f"{v_eager_tpot_ms:.3f}"
        tpot_sp_str = f"{tpot_sp_comp:.2f}x" if tpot_sp_comp is not None else f"{tpot_sp_eager:.2f}x"

        e2e_sp_str = f"{e2e_sp_comp:.2f}x" if e2e_sp_comp is not None else f"{e2e_sp_eager:.2f}x"

        e2e_rows.append(
            [
                str(B),
                str(actual_max),
                str(gen_steps),
                f"{pad_waste_pct:.1f}%",
                b_ttft_str,
                v_ttft_str,
                ttft_sp_str,
                b_tpot_str,
                v_tpot_str,
                tpot_sp_str,
                e2e_sp_str,
                f"{v_vram_mb:.1f} / {b_vram_mb:.1f}",
            ]
        )

        del x_prefill_b, x_prefill_p, x_step_p, x_step_b
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "\n"
        + format_markdown_table(
            [
                "B",
                "MaxL",
                "Gen",
                "Pad%",
                "Base TTFT(ms)",
                "Ours TTFT(ms)",
                "TTFT Speedup",
                "Base TPOT(ms)",
                "Ours TPOT(ms)",
                "TPOT Speedup",
                "E2E Speedup",
                "VRAM MB (V/B)",
            ],
            e2e_rows,
        )
    )

    if save_plot:
        _save_inference_plot(decode_results, e2e_results, save_plot)

    return {"decode": decode_results, "e2e": e2e_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark for VarlenFlashGQA vs torch.compile Baseline.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--max-lens", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    parser.add_argument("--distributions", nargs="+", default=["skewed", "uniform"])
    parser.add_argument("--e2e-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--e2e-max-lens", type=int, nargs="+", default=None)
    parser.add_argument("--e2e-quick", action="store_true", help="Run a quick 5-case subset for Part B E2E.")
    parser.add_argument("--num-heads-q", type=int, default=16)
    parser.add_argument("--num-heads-kv", type=int, default=4)
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
