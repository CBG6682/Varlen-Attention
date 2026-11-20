"""Inference (Decode L=1 & End-to-End Prefill+Generation) speed and VRAM benchmark."""

import argparse
import gc
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

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
    varlen_flash_decode,
)


def _repeat_kv_batched(x: torch.Tensor, rep: int) -> torch.Tensor:
    if rep == 1:
        return x
    B, H_kv, L, D = x.shape
    return x.unsqueeze(2).expand(B, H_kv, rep, L, D).reshape(B, H_kv * rep, L, D)


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
    skewed_rows = [r for r in decode_results if r["dist"] == "skewed"] or decode_results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    labels = [f"B={r['B']}, L={r['max_len']}" for r in skewed_rows]
    x_idx = list(range(len(labels)))
    width = 0.34

    t_varlen = [r["varlen_dec_ms"] for r in skewed_rows]
    t_batched = [r["batched_dec_ms"] for r in skewed_rows]

    ax1.bar([i - width / 2 for i in x_idx], t_batched, width, label="Batched SDPA Decode (Padded)", color="#6c757d")
    ax1.bar([i + width / 2 for i in x_idx], t_varlen, width, label="Varlen Split-K Decode (CUDA)", color="#0d6efd")
    for i, r in enumerate(skewed_rows):
        ax1.annotate(
            f"{r['speedup']:.2f}x\n({r['bw_gbps']:.0f} GB/s)",
            xy=(i + width / 2, t_varlen[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )
    ax1.set_title("Decode Kernel Latency L_q=1 (Cold L2, Lower is Better)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend()

    if e2e_results:
        e2e_labels = [f"B={r['B']}, L={r['max_len']}" for r in e2e_results]
        e_idx = list(range(len(e2e_labels)))
        v_tpot = [r["varlen_tpot_ms"] for r in e2e_results]
        b_tpot = [r["batched_tpot_ms"] for r in e2e_results]
        ax2.bar([i - width / 2 for i in e_idx], b_tpot, width, label="Batched SDPA TPOT", color="#fd7e14")
        ax2.bar([i + width / 2 for i in e_idx], v_tpot, width, label="Varlen GQA TPOT", color="#20c997")
        ax2.set_title("End-to-End Generation TPOT (ms/token)")
        ax2.set_ylabel("Time per Output Token (ms)")
        ax2.set_xticks(e_idx)
        ax2.set_xticklabels(e2e_labels, rotation=20, ha="right")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved inference benchmark plot -> {save_path}")


def run_inference_benchmark(
    batch_sizes: Sequence[int] = (1, 8, 32),
    max_lens: Sequence[int] = (256, 1024, 2048, 4096),
    distributions: Sequence[str] = ("skewed", "uniform"),
    num_heads_q: int = 16,
    num_heads_kv: int = 4,
    head_dim: int = 64,
    gen_steps: int = 16,
    use_rope: bool = True,
    include_compile: bool = True,
    clear_l2: bool = True,
    warmup: int = 15,
    rep: int = 60,
    save_plot: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Decode-only kernel benchmark and End-to-End Prefill + Generation benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference benchmark.")
    if not HAS_CUDA_EXT:
        raise RuntimeError("varlen_transformer._C extension is not compiled.")

    seed_everything(seed)
    dev = torch.device("cuda")
    embed_dim = num_heads_q * head_dim
    rep_gqa = num_heads_q // num_heads_kv
    scale = 1.0 / math.sqrt(head_dim)

    max_table_len = max(max_lens) + gen_steps + 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_table_len, device=dev) if use_rope else None
    cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16) if rope is not None else (None, None)

    compiled_decode_op = None
    if include_compile:
        try:
            @torch.compile(backend="inductor", dynamic=True)
            def _decode_op(q_in, k_in, v_in, cu_in, max_k, c_in, s_in):
                return varlen_flash_decode(q_in, k_in, v_in, cu_in, max_seqlen_k=max_k, cos=c_in, sin=s_in)

            compiled_decode_op = _decode_op
        except Exception as exc:
            print(f"[WARN] Could not torch.compile(varlen_flash_decode): {exc}")

    # =========================================================================
    # PART A: Microbenchmark Decode Attention L_q = 1 (Varlen Split-K vs Batched SDPA)
    # =========================================================================
    print(
        f"\nRunning Part A - Decode L_q=1 Benchmark: H_q={num_heads_q}, H_kv={num_heads_kv}, D={head_dim}, "
        f"rope={use_rope}, clear_l2={clear_l2}"
    )

    decode_results: List[Dict[str, Any]] = []
    decode_rows: List[List[str]] = []

    for dist in distributions:
        for B in batch_sizes:
            for max_len in max_lens:
                min_len = max(32, max_len // 16)
                seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
                    B, min_len, max_len, distribution=dist, device=dev
                )
                total_lk = int(cu_seqlens[-1].item())

                q_dec = torch.randn((num_heads_q, B, head_dim), device=dev, dtype=torch.float16)
                k_packed = torch.randn((num_heads_kv, total_lk, head_dim), device=dev, dtype=torch.float16)
                v_packed = torch.randn((num_heads_kv, total_lk, head_dim), device=dev, dtype=torch.float16)

                # Batched padded KV cache: (B, H_kv, actual_max, D) + padding mask (B, 1, 1, actual_max)
                q_batched = q_dec.transpose(0, 1).unsqueeze(2).contiguous()  # (B, H_q, 1, D)
                k_batched = torch.randn((B, num_heads_kv, actual_max, head_dim), device=dev, dtype=torch.float16)
                v_batched = torch.randn((B, num_heads_kv, actual_max, head_dim), device=dev, dtype=torch.float16)
                pad_mask = build_padding_mask(cu_seqlens, actual_max)

                def _varlen_dec_fn():
                    with torch.no_grad():
                        return varlen_flash_decode(
                            q_dec, k_packed, v_packed, cu_seqlens, max_seqlen_k=actual_max, cos=cos, sin=sin
                        )

                def _batched_dec_fn():
                    with torch.no_grad():
                        q_in = rope(q_batched, offset=actual_max - 1) if rope is not None else q_batched
                        k_exp = _repeat_kv_batched(k_batched, rep_gqa)
                        v_exp = _repeat_kv_batched(v_batched, rep_gqa)
                        return F.scaled_dot_product_attention(
                            q_in, k_exp, v_exp, attn_mask=pad_mask, scale=scale
                        )

                v_ms, _, _ = cuda_benchmark(_varlen_dec_fn, warmup=warmup, rep=rep, clear_l2=clear_l2)
                b_ms, _, _ = cuda_benchmark(_batched_dec_fn, warmup=warmup, rep=rep, clear_l2=clear_l2)

                v_comp_ms: Optional[float] = None
                if compiled_decode_op is not None:
                    def _varlen_dec_comp_fn():
                        with torch.no_grad():
                            return compiled_decode_op(
                                q_dec, k_packed, v_packed, cu_seqlens, actual_max, cos, sin
                            )

                    v_comp_ms, _, _ = cuda_benchmark(
                        _varlen_dec_comp_fn, warmup=max(5, warmup // 2), rep=rep, clear_l2=clear_l2
                    )

                speedup = b_ms / max(v_ms, 1e-6)
                bw_gbps = compute_decode_bandwidth_gbps(
                    total_lk, num_heads_kv, num_heads_q, B, head_dim, v_ms
                )
                tok_per_s = B / (v_ms * 1e-3)

                decode_results.append(
                    {
                        "dist": dist,
                        "B": B,
                        "max_len": actual_max,
                        "total_lk": total_lk,
                        "varlen_dec_ms": v_ms,
                        "varlen_comp_ms": v_comp_ms,
                        "batched_dec_ms": b_ms,
                        "speedup": speedup,
                        "bw_gbps": bw_gbps,
                        "tok_per_s": tok_per_s,
                    }
                )
                decode_rows.append(
                    [
                        dist,
                        str(B),
                        str(actual_max),
                        str(total_lk),
                        f"{v_ms:.3f}",
                        f"{v_comp_ms:.3f}" if v_comp_ms is not None else "N/A",
                        f"{b_ms:.3f}",
                        f"{speedup:.2f}x",
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
            ["Dist", "B", "MaxK", "TotalLk", "Varlen(ms)", "Varlen+Comp", "Batched(ms)", "Speedup", "BW(GB/s)", "Tok/s"],
            decode_rows,
        )
    )

    # =========================================================================
    # PART B: End-to-End Prefill (TTFT) + Multi-Step Generation (TPOT & VRAM)
    # =========================================================================
    print(f"\nRunning Part B - End-to-End Prefill + {gen_steps}-Step Generation Benchmark")

    mod_varlen = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched = BatchedFlashGQABaselineSpeed(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched.load_state_dict(mod_varlen.state_dict())

    e2e_configs = [(1, 1024), (8, 1024), (16, 1024)]
    e2e_results: List[Dict[str, Any]] = []
    e2e_rows: List[List[str]] = []

    for B, max_len in e2e_configs:
        seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
            B, max(64, max_len // 16), max_len, distribution="skewed", device=dev
        )
        x_prefill_b = torch.randn((B, actual_max, embed_dim), device=dev, dtype=torch.float16)
        x_prefill_p = pack_varlen_tensor(x_prefill_b, cu_seqlens)
        x_step_p = torch.randn((B, embed_dim), device=dev, dtype=torch.float16)
        x_step_b = x_step_p.unsqueeze(1)  # (B, 1, embed_dim)

        def _e2e_varlen_prefill_and_gen():
            with torch.no_grad():
                _, cache = mod_varlen(
                    x_prefill_p,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=actual_max,
                    max_seqlen_k=actual_max,
                    is_causal=True,
                    use_cache=True,
                )
                for _ in range(gen_steps):
                    _, cache = mod_varlen(x_step_p, kv_cache=cache, use_cache=True, decode=True)
                return cache

        def _e2e_batched_prefill_and_gen():
            with torch.no_grad():
                _, b_cache = mod_batched(x_prefill_b, is_causal=True, use_cache=True)
                for step in range(gen_steps):
                    _, b_cache = mod_batched(
                        x_step_b, kv_cache=b_cache, use_cache=True, offset=actual_max + step
                    )
                return b_cache

        def _varlen_ttft_fn():
            with torch.no_grad():
                return mod_varlen(
                    x_prefill_p,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=actual_max,
                    max_seqlen_k=actual_max,
                    is_causal=True,
                    use_cache=True,
                )

        def _batched_ttft_fn():
            with torch.no_grad():
                return mod_batched(x_prefill_b, is_causal=True, use_cache=True)

        v_ttft_ms, _, _ = cuda_benchmark(_varlen_ttft_fn, warmup=10, rep=30)
        b_ttft_ms, _, _ = cuda_benchmark(_batched_ttft_fn, warmup=10, rep=30)

        _, v_vram_mb = measure_peak_memory_mb(_e2e_varlen_prefill_and_gen)
        v_e2e_ms, _, _ = cuda_benchmark(_e2e_varlen_prefill_and_gen, warmup=5, rep=20, clear_l2=clear_l2)

        _, b_vram_mb = measure_peak_memory_mb(_e2e_batched_prefill_and_gen)
        b_e2e_ms, _, _ = cuda_benchmark(_e2e_batched_prefill_and_gen, warmup=5, rep=20, clear_l2=clear_l2)

        v_tpot_ms = max(0.0, v_e2e_ms - v_ttft_ms) / float(gen_steps)
        b_tpot_ms = max(0.0, b_e2e_ms - b_ttft_ms) / float(gen_steps)

        e2e_results.append(
            {
                "B": B,
                "max_len": actual_max,
                "varlen_ttft_ms": v_ttft_ms,
                "batched_ttft_ms": b_ttft_ms,
                "varlen_tpot_ms": v_tpot_ms,
                "batched_tpot_ms": b_tpot_ms,
                "varlen_e2e_ms": v_e2e_ms,
                "batched_e2e_ms": b_e2e_ms,
                "varlen_vram_mb": v_vram_mb,
                "batched_vram_mb": b_vram_mb,
            }
        )
        e2e_rows.append(
            [
                str(B),
                str(actual_max),
                f"{v_ttft_ms:.2f}",
                f"{b_ttft_ms:.2f}",
                f"{v_tpot_ms:.3f}",
                f"{b_tpot_ms:.3f}",
                f"{b_e2e_ms / max(v_e2e_ms, 1e-6):.2f}x",
                f"{v_vram_mb:.1f} / {b_vram_mb:.1f}",
            ]
        )

        del x_prefill_b, x_prefill_p, x_step_p, x_step_b
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "\n"
        + format_markdown_table(
            ["B", "MaxK", "V-TTFT(ms)", "B-TTFT(ms)", "V-TPOT(ms)", "B-TPOT(ms)", "E2E Speedup", "VRAM MB (V/B)"],
            e2e_rows,
        )
    )

    if save_plot:
        _save_inference_plot(decode_results, e2e_results, save_plot)

    return {"decode": decode_results, "e2e": e2e_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark for VarlenFlashGQA decode & E2E.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--max-lens", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    parser.add_argument("--distributions", nargs="+", default=["skewed", "uniform"])
    parser.add_argument("--num-heads-q", type=int, default=16)
    parser.add_argument("--num-heads-kv", type=int, default=4)
    parser.add_argument("--gen-steps", type=int, default=16)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-clear-l2", action="store_true")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--rep", type=int, default=60)
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_inference_benchmark(
        batch_sizes=args.batch_sizes,
        max_lens=args.max_lens,
        distributions=args.distributions,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        gen_steps=args.gen_steps,
        use_rope=not args.no_rope,
        include_compile=not args.no_compile,
        clear_l2=not args.no_clear_l2,
        warmup=args.warmup,
        rep=args.rep,
        save_plot=args.save_plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
