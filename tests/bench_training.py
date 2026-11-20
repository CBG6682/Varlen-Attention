"""Training (Forward + Backward) speed and VRAM benchmark."""

import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tests.utils import (
    build_padding_mask,
    compute_attention_flops,
    compute_tflops,
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
    VarlenFlashGQABaselineCorrectness,
)


def _compile_or_warn(module: torch.nn.Module, label: str):
    try:
        return torch.compile(module, backend="inductor", dynamic=True)
    except Exception as exc:
        print(f"[WARN] Could not torch.compile({label}): {exc}")
        return None


def _save_training_plot(results: List[Dict[str, Any]], save_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping training plot.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    skewed_rows = [r for r in results if r["dist"] == "skewed"] or results

    labels = [f"B={r['B']}, L={r['max_len']}" for r in skewed_rows]
    x_idx = list(range(len(labels)))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    t_varlen = [r["varlen_fwdbwd_ms"] for r in skewed_rows]
    t_batched = [r["batched_fwdbwd_ms"] for r in skewed_rows]
    has_comp = all(r.get("varlen_comp_fwdbwd_ms") is not None for r in skewed_rows)

    ax1.bar([i - width / 2 for i in x_idx], t_batched, width, label="Batched SDPA (Padded)", color="#6c757d")
    ax1.bar([i + width / 2 for i in x_idx], t_varlen, width, label="Varlen CUDA (Fused RoPE)", color="#0d6efd")
    if has_comp:
        t_vcomp = [r["varlen_comp_fwdbwd_ms"] for r in skewed_rows]
        ax1.plot(x_idx, t_vcomp, marker="o", color="#198754", linewidth=2, label="Varlen + torch.compile")

    for i, r in enumerate(skewed_rows):
        sp = r["speedup_fwdbwd"]
        ax1.annotate(
            f"{sp:.2f}x",
            xy=(i + width / 2, t_varlen[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax1.set_title("Training Fwd+Bwd Latency (Lower is Better)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend()

    m_varlen = [r["varlen_vram_mb"] for r in skewed_rows]
    m_batched = [r["batched_vram_mb"] for r in skewed_rows]
    ax2.bar([i - width / 2 for i in x_idx], m_batched, width, label="Batched SDPA (Padded)", color="#fd7e14")
    ax2.bar([i + width / 2 for i in x_idx], m_varlen, width, label="Varlen CUDA (Packed)", color="#20c997")
    ax2.set_title("Peak Allocated VRAM during Fwd+Bwd (Lower is Better)")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved training benchmark plot -> {save_path}")


def run_training_benchmark(
    batch_sizes: Sequence[int] = (4, 8, 16),
    max_lens: Sequence[int] = (512, 1024, 2048),
    distributions: Sequence[str] = ("skewed", "uniform"),
    num_heads_q: int = 16,
    num_heads_kv: int = 4,
    head_dim: int = 64,
    is_causal: bool = True,
    use_rope: bool = True,
    include_compile: bool = True,
    include_varlen_ref: bool = False,
    warmup: int = 15,
    rep: int = 50,
    save_plot: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run Training (Fwd & Fwd+Bwd) speed and memory benchmark vs Baselines."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training benchmark.")
    if not HAS_CUDA_EXT:
        raise RuntimeError("varlen_transformer._C extension is not compiled.")

    seed_everything(seed)
    dev = torch.device("cuda")
    embed_dim = num_heads_q * head_dim

    max_table_len = max(max_lens) + 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_table_len, device=dev) if use_rope else None

    mod_varlen = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched = BatchedFlashGQABaselineSpeed(embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched.load_state_dict(mod_varlen.state_dict())

    mod_vref = None
    if include_varlen_ref:
        mod_vref = VarlenFlashGQABaselineCorrectness(
            embed_dim, num_heads_q, num_heads_kv, rope=rope, bias=False
        ).to(device=dev, dtype=torch.float16)
        mod_vref.load_state_dict(mod_varlen.state_dict())

    mod_varlen_comp = _compile_or_warn(mod_varlen, "VarlenFlashGQA") if include_compile else None
    mod_batched_comp = _compile_or_warn(mod_batched, "BatchedSDPA") if include_compile else None

    results: List[Dict[str, Any]] = []
    table_rows: List[List[str]] = []

    print(
        f"\nRunning Training Benchmark: H_q={num_heads_q}, H_kv={num_heads_kv}, D={head_dim}, "
        f"causal={is_causal}, rope={use_rope}"
    )

    for dist in distributions:
        for B in batch_sizes:
            for max_len in max_lens:
                min_len = max(16, max_len // 16)
                seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
                    B, min_len, max_len, distribution=dist, device=dev
                )
                total_tokens = int(cu_seqlens[-1].item())
                pad_waste_pct = (1.0 - float(total_tokens) / float(B * actual_max)) * 100.0

                x_padded = torch.randn(
                    (B, actual_max, embed_dim), device=dev, dtype=torch.float16, requires_grad=True
                )
                go_padded = torch.randn_like(x_padded) * 0.1
                attn_mask = None if is_causal else build_padding_mask(cu_seqlens, actual_max)

                x_packed = pack_varlen_tensor(x_padded.detach(), cu_seqlens).requires_grad_(True)
                go_packed = pack_varlen_tensor(go_padded, cu_seqlens)

                active_flops_fwd = compute_attention_flops(
                    seqlens, seqlens, num_heads_q=num_heads_q, head_dim=head_dim, is_causal=is_causal, include_backward=False
                )
                active_flops_fwdbwd = compute_attention_flops(
                    seqlens, seqlens, num_heads_q=num_heads_q, head_dim=head_dim, is_causal=is_causal, include_backward=True
                )

                # --- 1. Varlen CUDA Eager ---
                def _varlen_fwd():
                    with torch.no_grad():
                        return mod_varlen(
                            x_packed,
                            cu_seqlens_q=cu_seqlens,
                            max_seqlen_q=actual_max,
                            max_seqlen_k=actual_max,
                            is_causal=is_causal,
                        )

                def _varlen_fwdbwd():
                    mod_varlen.zero_grad(set_to_none=True)
                    x_packed.grad = None
                    out = mod_varlen(
                        x_packed,
                        cu_seqlens_q=cu_seqlens,
                        max_seqlen_q=actual_max,
                        max_seqlen_k=actual_max,
                        is_causal=is_causal,
                    )
                    out.backward(go_packed)
                    return out

                v_fwd_ms, _, _ = cuda_benchmark(_varlen_fwd, warmup=warmup, rep=rep)
                _, v_vram_mb = measure_peak_memory_mb(_varlen_fwdbwd)
                v_fwdbwd_ms, _, _ = cuda_benchmark(_varlen_fwdbwd, warmup=warmup, rep=rep)

                # --- 2. Batched SDPA Speed Baseline Eager ---
                def _batched_fwd():
                    with torch.no_grad():
                        return mod_batched(x_padded, attn_mask=attn_mask, is_causal=is_causal)

                def _batched_fwdbwd():
                    mod_batched.zero_grad(set_to_none=True)
                    x_padded.grad = None
                    out = mod_batched(x_padded, attn_mask=attn_mask, is_causal=is_causal)
                    out.backward(go_padded)
                    return out

                b_fwd_ms, _, _ = cuda_benchmark(_batched_fwd, warmup=warmup, rep=rep)
                _, b_vram_mb = measure_peak_memory_mb(_batched_fwdbwd)
                b_fwdbwd_ms, _, _ = cuda_benchmark(_batched_fwdbwd, warmup=warmup, rep=rep)

                # --- 3. Compiled Variants ---
                v_comp_fwdbwd_ms: Optional[float] = None
                if mod_varlen_comp is not None:
                    def _varlen_comp_fwdbwd():
                        mod_varlen.zero_grad(set_to_none=True)
                        x_packed.grad = None
                        out = mod_varlen_comp(
                            x_packed,
                            cu_seqlens_q=cu_seqlens,
                            max_seqlen_q=actual_max,
                            max_seqlen_k=actual_max,
                            is_causal=is_causal,
                        )
                        out.backward(go_packed)
                        return out

                    v_comp_fwdbwd_ms, _, _ = cuda_benchmark(_varlen_comp_fwdbwd, warmup=max(5, warmup // 2), rep=rep)

                b_comp_fwdbwd_ms: Optional[float] = None
                if mod_batched_comp is not None:
                    def _batched_comp_fwdbwd():
                        mod_batched.zero_grad(set_to_none=True)
                        x_padded.grad = None
                        out = mod_batched_comp(x_padded, attn_mask=attn_mask, is_causal=is_causal)
                        out.backward(go_padded)
                        return out

                    b_comp_fwdbwd_ms, _, _ = cuda_benchmark(_batched_comp_fwdbwd, warmup=max(5, warmup // 2), rep=rep)

                # --- 4. Optional Python Varlen Reference ---
                vref_fwdbwd_ms: Optional[float] = None
                if mod_vref is not None and actual_max <= 1024:
                    def _vref_fwdbwd():
                        mod_vref.zero_grad(set_to_none=True)
                        x_packed.grad = None
                        out = mod_vref(x_packed, cu_seqlens_q=cu_seqlens, is_causal=is_causal)
                        out.backward(go_packed)
                        return out

                    vref_fwdbwd_ms, _, _ = cuda_benchmark(_vref_fwdbwd, warmup=3, rep=15)

                speedup_fwd = b_fwd_ms / max(v_fwd_ms, 1e-6)
                speedup_fwdbwd = b_fwdbwd_ms / max(v_fwdbwd_ms, 1e-6)
                tflops_fwd = compute_tflops(active_flops_fwd, v_fwd_ms)
                tflops_fwdbwd = compute_tflops(active_flops_fwdbwd, v_fwdbwd_ms)

                row_dict = {
                    "dist": dist,
                    "B": B,
                    "max_len": actual_max,
                    "total_tokens": total_tokens,
                    "pad_waste_pct": pad_waste_pct,
                    "varlen_fwd_ms": v_fwd_ms,
                    "batched_fwd_ms": b_fwd_ms,
                    "speedup_fwd": speedup_fwd,
                    "varlen_fwdbwd_ms": v_fwdbwd_ms,
                    "batched_fwdbwd_ms": b_fwdbwd_ms,
                    "speedup_fwdbwd": speedup_fwdbwd,
                    "varlen_comp_fwdbwd_ms": v_comp_fwdbwd_ms,
                    "batched_comp_fwdbwd_ms": b_comp_fwdbwd_ms,
                    "vref_fwdbwd_ms": vref_fwdbwd_ms,
                    "tflops_fwd": tflops_fwd,
                    "tflops_fwdbwd": tflops_fwdbwd,
                    "varlen_vram_mb": v_vram_mb,
                    "batched_vram_mb": b_vram_mb,
                }
                results.append(row_dict)

                row_str = [
                    dist,
                    str(B),
                    str(actual_max),
                    f"{pad_waste_pct:.1f}%",
                    f"{v_fwd_ms:.3f} ({speedup_fwd:.2f}x)",
                    f"{b_fwd_ms:.3f}",
                    f"{v_fwdbwd_ms:.3f} ({speedup_fwdbwd:.2f}x)",
                    f"{b_fwdbwd_ms:.3f}",
                    f"{v_comp_fwdbwd_ms:.3f}" if v_comp_fwdbwd_ms is not None else "N/A",
                    f"{b_comp_fwdbwd_ms:.3f}" if b_comp_fwdbwd_ms is not None else "N/A",
                    f"{tflops_fwdbwd:.2f}",
                    f"{v_vram_mb:.1f} / {b_vram_mb:.1f}",
                ]
                if include_varlen_ref:
                    row_str.append(f"{vref_fwdbwd_ms:.2f}" if vref_fwdbwd_ms is not None else "skipped")
                table_rows.append(row_str)

                del x_padded, go_padded, x_packed, go_packed, attn_mask
                gc.collect()
                torch.cuda.empty_cache()

    headers = [
        "Dist",
        "B",
        "MaxLen",
        "Pad%",
        "Varlen Fwd(ms)",
        "Batched Fwd(ms)",
        "Varlen Fwd+Bwd",
        "Batched Fwd+Bwd",
        "Varlen+Compile",
        "Batched+Compile",
        "Eff TFLOP/s",
        "VRAM MB (V/B)",
    ]
    if include_varlen_ref:
        headers.append("PyVarlenRef(ms)")

    print("\n" + format_markdown_table(headers, table_rows))

    if save_plot:
        _save_training_plot(results, save_plot)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Training benchmark for VarlenFlashGQA vs Baselines.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--max-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--distributions", nargs="+", default=["skewed", "uniform"])
    parser.add_argument("--num-heads-q", type=int, default=16)
    parser.add_argument("--num-heads-kv", type=int, default=4)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--include-varlen-ref", action="store_true")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training_benchmark(
        batch_sizes=args.batch_sizes,
        max_lens=args.max_lens,
        distributions=args.distributions,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        is_causal=not args.no_causal,
        use_rope=not args.no_rope,
        include_compile=not args.no_compile,
        include_varlen_ref=args.include_varlen_ref,
        warmup=args.warmup,
        rep=args.rep,
        save_plot=args.save_plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
