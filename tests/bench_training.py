"""Training (Forward + Backward) speed and VRAM benchmark with torch.compile baseline."""

import argparse
import gc
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _compile_or_warn(target: Any, label: str):
    """Compile a module or function with TorchInductor; return None on failure."""
    try:
        return torch.compile(target, backend="inductor", dynamic=True)
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

    labels = [f"B={r['B']}\nL={r['max_len']}" for r in skewed_rows]
    x_idx = list(range(len(labels)))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    has_comp = all(
        r.get("batched_comp_fwdbwd_ms") is not None and r.get("varlen_comp_fwdbwd_ms") is not None
        for r in skewed_rows
    )

    if has_comp:
        t_base_comp = [r["batched_comp_fwdbwd_ms"] for r in skewed_rows]
        t_var_comp = [r["varlen_comp_fwdbwd_ms"] for r in skewed_rows]
        ax1.bar(
            [i - width / 2 for i in x_idx],
            t_base_comp,
            width,
            label="Batched SDPA + torch.compile (Baseline)",
            color="#e63946",
        )
        ax1.bar(
            [i + width / 2 for i in x_idx],
            t_var_comp,
            width,
            label="Varlen CUDA + torch.compile (Ours)",
            color="#2a9d8f",
        )

        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_comp_fwdbwd", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + width / 2, t_var_comp[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        ax1.set_title("Training Fwd+Bwd Latency vs torch.compile Baseline (Lower is Better)")
    else:
        t_base_eager = [r["batched_fwdbwd_ms"] for r in skewed_rows]
        t_var_eager = [r["varlen_fwdbwd_ms"] for r in skewed_rows]
        ax1.bar([i - width / 2 for i in x_idx], t_base_eager, width, label="Batched SDPA (Eager)", color="#6c757d")
        ax1.bar([i + width / 2 for i in x_idx], t_var_eager, width, label="Varlen CUDA (Eager)", color="#0d6efd")
        for i, r in enumerate(skewed_rows):
            sp = r.get("speedup_eager_fwdbwd", 1.0)
            ax1.annotate(
                f"{sp:.2f}x",
                xy=(i + width / 2, t_var_eager[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        ax1.set_title("Training Fwd+Bwd Latency (Lower is Better)")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=0, ha="center")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    m_varlen = [r["varlen_vram_mb"] for r in skewed_rows]
    m_batched = [r["batched_vram_mb"] for r in skewed_rows]
    ax2.bar([i - width / 2 for i in x_idx], m_batched, width, label="Batched SDPA (Padded)", color="#f4a261")
    ax2.bar([i + width / 2 for i in x_idx], m_varlen, width, label="Varlen CUDA (Packed)", color="#264653")
    ax2.set_title("Peak Allocated VRAM during Fwd+Bwd (Lower is Better)")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels, rotation=0, ha="center")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")

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
    """
    Run Training (Forward & Forward+Backward) speed and memory benchmark.

    Establishes `BatchedFlashGQABaselineSpeed` with `torch.compile` (TorchInductor)
    as the primary speed baseline and benchmarks `VarlenFlashGQA` with `torch.compile`
    directly against it, alongside eager comparisons.
    """
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

    # Compile both models with TorchInductor
    mod_varlen_comp = _compile_or_warn(mod_varlen, "VarlenFlashGQA") if include_compile else None
    mod_batched_comp = _compile_or_warn(mod_batched, "BatchedSDPA") if include_compile else None

    results: List[Dict[str, Any]] = []
    table_rows: List[List[str]] = []
    fwd_rows: List[List[str]] = []

    print("\n" + "=" * 90)
    print(" TRAINING BENCHMARK: Varlen CUDA vs torch.compile Baseline Speed")
    print(f" Config: H_q={num_heads_q}, H_kv={num_heads_kv}, D={head_dim}, causal={is_causal}, rope={use_rope}")
    print(f" Primary Baseline: BatchedFlashGQABaselineSpeed + torch.compile(backend='inductor', dynamic=True)")
    print(f" Custom Kernel:    VarlenFlashGQA + torch.compile(backend='inductor', dynamic=True)")
    print("=" * 90)

    for dist in distributions:
        for B in batch_sizes:
            for max_len in max_lens:
                min_len = max(16, max_len // 16)
                seqlens, cu_seqlens, actual_max = generate_varlen_seqlens(
                    B, min_len, max_len, distribution=dist, device=dev
                )
                total_tokens = int(cu_seqlens[-1].item())
                pad_waste_pct = (1.0 - float(total_tokens) / float(B * actual_max)) * 100.0

                # Mask out padding positions on both x_padded and go_padded so Batched causal backward
                # accumulates zero gradient from padding rows (mathematically equivalent to Varlen).
                pos_ids = torch.arange(actual_max, device=dev, dtype=torch.int32).unsqueeze(0)
                valid_tok_mask = (pos_ids < seqlens.unsqueeze(1)).unsqueeze(-1).to(dtype=torch.float16)

                x_padded_raw = torch.randn((B, actual_max, embed_dim), device=dev, dtype=torch.float16) * valid_tok_mask
                go_padded = (torch.randn_like(x_padded_raw) * 0.1) * valid_tok_mask
                attn_mask = None if is_causal else build_padding_mask(cu_seqlens, actual_max)

                # Create packed tensors on CPU temporarily so Phase 1 VRAM only reflects Batched buffers
                x_packed_cpu = pack_varlen_tensor(x_padded_raw, cu_seqlens).cpu()
                go_packed_cpu = pack_varlen_tensor(go_padded, cu_seqlens).cpu()

                x_padded = x_padded_raw.requires_grad_(True)
                del x_padded_raw, pos_ids, valid_tok_mask
                gc.collect()
                torch.cuda.empty_cache()

                active_flops_fwd = compute_attention_flops(
                    seqlens, seqlens, num_heads_q=num_heads_q, head_dim=head_dim, is_causal=is_causal, include_backward=False
                )
                active_flops_fwdbwd = compute_attention_flops(
                    seqlens, seqlens, num_heads_q=num_heads_q, head_dim=head_dim, is_causal=is_causal, include_backward=True
                )

                # =============================================================
                # PHASE 1: BATCHED SDPA BASELINE (Compiled & Eager + Isolated VRAM)
                # =============================================================
                b_comp_fwd_ms: Optional[float] = None
                b_comp_fwdbwd_ms: Optional[float] = None
                if mod_batched_comp is not None:
                    def _batched_comp_fwd():
                        with torch.no_grad():
                            return mod_batched_comp(x_padded, attn_mask=attn_mask, is_causal=is_causal)

                    def _batched_comp_fwdbwd():
                        mod_batched.zero_grad(set_to_none=True)
                        x_padded.grad = None
                        out = mod_batched_comp(x_padded, attn_mask=attn_mask, is_causal=is_causal)
                        out.backward(go_padded)
                        return out

                    b_comp_fwd_ms, _, _ = cuda_benchmark(
                        _batched_comp_fwd, warmup=max(6, warmup // 2), rep=rep
                    )
                    b_comp_fwdbwd_ms, _, _ = cuda_benchmark(
                        _batched_comp_fwdbwd, warmup=max(6, warmup // 2), rep=rep
                    )
                    mod_batched.zero_grad(set_to_none=True)
                    x_padded.grad = None
                    torch.cuda.empty_cache()
                    _, b_vram_mb = measure_peak_memory_mb(_batched_comp_fwdbwd)
                    mod_batched.zero_grad(set_to_none=True)
                    x_padded.grad = None

                def _batched_eager_fwd():
                    with torch.no_grad():
                        return mod_batched(x_padded, attn_mask=attn_mask, is_causal=is_causal)

                def _batched_eager_fwdbwd():
                    mod_batched.zero_grad(set_to_none=True)
                    x_padded.grad = None
                    out = mod_batched(x_padded, attn_mask=attn_mask, is_causal=is_causal)
                    out.backward(go_padded)
                    return out

                b_eager_fwd_ms, _, _ = cuda_benchmark(_batched_eager_fwd, warmup=warmup, rep=rep)
                b_eager_fwdbwd_ms, _, _ = cuda_benchmark(_batched_eager_fwdbwd, warmup=warmup, rep=rep)
                if mod_batched_comp is None:
                    mod_batched.zero_grad(set_to_none=True)
                    x_padded.grad = None
                    torch.cuda.empty_cache()
                    _, b_vram_mb = measure_peak_memory_mb(_batched_eager_fwdbwd)

                # Clear ALL Batched GPU tensors & grads before Phase 2
                mod_batched.zero_grad(set_to_none=True)
                del x_padded, go_padded, attn_mask
                gc.collect()
                torch.cuda.empty_cache()

                # =============================================================
                # PHASE 2: VARLEN CUDA (Compiled & Eager + Isolated VRAM)
                # =============================================================
                x_packed = x_packed_cpu.to(device=dev).requires_grad_(True)
                go_packed = go_packed_cpu.to(device=dev)
                del x_packed_cpu, go_packed_cpu

                v_comp_fwd_ms: Optional[float] = None
                v_comp_fwdbwd_ms: Optional[float] = None
                if mod_varlen_comp is not None:
                    def _varlen_comp_fwd():
                        with torch.no_grad():
                            return mod_varlen_comp(
                                x_packed,
                                cu_seqlens_q=cu_seqlens,
                                max_seqlen_q=actual_max,
                                max_seqlen_k=actual_max,
                                is_causal=is_causal,
                            )

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

                    v_comp_fwd_ms, _, _ = cuda_benchmark(
                        _varlen_comp_fwd, warmup=max(6, warmup // 2), rep=rep
                    )
                    v_comp_fwdbwd_ms, _, _ = cuda_benchmark(
                        _varlen_comp_fwdbwd, warmup=max(6, warmup // 2), rep=rep
                    )
                    mod_varlen.zero_grad(set_to_none=True)
                    x_packed.grad = None
                    torch.cuda.empty_cache()
                    _, v_vram_mb = measure_peak_memory_mb(_varlen_comp_fwdbwd)
                    mod_varlen.zero_grad(set_to_none=True)
                    x_packed.grad = None

                def _varlen_eager_fwd():
                    with torch.no_grad():
                        return mod_varlen(
                            x_packed,
                            cu_seqlens_q=cu_seqlens,
                            max_seqlen_q=actual_max,
                            max_seqlen_k=actual_max,
                            is_causal=is_causal,
                        )

                def _varlen_eager_fwdbwd():
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

                v_eager_fwd_ms, _, _ = cuda_benchmark(_varlen_eager_fwd, warmup=warmup, rep=rep)
                v_eager_fwdbwd_ms, _, _ = cuda_benchmark(_varlen_eager_fwdbwd, warmup=warmup, rep=rep)
                if mod_varlen_comp is None:
                    mod_varlen.zero_grad(set_to_none=True)
                    x_packed.grad = None
                    torch.cuda.empty_cache()
                    _, v_vram_mb = measure_peak_memory_mb(_varlen_eager_fwdbwd)

                # =============================================================
                # PHASE 3: Optional Python Varlen Reference
                # =============================================================
                vref_fwdbwd_ms: Optional[float] = None
                if mod_vref is not None and actual_max <= 1024:
                    def _vref_fwdbwd():
                        mod_vref.zero_grad(set_to_none=True)
                        x_packed.grad = None
                        out = mod_vref(x_packed, cu_seqlens_q=cu_seqlens, is_causal=is_causal)
                        out.backward(go_packed)
                        return out

                    vref_fwdbwd_ms, _, _ = cuda_benchmark(_vref_fwdbwd, warmup=3, rep=15)
                    mod_vref.zero_grad(set_to_none=True)

                # Primary Speedup: Compiled Varlen vs Compiled Baseline
                if b_comp_fwdbwd_ms is not None and v_comp_fwdbwd_ms is not None:
                    speedup_comp_fwdbwd = b_comp_fwdbwd_ms / max(v_comp_fwdbwd_ms, 1e-6)
                    speedup_comp_fwd = (
                        b_comp_fwd_ms / max(v_comp_fwd_ms, 1e-6)
                        if b_comp_fwd_ms is not None and v_comp_fwd_ms is not None
                        else None
                    )
                else:
                    speedup_comp_fwdbwd = None
                    speedup_comp_fwd = None

                # Secondary Speedup: Eager Varlen vs Eager Batched
                speedup_eager_fwdbwd = b_eager_fwdbwd_ms / max(v_eager_fwdbwd_ms, 1e-6)
                speedup_eager_fwd = b_eager_fwd_ms / max(v_eager_fwd_ms, 1e-6)

                # Inductor gain on Custom Kernel
                inductor_gain = (
                    v_eager_fwdbwd_ms / max(v_comp_fwdbwd_ms, 1e-6)
                    if v_comp_fwdbwd_ms is not None
                    else 1.0
                )

                # TFLOP/s based on compiled or eager execution
                best_v_fwd = v_comp_fwd_ms if v_comp_fwd_ms is not None else v_eager_fwd_ms
                best_v_fwdbwd = v_comp_fwdbwd_ms if v_comp_fwdbwd_ms is not None else v_eager_fwdbwd_ms
                tflops_fwd = compute_tflops(active_flops_fwd, best_v_fwd)
                tflops_fwdbwd = compute_tflops(active_flops_fwdbwd, best_v_fwdbwd)

                row_dict = {
                    "dist": dist,
                    "B": B,
                    "max_len": actual_max,
                    "total_tokens": total_tokens,
                    "pad_waste_pct": pad_waste_pct,
                    "batched_comp_fwd_ms": b_comp_fwd_ms,
                    "varlen_comp_fwd_ms": v_comp_fwd_ms,
                    "speedup_comp_fwd": speedup_comp_fwd,
                    "batched_comp_fwdbwd_ms": b_comp_fwdbwd_ms,
                    "varlen_comp_fwdbwd_ms": v_comp_fwdbwd_ms,
                    "speedup_comp_fwdbwd": speedup_comp_fwdbwd,
                    "batched_fwd_ms": b_eager_fwd_ms,
                    "varlen_fwd_ms": v_eager_fwd_ms,
                    "speedup_eager_fwd": speedup_eager_fwd,
                    "batched_fwdbwd_ms": b_eager_fwdbwd_ms,
                    "varlen_fwdbwd_ms": v_eager_fwdbwd_ms,
                    "speedup_eager_fwdbwd": speedup_eager_fwdbwd,
                    "inductor_gain": inductor_gain,
                    "vref_fwdbwd_ms": vref_fwdbwd_ms,
                    "tflops_fwd": tflops_fwd,
                    "tflops_fwdbwd": tflops_fwdbwd,
                    "varlen_vram_mb": v_vram_mb,
                    "batched_vram_mb": b_vram_mb,
                }
                results.append(row_dict)

                comp_sp_str = f"{speedup_comp_fwdbwd:.2f}x" if speedup_comp_fwdbwd is not None else "N/A"
                b_comp_str = f"{b_comp_fwdbwd_ms:.3f}" if b_comp_fwdbwd_ms is not None else "N/A"
                v_comp_str = f"{v_comp_fwdbwd_ms:.3f}" if v_comp_fwdbwd_ms is not None else "N/A"

                table_rows.append(
                    [
                        dist,
                        str(B),
                        str(actual_max),
                        f"{pad_waste_pct:.1f}%",
                        b_comp_str,
                        v_comp_str,
                        comp_sp_str,
                        f"{b_eager_fwdbwd_ms:.3f}",
                        f"{v_eager_fwdbwd_ms:.3f}",
                        f"{speedup_eager_fwdbwd:.2f}x",
                        f"{inductor_gain:.2f}x",
                        f"{tflops_fwdbwd:.2f}",
                        f"{v_vram_mb:.1f} / {b_vram_mb:.1f}",
                    ]
                )

                comp_fwd_sp_str = f"{speedup_comp_fwd:.2f}x" if speedup_comp_fwd is not None else "N/A"
                b_comp_fwd_str = f"{b_comp_fwd_ms:.3f}" if b_comp_fwd_ms is not None else "N/A"
                v_comp_fwd_str = f"{v_comp_fwd_ms:.3f}" if v_comp_fwd_ms is not None else "N/A"
                fwd_ind_gain = (
                    v_eager_fwd_ms / max(v_comp_fwd_ms, 1e-6)
                    if v_comp_fwd_ms is not None
                    else 1.0
                )
                fwd_rows.append(
                    [
                        dist,
                        str(B),
                        str(actual_max),
                        f"{pad_waste_pct:.1f}%",
                        b_comp_fwd_str,
                        v_comp_fwd_str,
                        comp_fwd_sp_str,
                        f"{b_eager_fwd_ms:.3f}",
                        f"{v_eager_fwd_ms:.3f}",
                        f"{speedup_eager_fwd:.2f}x",
                        f"{fwd_ind_gain:.2f}x",
                        f"{tflops_fwd:.2f}",
                    ]
                )

                mod_varlen.zero_grad(set_to_none=True)
                del x_packed, go_packed
                gc.collect()
                torch.cuda.empty_cache()

    headers_fwdbwd = [
        "Dist",
        "B",
        "MaxLen",
        "Pad%",
        "Batched+Comp(ms) [Baseline]",
        "Varlen+Comp(ms) [Ours]",
        "Comp Speedup",
        "Batched Eager",
        "Varlen Eager",
        "Eager Speedup",
        "Inductor Gain",
        "TFLOP/s",
        "VRAM MB (V/B)",
    ]
    headers_fwd = [
        "Dist",
        "B",
        "MaxLen",
        "Pad%",
        "Batched+Comp Fwd(ms)",
        "Varlen+Comp Fwd(ms)",
        "Comp Speedup",
        "Batched Eager Fwd",
        "Varlen Eager Fwd",
        "Eager Speedup",
        "Inductor Gain",
        "Fwd TFLOP/s",
    ]

    print("\n### Training Forward + Backward (Speed & Peak VRAM)\n" + format_markdown_table(headers_fwdbwd, table_rows))
    print("\n### Training Forward-Only (Speed & TFLOP/s)\n" + format_markdown_table(headers_fwd, fwd_rows))

    if save_plot:
        _save_training_plot(results, save_plot)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Training benchmark for VarlenFlashGQA vs torch.compile Baseline.")
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
