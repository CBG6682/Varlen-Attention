<h1 align="center">
  <img src="docs/images/title.svg" alt="Varlen-Attention" width="720">
</h1>

<p align="center"><em>Pure THD variable-length FlashAttention, Paged KV Cache, and C++ fused linear projections for NVIDIA Turing (SM75).</em></p>

<p align="center">
  <img src="https://img.shields.io/github/license/CBG6682/Varlen-Attention?style=flat&logo=opensourceinitiative&logoColor=white&labelColor=454545&color=E91E63" alt="license">
  <img src="https://img.shields.io/github/last-commit/CBG6682/Varlen-Attention?style=flat&logo=git&logoColor=white&labelColor=454545&color=E91E63" alt="last commit">
  <img src="https://img.shields.io/github/languages/top/CBG6682/Varlen-Attention?style=flat&logo=nvidia&logoColor=white&labelColor=454545&color=E91E63" alt="top language">
  <img src="https://img.shields.io/github/languages/count/CBG6682/Varlen-Attention?style=flat&logo=github&logoColor=white&labelColor=454545&color=E91E63" alt="languages">
</p>

<p align="center"><em>Built with the tools and technologies:</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/C%2B%2B20-00599C.svg?style=flat&logo=cplusplus&logoColor=white" alt="C++20">
</p>

<div align="center">

| **3.43x** | **27.01x** | **8.52x** | **-68%** |
|:---:|:---:|:---:|:---:|
| faster training Fwd+Bwd | faster skewed L_q=1 decode | faster E2E TPOT | peak VRAM saved |

</div>

<details>
<summary><b>Table of Contents</b></summary>

- [Overview](#overview)
- [Architecture and performance analysis](#architecture-and-performance-analysis)
- [Benchmarks](#benchmarks)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Demo](#demo)
- [Project structure](#project-structure)
- [License](#license)

</details>

---

## Overview

`Varlen-Attention` (`varlen_transformer`) v2 provides zero-padding variable-length FlashAttention (`(Total_L, H, 64)` `THD` layout), fused non-interleaved RoPE, Paged KV Cache (`PAGE_SIZE = 16`), and C++ fused linear projections (`at::mm` via cuBLAS Tensor Core) for NVIDIA Turing (`SM75`). On an NVIDIA Tesla T4 (`SM75`), **pure Eager** execution outperforms `torch.compile` (`SDPBackend.EFFICIENT_ATTENTION`) by up to **3.43x** in training (`5.33x` forward-only) with **68%** lower peak VRAM, up to **27.01x** in skewed `L_q = 1` decode (`36.24x` on short uniform batches), and **8.52x** faster end-to-end token generation (`TPOT`).

### Why Varlen-Attention v2

- 🎯 **Built for Turing (`SM75`):** Targets Tesla T4 and RTX 20 series GPUs, where `Triton` kernels focus on Ampere (`SM80+`) and legacy `FlashAttention-1` requires downgrading `PyTorch` and C++ compilers.
- ⚡ **Zero-compile Eager speed:** Moves bias-free QKV/output `Linear` projections and Paged KV cache ops into C++ (`src/gqa_fused.cu` and `src/kv_cache.cu`), eliminating `torch.compile` graph breaks and Python dispatch overhead while retaining `TORCH_LIBRARY` and `FakeTensor` compatibility.
- 📦 **Pure `THD` zero-copy layout:** Operates natively on strided `(Total_L, H, 64)` and `(B, H, 64)` views sliced directly out of `qkv` / `kv`, removing transpose and concatenation buffers across self-attention and cross-attention.
- 🧩 **Paged KV Cache (`PAGE_SIZE = 16`):** Replaces $O(Total\_Lk)$ linear cache reallocation (`searchsorted` + `index_select` + `cat`) with $O(B)$ in-place `paged_kv_append` and page-aligned Split-K decoding.

| Solution | Turing (`SM75`) speed | Zero-compile Eager | Zero-padding varlen | Paged KV & cross-attn |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch SDPA** (`F.scaled_dot_product_attention`) | Falls back to `MATH` with `bool` masks | Standard eager overhead | No (requires padding) | No |
| **Triton / FlexAttention** | Tuned for Ampere (`A100 / SM80+`) | Requires JIT compile | Partial | Custom user glue |
| **FlashAttention-1 (`v1.0.9`)** | Fast | Unmaintained toolchain | Yes | Separate projection/cache glue |
| **Varlen-Attention v2 (`Ours`)** | **Optimized for `SM75`** | **Single-call C++ Eager** | **Yes (`THD` strided)** | **Yes (Paged KV + Cross-Attn)** |

---

## Architecture and performance analysis

### v1 vs v2 upgrade summary

| Dimension | v1 Architecture | v2 Architecture (`Current`) | Measured impact on T4 (`SM75`) |
| :--- | :--- | :--- | :--- |
| Tensor layout | Head-first `(H, Total_L, 64)` | Pure `THD` strided `(Total_L, H, 64)` | Removes transpose + `cat` round-trips |
| Linear fusion | Relied on `torch.compile` | C++ `at::mm` + kernel launch in 1 call | Up to `3.43x` train (`4.257 -> 1.579 ms` at `B=4, L=508`) |
| KV Cache append | Linear `searchsorted` + `cat` (`O(Total_Lk)`) | Paged pool (`PAGE_SIZE = 16`, `O(B)` append) | E2E `TPOT` at `B=64, L=510`: `1.129 -> 0.539 ms` (`2.09x` over v1) |
| Attention coverage | Self-attention + core cross kernels | Full fused Eager Self & Cross (`Fwd+Bwd`) | Single `d_qkv` buffer + FP32->FP16 cast kernel |

### Bottlenecks and VRAM behavior

- **Training and prefill (compute-bound & GMEM activation bound):** As sequence length grows, attention compute scales quadratically while `Batched SDPA` allocates padded `[B, MaxLen, ...]` activations, transpose/contiguous buffers, and RoPE/KV-repeat intermediates on global memory (`GMEM`). Fusing linear projections, RoPE, and FlashAttention in pure `THD` layout keeps $O(L^2)$ attention tiles in registers/SMEM (`16 KB` fwd, `64 KB` bwd) and writes only minimal packed activations back to `GMEM`, cutting peak VRAM by up to **68%** (`70.0 MB` vs `221.6 MB` at `B = 16, MaxLen = 471`, and `336.9 MB` vs `862.7 MB` at `B = 16, MaxLen = 2041`).
- **Decode at small batch / short KV (launch-overhead bound):** With `L_q = 1` and short contexts (`B = 1, MaxK = 38..44`), arithmetic intensity is minimal and GPU kernel launch plus Python/dispatcher overhead dominates total runtime. Executing QKV GEMM, `paged_kv_append`, `fused_mha_varlen_decode`, and output GEMM inside a single C++ call (`fused_gqa_varlen_decode`) lowers `L_q = 1` attention latency down to **0.007–0.014 ms**, producing **27.01x–36.24x** speedups against compiled `Batched SDPA`.
- **Decode at heavy KV cache (HBM read-bandwidth bound):** When `TotalLk` scales into tens of thousands of tokens (`TotalLk = 50,267–143,272` tokens at `B = 32..64, MaxK = 4080`), KV tensors exceed T4's `4 MB` L2 cache (`88–147 MB` FP16 K/V footprint), so execution becomes strictly limited by streaming KV pages from HBM (`GMEM -> SMEM/registers`). Speedup naturally narrows from launch-bound regimes (`18.15x` at `B = 64, MaxK = 1021`) to the active-data reduction ratio (**3.11x–5.65x** at `MaxK ~ 4080`), while still outperforming `Batched SDPA` because paged varlen reads zero padded tokens and avoids `4x` GQA KV head expansion.
- **Paged KV Cache scope in static benchmarks:** In `tests/bench_inference.py`, each static benchmark batch allocates its own rectangular `block_table` (`for_prefill(..., extra_pages_per_seq=4)`), so external memory fragmentation across dynamic multi-tenant lifetimes does not occur and pre-reserved decode headroom adds a small fixed VRAM margin compared to exact-sized packed buffers (for example `320.5 MB` in v2 vs `292.3 MB` in v1 at `B = 64, MaxL = 510`, still `-65%` vs `914.5 MB` baseline). Dedicated continuous-batching serving benchmarks with staggered arrivals, evictions, and prefix sharing remain future validation work to measure fragmentation savings directly.

---

## Benchmarks

> [!NOTE]
> **Setup:** Measured on **NVIDIA Tesla T4 (15 GB, `SM75`)** with **PyTorch `2.14.1+cu130`**, **CUDA `13.0`**, Driver `580.82.07`, `head_dim = 64`, FP16, `RoPE = True`, median latency over `rep = 50` (`warmup = 15`; `clear_l2 = True` for inference).
> **Fair Baseline:** `BatchedFlashGQABaselineSpeed` compiled with `torch.compile(backend="inductor")`, using **FP16 additive padding masks (`0.0` / `-inf`)** aligned to multiples of 8 (`stride(-1) = 1`, `stride(-2) % 8 == 0`) to explicitly activate **`SDPBackend.EFFICIENT_ATTENTION`** (xFormers CUTLASS) on T4 (`choice=2` verified). `VarlenFlashGQA` runs **Pure Eager** without `torch.compile`.

### Training (`Fwd+Bwd`, `H_q = 16, H_kv = 4, D = 64, causal = True`)

| Dist | `B` | `MaxLen` | Pad (%) | Batched+Comp (ms) ↓ | Varlen Eager (ms) ↓ | vs Comp ↑ | Batched Eager (ms) ↓ | TFLOP/s ↑ | VRAM (Ours / Base, MB) ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skewed | 4 | 508 | 39.8% | 4.257 | **1.579** | **2.70x** | 3.253 | 2.27 | **53.1** / 78.8 (-33%) |
| skewed | 4 | 1945 | 59.9% | 16.408 | **5.904** | **2.78x** | 18.928 | 5.22 | **89.9** / 234.9 (-62%) |
| skewed | 8 | 2020 | 59.2% | 33.993 | **12.376** | **2.75x** | 39.875 | 5.41 | **159.5** / 444.4 (-64%) |
| skewed | 16 | 471 | 72.1% | 8.655 | **2.525** | **3.43x** | 11.085 | 1.55 | **70.0** / 221.6 (**-68%**) |
| skewed | 16 | 2041 | 52.0% | 72.047 | **32.664** | **2.21x** | 84.267 | **5.53** | **336.9** / 862.7 (-61%) |
| uniform | 8 | 1381 | 41.3% | 20.857 | **11.680** | **1.79x** | 25.498 | 4.36 | **156.4** / 313.8 (-50%) |
| uniform | 16 | 1794 | 50.3% | 64.441 | **28.146** | **2.29x** | 79.565 | 4.08 | **309.6** / 763.4 (-59%) |

<p align="center">
  <img src="assets/training_benchmark.png" alt="Training forward plus backward latency and peak VRAM comparison on Tesla T4" width="90%">
</p>

```bash
python tests/bench_training.py --batch-sizes 4 8 16 --max-lens 512 1024 2048 --distributions skewed uniform --save-plot assets/training_benchmark.png
```

### Inference — Cold-L2 Paged Decode Attention (`L_q = 1`, `H_q = 32, H_kv = 8, D = 64`)

| Dist | `B` | `MaxK` | `TotalLk` | Batched+Comp (ms) ↓ | Paged Eager (ms) ↓ | vs Comp ↑ | vs Eager ↑ | BW (GB/s) ↑ | Throughput (tok/s) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skewed | 1 | 44 | 44 | 0.377 | **0.014** | **27.01x** | 11.71x | 7.0 | 71,674 |
| skewed | 1 | 3712 | 3712 | 0.960 | **0.107** | **8.97x** | 7.72x | 71.1 | 9,348 |
| skewed | 8 | 996 | 5286 | 1.026 | **0.139** | **7.38x** | 6.68x | 78.4 | 57,577 |
| skewed | 8 | 3923 | 11989 | 2.092 | **0.188** | **11.15x** | 10.47x | 131.2 | 42,640 |
| skewed | 32 | 126 | 1546 | 0.385 | **0.033** | **11.81x** | 6.18x | 105.1 | **981,354** |
| skewed | 32 | 1018 | 15593 | 1.747 | **0.173** | **10.07x** | 10.84x | **185.6** | 184,468 |
| skewed | 64 | 1021 | 17934 | 3.754 | **0.207** | **18.15x** | **18.07x** | 180.1 | 309,406 |
| skewed | 64 | 4050 | 86245 | 14.804 | **2.619** | **5.65x** | 5.67x | 67.6 | 24,438 |
| uniform | 64 | 4080 | 143272 | 15.286 | **4.917** | **3.11x** | 3.13x | 59.8 | 13,015 |

### Inference — End-to-End Prefill (`TTFT`) + 32-Step Generation (`TPOT`) + Peak VRAM

| `B` | `MaxL` | Pad (%) | Base TTFT (ms) ↓ | Ours TTFT (ms) ↓ | Base TPOT (ms) ↓ | Ours TPOT (ms) ↓ | Base E2E (ms) ↓ | Ours E2E (ms) ↓ | E2E VRAM (Ours / Base, MB) ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 207 | 0.0% | **0.48** | 0.51 (0.95x) | 0.451 | **0.121** (3.74x) | 14.80 | **4.12** (3.59x) | **54.4** / 55.7 (-2%) |
| 4 | 1988 | 41.3% | 15.79 | **8.63** (1.83x) | 1.320 | **0.232** (5.70x) | 56.74 | **15.72** (3.61x) | **149.1** / 264.6 (-44%) |
| 16 | 998 | 56.6% | 21.86 | **11.53** (1.90x) | 2.281 | **0.306** (7.46x) | 95.07 | **20.87** (4.55x) | **208.4** / 472.2 (-56%) |
| 32 | 254 | 60.4% | 10.87 | **2.90** (**3.75x**) | 1.337 | **0.249** (5.37x) | 52.25 | **12.57** (4.16x) | **127.0** / 266.3 (-52%) |
| 32 | 511 | 65.4% | 20.02 | **8.52** (2.35x) | 2.359 | **0.322** (7.34x) | 94.26 | **18.82** (5.01x) | **186.7** / 482.4 (-61%) |
| 64 | 510 | 65.6% | 40.84 | **15.58** (2.62x) | 4.591 | **0.539** (**8.52x**) | 186.78 | **33.13** (**5.64x**) | **320.5** / 914.5 (**-65%**) |
| 96 | 255 | 62.1% | 28.55 | **11.95** (2.39x) | 3.721 | **0.534** (6.97x) | 148.07 | **28.84** (5.13x) | **273.7** / 698.6 (-61%) |

<p align="center">
  <img src="assets/inference_benchmark.png" alt="Inference paged decode latency, E2E TTFT/TPOT, and peak VRAM comparison on Tesla T4" width="90%">
</p>

```bash
python tests/bench_inference.py --batch-sizes 1 8 32 64 --max-lens 128 256 1024 2048 4096 --num-heads-q 32 --num-heads-kv 8 --distributions skewed uniform --gen-steps 32 --save-plot assets/inference_benchmark.png
```

---

## Requirements

| Component | Tested | Minimum |
| :--- | :--- | :--- |
| GPU | NVIDIA Tesla T4 (`SM75`) | Turing (`SM75`) or newer CUDA GPU |
| CUDA / Driver | CUDA `13.0` / Driver `580.82.07` | CUDA `11.8` |
| PyTorch | `2.14.1+cu130` | `2.0` |
| Python | `3.13` | `3.10` |
| C++ Compiler | C++20 (`g++` / `nvcc`) | C++20 (`PyTorch >= 2.1`) |

---

## Getting started

### Prerequisites

Install `PyTorch` with CUDA support along with `ninja`, `wheel`, and `setuptools`.

### Installation

> [!TIP]
> On Google Colab or compatible Python `3.13` / CUDA `13.0` environments, install directly from `wheels/` without waiting for NVCC compilation.

```bash
# Option 1: Install prebuilt v2 wheel
pip install --no-deps wheels/varlen_transformer-1.0.0-py3-none-any.whl
```

To build from source (`setup.py` automatically detects `third_party/cutlass` or `CUTLASS_PATH` and shallow-clones NVIDIA CUTLASS if missing):

```bash
# Option 2: Build from source
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .

# Optional: Package a reusable wheel into wheels/
python -m pip wheel --no-deps --no-build-isolation -w wheels .
```

### Usage

`VarlenFlashGQA` runs directly in **Pure Eager** mode (no `torch.compile` wrapper needed, bias-free `bias=False`) for Self-Attention training, Cross-Attention training, and Paged KV Prefill/Decode:

```python
import torch
from varlen_transformer import RotaryPositionalEmbedding, VarlenFlashGQA

device = "cuda"
embed_dim, num_heads_q, num_heads_kv, head_dim = 1024, 16, 4, 64

rope = RotaryPositionalEmbedding(head_dim, max_seq_len=2048, device=device)
model = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope).to(
    device=device, dtype=torch.float16
)

# Packed batch of 3 sequences: lengths [128, 512, 65] -> Total_Lq = 705
cu_seqlens = torch.tensor([0, 128, 640, 705], dtype=torch.int32, device=device)
x = torch.randn((705, embed_dim), device=device, dtype=torch.float16, requires_grad=True)

out = model(
    x,
    cu_seqlens_q=cu_seqlens,
    max_seqlen_q=512,
    is_causal=True,
)
out.sum().backward()
```

```python
# Self-Attention Prefill + Paged KV Decode (PAGE_SIZE = 16)
with torch.no_grad():
    prefill_out, kv_cache = model(
        x,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=512,
        is_causal=True,
        use_cache=True,
    )

    # Decode step (Lq = 1 token per sequence in batch B = 3)
    x_next = torch.randn((3, embed_dim), device=device, dtype=torch.float16)
    decode_out, kv_cache = model(x_next, kv_cache=kv_cache, use_cache=True, decode=True)

# Cross-Attention Training (packed Q and context in THD layout)
cross_attn = VarlenFlashGQA(
    embed_dim, num_heads_q, num_heads_kv, is_cross_attention=True, context_dim=512
).to(device=device, dtype=torch.float16)
context = torch.randn((320, 512), device=device, dtype=torch.float16, requires_grad=True)
cu_k = torch.tensor([0, 100, 220, 320], dtype=torch.int32, device=device)
cross_out = cross_attn(x, cu_seqlens_q=cu_seqlens, context=context, cu_seqlens_k=cu_k, max_seqlen_q=512, max_seqlen_k=120)
cross_out.sum().backward()
```

---

## Demo

Run the full v2 correctness suite (Pure THD `fwd`/`bwd` + strided slice check, Paged KV `prefill_to_paged_kv` / `paged_kv_append` / shuffled `block_table` decode, Self & Cross Attention Fwd+Bwd, 20-step Paged Decode, and `FakeTensor` registrations) or open the interactive Colab/Kaggle notebook [benchmark/demo_varlen_transformer.ipynb](benchmark/demo_varlen_transformer.ipynb):

```bash
python tests/test_correctness.py
```

---

## Project structure

```text
Varlen-Attention/
├── include/                  # CUDA/CuTe headers (mha_common.cuh, mha_ops.cuh)
├── src/                      # CUDA/C++ kernels & bindings (fwd, bwd, kv_cache, decode, gqa_fused)
├── varlen_transformer/       # Python package (ops.py, model.py, __init__.py)
├── tests/                    # Correctness & benchmark suites (utils, correctness, training, inference)
├── benchmark/                # End-to-end Colab demo notebook
├── assets/                   # Benchmark plots (training_benchmark.png, inference_benchmark.png)
├── docs/images/              # Animated SVG title
├── wheels/                   # Pre-built v2 wheel (.whl) directory
├── setup.py
├── requirements.txt
└── LICENSE                   # Apache License 2.0
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
