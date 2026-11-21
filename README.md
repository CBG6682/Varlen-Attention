<h1 align="center">
  <img src="docs/images/title.svg" alt="Varlen-Attention" width="720">
</h1>

<p align="center"><em>Variable-length FlashAttention, Fused RoPE, and Adaptive Split-K Flash Decoding for NVIDIA Turing (SM75).</em></p>

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

<details>
<summary><b>Table of Contents</b></summary>

- [Overview](#overview)
- [Benchmarks](#benchmarks)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [License](#license)

</details>

---

## Overview

`Varlen-Attention` (`varlen_transformer`) provides packed variable-length attention, fused rotary embeddings, and adaptive decode kernels engineered for NVIDIA Turing (`SM75`) GPUs with `PyTorch` Autograd and `torch.compile` support. On an NVIDIA Tesla T4 (`SM75`), it achieves up to **3.00x** faster training with **65%** lower peak VRAM, **9.22x** faster `L_q = 1` decode, and **4.15x** faster end-to-end token generation (`TPOT`) than compiled `SDPBackend.EFFICIENT_ATTENTION`.

### Why Varlen-Attention?

This library is built for students, researchers, and self-learners running Transformers on accessible Turing hardware (`Tesla T4`, `RTX 20 series`). The core features include:

- 🎯 **Built for Turing (`SM75`):** Targets T4 and RTX 20 series GPUs, where `Triton` kernels focus on Ampere (`SM80+`) and legacy `FlashAttention-1` requires downgrading `PyTorch` and C++ compilers.
- 📦 **Zero-Padding Varlen Layout:** Operates directly on packed `(H, Total_L, 64)` tensors, cutting peak VRAM by up to **68%** (`292.3 MB` vs `914.5 MB` at `B = 64, L = 510`).
- ⚡ **`torch.compile` Native:** Exposes `TORCH_LIBRARY` operators and `FakeTensor` registrations so `TorchInductor` compiles models without graph breaks, unlike opaque `F.scaled_dot_product_attention`.
- 🚀 **Prebuilt Distribution:** Ships a ready-to-install `.whl` in `wheels/` for modern `PyTorch` and C++20 environments.

---

## Benchmarks

**Setup:** NVIDIA Tesla T4 (15 GB, `SM75`), Driver `580.82.07`, CUDA `13.0`, `PyTorch 2.14.1+cu130`, FP16, `head_dim = 64`, `RoPE = True`, median latency over `rep = 50` (`warmup = 15`; `clear_l2 = True` for inference). Baseline uses `BatchedFlashGQABaselineSpeed` + `torch.compile(backend="inductor")` with FP16 additive masks aligned to multiples of 8 (`SDPBackend.EFFICIENT_ATTENTION`, `choice=2` verified).

### Training (`Fwd+Bwd`, `H_q = 16, H_kv = 4, D = 64, causal = True`)

- **Skewed (`B = 4, MaxLen = 1945`, 59.9% pad) — Baseline:** `16.763 ms`, `234.9 MB` VRAM | **Ours:** **`6.388 ms`** (**`2.62x`**), `4.83 TFLOP/s`, **`97.0 MB`** (-59% VRAM)
- **Skewed (`B = 8, MaxLen = 2020`, 59.2% pad) — Baseline:** `34.894 ms`, `444.4 MB` VRAM | **Ours:** **`13.756 ms`** (**`2.54x`**), `4.86 TFLOP/s`, **`172.4 MB`** (-61% VRAM)
- **Skewed (`B = 16, MaxLen = 471`, 72.1% pad) — Baseline:** `8.813 ms`, `221.6 MB` VRAM | **Ours:** **`2.937 ms`** (**`3.00x`**), `1.33 TFLOP/s`, **`77.1 MB`** (-65% VRAM)
- **Skewed (`B = 16, MaxLen = 2041`, 52.0% pad) — Baseline:** `74.173 ms`, `862.7 MB` VRAM | **Ours:** **`35.003 ms`** (**`2.12x`**), **`5.16 TFLOP/s`**, **`367.8 MB`** (-57% VRAM)

```bash
python tests/bench_training.py --batch-sizes 4 8 16 --max-lens 512 1024 2048 --distributions skewed uniform --save-plot assets/training_benchmark.png
```

### Inference (`H_q = 32, H_kv = 8, D = 64`, skewed)

- **Decode `L_q = 1` (`B = 8, MaxK = 3923`, `TotalLk = 11989`) — Baseline:** `2.059 ms` compiled | **Ours:** **`0.270 ms`** (**`7.62x`** compiled, `8.96x` eager), `91.1 GB/s`, `29,593 tok/s`
- **Decode `L_q = 1` (`B = 32, MaxK = 2034`, `TotalLk = 25431`) — Baseline:** `3.795 ms` compiled | **Ours:** **`0.412 ms`** (**`9.22x`** compiled, `6.39x` eager), **`127.2 GB/s`**, `77,736 tok/s`
- **Decode `L_q = 1` (`B = 32, MaxK = 4095`, `TotalLk = 50267`) — Baseline:** `7.711 ms` compiled | **Ours:** **`1.112 ms`** (**`6.94x`** compiled, `5.63x` eager), `92.8 GB/s`, `28,784 tok/s`
- **End-to-End (`B = 32, MaxL = 511`, 65.4% pad, 32 steps) — Baseline:** `19.93 ms` TTFT, `2.445 ms` TPOT, `97.57 ms` E2E, `482.4 MB` | **Ours:** **`9.27 ms`** TTFT (`2.15x`), **`0.699 ms`** TPOT (**`3.50x`**), **`29.39 ms`** E2E (**`3.32x`**), **`172.8 MB`** (-64% VRAM)
- **End-to-End (`B = 64, MaxL = 510`, 65.6% pad, 32 steps) — Baseline:** `40.39 ms` TTFT, `4.690 ms` TPOT, `189.92 ms` E2E, `914.5 MB` | **Ours:** **`16.17 ms`** TTFT (`2.50x`), **`1.129 ms`** TPOT (**`4.15x`**), **`50.60 ms`** E2E (**`3.75x`**), **`292.3 MB`** (**-68%** VRAM)

```bash
python tests/bench_inference.py --batch-sizes 1 8 32 64 --max-lens 128 256 1024 2048 4096 --num-heads-q 32 --num-heads-kv 8 --distributions skewed uniform --gen-steps 32 --save-plot assets/inference_benchmark.png
```

---

## Requirements

- **GPU:** NVIDIA Tesla T4 (`SM75`) tested; Turing (`SM75`) or newer CUDA GPU minimum.
- **CUDA and Driver:** CUDA `13.0` / Driver `580.82.07` tested; CUDA `11.8+` minimum.
- **PyTorch:** `2.14.1+cu130` tested; `2.0` minimum.
- **Python and Compiler:** Python `3.13` and C++20 (`g++` / `nvcc`) tested; Python `3.10` minimum (C++20 required for `PyTorch >= 2.1`).

---

## Getting started

### Prerequisites

Install `PyTorch` with CUDA support along with `ninja`, `wheel`, and `setuptools`.

### Installation

> [!TIP]
> On Python `3.13` and CUDA `13.0` (e.g., Google Colab), install the prebuilt wheel in `wheels/` to skip NVCC compilation.

```bash
pip install --no-deps wheels/varlen_transformer-1.0.0-py3-none-any.whl
python tests/test_correctness.py
```

To build from source instead (`setup.py` auto-clones NVIDIA CUTLASS into `third_party/cutlass` if missing):

```bash
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .
```

### Usage

```python
import torch
from varlen_transformer import RotaryPositionalEmbedding, VarlenFlashGQA

device = "cuda"
rope = RotaryPositionalEmbedding(64, max_seq_len=2048, device=device)
model = torch.compile(VarlenFlashGQA(1024, 16, 4, rope=rope).to(device=device, dtype=torch.float16))

cu_seqlens = torch.tensor([0, 128, 640, 705], dtype=torch.int32, device=device)
x = torch.randn((705, 1024), device=device, dtype=torch.float16, requires_grad=True)

out = model(x, cu_seqlens_q=cu_seqlens, max_seqlen_q=512, max_seqlen_k=512, is_causal=True)
out.sum().backward()

with torch.no_grad():
    _, kv_cache = model(x, cu_seqlens_q=cu_seqlens, max_seqlen_q=512, max_seqlen_k=512, is_causal=True, use_cache=True)
    decode_out, kv_cache = model(torch.randn((3, 1024), device=device, dtype=torch.float16), kv_cache=kv_cache, use_cache=True, decode=True)
```

---

## Project structure

```text
Varlen-Attention/
├── include/                  # CUDA/CuTe headers (mha_common.cuh, mha_ops.cuh)
├── src/                      # CUDA kernels and PyTorch TORCH_LIBRARY bindings
├── varlen_transformer/       # Python package (FakeTensor ops, VarlenKVCache, FlashGQA)
├── tests/                    # Correctness tests and training/inference benchmarks
├── benchmark/                # End-to-end Colab/Kaggle demo notebook
├── assets/                   # Generated benchmark figures
├── docs/images/              # Animated SVG title
├── wheels/                   # Prebuilt wheel package
└── setup.py                  # SM75 NVCC build and CUTLASS lookup
```

---

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
