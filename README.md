<div align="center">

# Varlen-Attention (`varlen_transformer`)

**Variable-Length FlashAttention, Fused RoPE & Adaptive Split-K Flash Decoding for NVIDIA Turing (`SM75`)**

[![CUDA](https://img.shields.io/badge/CUDA-SM75%20%2F%20Turing-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-gpus)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torch.compile](https://img.shields.io/badge/torch.compile-TorchInductor%20Ready-blueviolet)](https://pytorch.org/docs/stable/torch.compiler.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Motivation](#motivation) · [Benchmarks](#benchmarks-nvidia-tesla-t4--sm75) · [Installation](#installation) · [Quick Start](#quick-start) · [Demo Notebook](Benchmark/demo_varlen_transformer.ipynb)

</div>

---

## At-a-Glance Highlights (Tesla T4 — `SM75`)

| Training (`Fwd+Bwd`) | Decode `L_q = 1` (Cold L2) | E2E Generation (`TPOT`) | Peak VRAM Saved |
| :---: | :---: | :---: | :---: |
| Up to **3.00x** faster | Up to **9.22x** faster | Up to **4.15x** faster | Up to **-68%** VRAM |

---

## Motivation

`varlen_transformer` was built to help **students, researchers, and self-learners** train and run Transformer models efficiently on accessible **Turing (`SM75`)** GPUs such as **Google Colab / Kaggle Tesla T4** and **RTX 20 series** cards:

| Solution | Turing (`SM75`) Speed | `torch.compile` Fusion | Modern PyTorch & C++20 | Zero-Padding Varlen |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch SDPA** (`F.scaled_dot_product_attention`) | Falls back to slow `MATH` with `bool` masks | Black box (no custom fusion) | Yes | No (requires padding) |
| **Triton / FlexAttention** | Tuned for Ampere (`A100 / SM80+`) | Yes | Yes | Partial |
| **FlashAttention-1 (`v1.0.9`)** | Fast | No | Unmaintained (requires PyTorch & C++ downgrades) | Yes |
| **Varlen-Attention (`Ours`)** | **Optimized for `SM75`** | **Full `FakeTensor` + Inductor** | **Yes (prebuilt `.whl` included)** | **Yes (packed layout)** |

---

## Benchmarks (NVIDIA Tesla T4 — `SM75`)

> [!NOTE]
> **Benchmark Environment:** Measured on **NVIDIA Tesla T4 (15 GB, `SM75`)** with **PyTorch `2.14.1+cu130`**, **CUDA `13.0`**, `head_dim = 64`, FP16, `RoPE = True`.
> **Fair Baseline:** `BatchedFlashGQABaselineSpeed` compiled with `torch.compile(backend="inductor")`, using **FP16 additive padding masks (`0.0` / `-inf`)** aligned to multiples of 8 (`stride(-1) = 1`, `stride(-2) % 8 == 0`) to explicitly activate **`SDPBackend.EFFICIENT_ATTENTION`** (xFormers CUTLASS) on T4 (`choice=2` verified).

### 1. Training (`Fwd+Bwd`, `H_q = 16, H_kv = 4, D = 64, causal = True`)

<p align="center">
  <img src="assets/training_benchmark.png" alt="Training Benchmark on Tesla T4" width="95%">
</p>

| Dist | `B` | `MaxLen` | Pad Waste | Batched + Compile | Varlen + Compile (Ours) | Speedup | TFLOP/s | Peak VRAM (Ours / Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| skewed | 4 | 508 | 39.8% | 4.277 ms | 2.738 ms | **1.56x** | 1.31 | 56.0 / 78.8 MB (**-29%**) |
| skewed | 4 | 1945 | 59.9% | 16.763 ms | 6.388 ms | **2.62x** | 4.83 | 97.0 / 234.9 MB (**-59%**) |
| skewed | 8 | 2020 | 59.2% | 34.894 ms | 13.756 ms | **2.54x** | 4.86 | 172.4 / 444.4 MB (**-61%**) |
| skewed | 16 | 471 | 72.1% | 8.813 ms | 2.937 ms | **3.00x** | 1.33 | 77.1 / 221.6 MB (**-65%**) |
| skewed | 16 | 2041 | 52.0% | 74.173 ms | 35.003 ms | **2.12x** | 5.16 | 367.8 / 862.7 MB (**-57%**) |
| uniform | 8 | 1381 | 41.3% | 22.011 ms | 12.675 ms | **1.74x** | 4.02 | 169.8 / 313.8 MB (**-46%**) |

---

### 2. Inference (`H_q = 32, H_kv = 8, D = 64`)

<p align="center">
  <img src="assets/inference_benchmark.png" alt="Inference Benchmark on Tesla T4" width="95%">
</p>

#### Part A — Cold-L2 Decode Attention (`L_q = 1`) vs `SDPBackend.EFFICIENT_ATTENTION` + `torch.compile`

| Dist | `B` | `MaxK` | `TotalLk` | Batched + Compile | Ours + Compile | Compiled Speedup | Eager Speedup | BW (GB/s) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| skewed | 1 | 3712 | 3712 | 0.957 ms | 0.273 ms | **3.51x** | 7.82x | 27.9 | 3,667 tok/s |
| skewed | 8 | 996 | 5286 | 1.164 ms | 0.219 ms | **5.31x** | 7.95x | 49.7 | 36,507 tok/s |
| skewed | 8 | 3923 | 11989 | 2.059 ms | 0.270 ms | **7.62x** | 8.96x | 91.1 | 29,593 tok/s |
| skewed | 32 | 126 | 1546 | 0.280 ms | 0.082 ms | **3.41x** | 6.61x | 41.8 | 389,864 tok/s |
| skewed | 32 | 1018 | 15593 | 2.040 ms | 0.273 ms | **7.47x** | 8.78x | 117.9 | 117,165 tok/s |
| skewed | 32 | 2034 | 25431 | 3.795 ms | 0.412 ms | **9.22x** | 6.39x | 127.2 | 77,736 tok/s |
| skewed | 32 | 4095 | 50267 | 7.711 ms | 1.112 ms | **6.94x** | 5.63x | 92.8 | 28,784 tok/s |

#### Part B — End-to-End Prefill (`TTFT`) + 32-Step Generation (`TPOT`) + Peak VRAM

| `B` | `MaxL` | Pad Waste | Base TTFT | Ours TTFT | Base TPOT | Ours TPOT | Base E2E | Ours E2E | E2E VRAM (Ours / Base) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 1988 | 41.3% | 16.14 ms | 8.86 ms (1.82x) | 1.378 ms | 0.688 ms (**2.00x**) | 60.53 ms | 25.08 ms (**2.41x**) | 123.9 / 264.6 MB (**-53%**) |
| 16 | 998 | 56.6% | 21.19 ms | 11.33 ms (1.87x) | 2.295 ms | 0.858 ms (**2.68x**) | 95.01 ms | 32.42 ms (**2.93x**) | 202.9 / 472.2 MB (**-57%**) |
| 32 | 254 | 60.4% | 10.59 ms | 2.76 ms (3.83x) | 1.369 ms | 0.686 ms (**1.99x**) | 53.54 ms | 22.68 ms (**2.36x**) | 119.5 / 266.3 MB (**-55%**) |
| 32 | 511 | 65.4% | 19.93 ms | 9.27 ms (2.15x) | 2.445 ms | 0.699 ms (**3.50x**) | 97.57 ms | 29.39 ms (**3.32x**) | 172.8 / 482.4 MB (**-64%**) |
| 64 | 255 | 65.5% | 19.44 ms | 8.20 ms (2.37x) | 2.625 ms | 0.731 ms (**3.59x**) | 102.12 ms | 31.40 ms (**3.25x**) | 172.3 / 482.5 MB (**-64%**) |
| 64 | 510 | 65.6% | 40.39 ms | 16.17 ms (2.50x) | 4.690 ms | 1.129 ms (**4.15x**) | 189.92 ms | 50.60 ms (**3.75x**) | 292.3 / 914.5 MB (**-68%**) |
| 96 | 255 | 62.1% | 28.25 ms | 12.88 ms (2.19x) | 3.824 ms | 1.077 ms (**3.55x**) | 151.46 ms | 46.93 ms (**3.23x**) | 249.9 / 698.6 MB (**-64%**) |

---

## Installation

### Requirements
- **GPU:** NVIDIA Turing (`SM75` — Tesla T4, RTX 2060 / 2070 / 2080) or compatible CUDA GPU
- **Software:** CUDA Toolkit + PyTorch `>= 2.0` (C++20 compiler required for PyTorch `>= 2.1`)

### Option 1: Install Pre-built Wheel (Fastest)

> [!TIP]
> On Google Colab or compatible Python 3.13 / CUDA 13.0 environments, install directly from `wheels/` without waiting for NVCC compilation:

```bash
pip install --no-deps wheels/varlen_transformer-1.0.0-py3-none-any.whl
```

### Option 2: Build from Source

`setup.py` automatically detects `third_party/cutlass` (or `CUTLASS_PATH`) and shallow-clones NVIDIA CUTLASS if missing:

```bash
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .
```

To package a reusable `.whl` into `wheels/`:

```bash
python -m pip wheel --no-deps --no-build-isolation -w wheels .
```

---

## Quick Start

### 1. Packed Varlen Training (`VarlenFlashGQA` + `torch.compile`)

```python
import torch
from varlen_transformer import RotaryPositionalEmbedding, VarlenFlashGQA

device = "cuda"
embed_dim, num_heads_q, num_heads_kv, head_dim = 1024, 16, 4, 64

rope = RotaryPositionalEmbedding(head_dim, max_seq_len=2048, device=device)
model = VarlenFlashGQA(embed_dim, num_heads_q, num_heads_kv, rope=rope).to(
    device=device, dtype=torch.float16
)
compiled_model = torch.compile(model, backend="inductor")

# Packed batch of 3 sequences: lengths [128, 512, 65] -> Total_Lq = 705
cu_seqlens = torch.tensor([0, 128, 640, 705], dtype=torch.int32, device=device)
x = torch.randn((705, embed_dim), device=device, dtype=torch.float16, requires_grad=True)

out = compiled_model(
    x,
    cu_seqlens_q=cu_seqlens,
    max_seqlen_q=512,
    max_seqlen_k=512,
    is_causal=True,
)
out.sum().backward()
```

### 2. Varlen Prefill + Autoregressive Decode (`VarlenKVCache`)

```python
with torch.no_grad():
    prefill_out, kv_cache = model(
        x,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=512,
        max_seqlen_k=512,
        is_causal=True,
        use_cache=True,
    )

    # Decode step (Lq = 1 token per sequence in batch B = 3)
    x_next = torch.randn((3, embed_dim), device=device, dtype=torch.float16)
    decode_out, kv_cache = model(x_next, kv_cache=kv_cache, use_cache=True, decode=True)
```

---

## Running Tests & Benchmarks

```bash
# 1. Numerical correctness (Fwd, Bwd, Decode, E2E GQA, VarlenKVCache, torch.compile)
python tests/test_correctness.py

# 2. Training (Fwd & Fwd+Bwd) speed & VRAM vs torch.compile Baseline
python tests/bench_training.py --save-plot assets/training_benchmark.png

# 3. Inference (Cold-L2 Decode L=1 & E2E Prefill+Generation) vs torch.compile Baseline
python tests/bench_inference.py --gen-steps 32 --save-plot assets/inference_benchmark.png
```

For an interactive end-to-end walkthrough (build, wheel packaging, verification, benchmarks, and `ncu` profiling), open [Benchmark/demo_varlen_transformer.ipynb](Benchmark/demo_varlen_transformer.ipynb).

---

## Project Structure

```text
Varlen-Attention/
├── include/                         # CUDA/CuTe headers (mha_common.cuh, mha_ops.cuh)
├── src/                             # CUDA kernels & PyTorch bindings (fwd, bwd, decode)
├── varlen_transformer/              # Python package (ops.py, model.py)
├── tests/                           # Correctness & benchmark suites
├── Benchmark/                       # End-to-end demo notebook
├── assets/                          # Benchmark plots
├── wheels/                          # Pre-built wheel (.whl) directory
├── setup.py
├── requirements.txt
└── LICENSE                          # Apache License 2.0
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
