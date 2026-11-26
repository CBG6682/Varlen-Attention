"""Numerical correctness tests comparing v2 THD CUDA/Paged-KV/Fused-GQA kernels vs PyTorch reference."""

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import torch

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tests.utils import (
    check_allclose,
    pack_varlen_tensor,
    seed_everything,
    unpack_varlen_tensor,
)
from varlen_transformer import (
    HAS_CUDA_EXT,
    BatchedFlashGQABaselineSpeed,
    RotaryPositionalEmbedding,
    VarlenFlashGQA,
    VarlenFlashGQABaselineCorrectness,
    VarlenKVCache,
    fused_mha_varlen_bwd,
    fused_mha_varlen_decode,
    fused_mha_varlen_fwd,
    paged_kv_append,
    prefill_to_paged_kv,
    varlen_attention_ref,
    varlen_decode_ref,
    varlen_flash_attention,
    varlen_flash_decode,
)


def _make_cu_seqlens(lengths: Sequence[int], device: torch.device) -> Tuple[torch.Tensor, int]:
    lens_t = torch.tensor(list(lengths), dtype=torch.int32, device=device)
    cu = torch.zeros((len(lengths) + 1,), dtype=torch.int32, device=device)
    if len(lengths) > 0:
        torch.cumsum(lens_t, dim=0, out=cu[1:])
    max_len = max(lengths) if lengths else 0
    return cu, max_len


def _require_gpu_and_ext() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run varlen_transformer correctness tests.")
    if not HAS_CUDA_EXT:
        raise RuntimeError(
            "varlen_transformer._C extension is not compiled. Run `pip install -e . --no-build-isolation`."
        )
    return torch.device("cuda")


def test_varlen_fwd_bwd_kernels(quick: bool = False, device: Optional[torch.device] = None) -> bool:
    """Test raw `fused_mha_varlen_fwd`, `fused_mha_varlen_bwd`, and `varlen_flash_attention` in Pure THD."""
    print("\n" + "=" * 80)
    print("1. Testing Varlen Forward & Backward CUDA Kernels (Pure THD) vs Reference")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    head_dim = 64
    all_passed = True

    if quick:
        cases = [
            ("GQA_causal_rope", [1, 17, 65, 129], [1, 17, 65, 129], 8, 2, True, True, [0, 5, 12, 3]),
            ("MHA_noncausal_norope", [15, 64, 127], [15, 64, 127], 8, 8, False, False, None),
            ("CrossAttn_rope_offsets", [19, 70], [45, 130], 8, 2, False, True, [4, 11]),
        ]
    else:
        cases = [
            ("MHA_causal_rope_odd", [1, 17, 63, 64, 65, 129], [1, 17, 63, 64, 65, 129], 8, 8, True, True, None),
            ("GQA_8_2_causal_rope_offsets", [1, 17, 63, 64, 65, 127, 257], [1, 17, 63, 64, 65, 127, 257], 8, 2, True, True, [0, 3, 17, 5, 29, 8, 13]),
            ("GQA_16_4_noncausal_rope", [31, 64, 128, 193], [31, 64, 128, 193], 16, 4, False, True, [7, 0, 15, 3]),
            ("GQA_8_1_causal_norope", [9, 64, 130], [9, 64, 130], 8, 1, True, False, None),
            ("CrossAttn_noncausal_rope", [19, 65, 3], [80, 129, 33], 8, 2, False, True, [0, 11, 6]),
            ("CrossAttn_no_rope", [32, 77], [16, 129], 8, 4, False, False, None),
        ]

    for name, q_lens, k_lens, h_q, h_kv, is_causal, use_rope, offset_list in cases:
        cu_q, max_q = _make_cu_seqlens(q_lens, dev)
        cu_k, max_k = _make_cu_seqlens(k_lens, dev)
        total_lq = int(cu_q[-1].item())
        total_lk = int(cu_k[-1].item())

        q = torch.randn((total_lq, h_q, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
        k = torch.randn((total_lk, h_kv, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
        v = torch.randn((total_lk, h_kv, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
        grad_o = torch.randn_like(q) * 0.25

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        offsets = None
        if offset_list is not None:
            offsets = torch.tensor(offset_list, dtype=torch.int32, device=dev)

        cos = sin = None
        if use_rope:
            max_off = max(offset_list) if offset_list is not None else 0
            rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max(max_q, max_k) + max_off + 64, device=dev)
            cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16)

        with torch.no_grad():
            o_raw, lse_raw = fused_mha_varlen_fwd(
                q.detach(), k.detach(), v.detach(), cu_q, cu_k, max_q,
                cos=cos, sin=sin, is_causal=is_causal, offsets=offsets,
            )
            dq_raw, dk_raw, dv_raw = fused_mha_varlen_bwd(
                grad_o, q.detach(), k.detach(), v.detach(), o_raw, lse_raw,
                cu_q, cu_k, offsets, max_k, cos=cos, sin=sin, is_causal=is_causal,
            )

        o_ref, lse_ref = varlen_attention_ref(
            q_ref, k_ref, v_ref, cu_q, cu_k, cos=cos, sin=sin, is_causal=is_causal, offsets=offsets
        )
        o_ref.backward(grad_o)

        o_auto = varlen_flash_attention(
            q, k, v, cu_q, cu_k, max_q, max_k, cos=cos, sin=sin, is_causal=is_causal, offsets=offsets
        )
        o_auto.backward(grad_o)

        print(f"\n-- Case [{name}] q_lens={q_lens}, k_lens={k_lens}, H=({h_q},{h_kv}), causal={is_causal}, rope={use_rope} --")
        all_passed &= check_allclose(f"{name}.fwd_o", o_raw, o_ref, atol=1.5e-2, rtol=1.5e-2)[0]
        all_passed &= check_allclose(f"{name}.fwd_lse", lse_raw, lse_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.bwd_dq", q.grad, q_ref.grad, atol=2.5e-2, rtol=2.5e-2)[0]
        all_passed &= check_allclose(f"{name}.bwd_dk", k.grad, k_ref.grad, atol=2.5e-2, rtol=2.5e-2)[0]
        all_passed &= check_allclose(f"{name}.bwd_dv", v.grad, v_ref.grad, atol=2.0e-2, rtol=2.0e-2)[0]
        all_passed &= check_allclose(f"{name}.raw_vs_auto_dq", dq_raw, q.grad, atol=1e-4, rtol=1e-4)[0]
        all_passed &= check_allclose(f"{name}.raw_vs_auto_dk", dk_raw, k.grad, atol=1e-4, rtol=1e-4)[0]
        all_passed &= check_allclose(f"{name}.raw_vs_auto_dv", dv_raw, v.grad, atol=1e-4, rtol=1e-4)[0]

    # Zero-copy strided slice test from packed QKV buffer
    print("\n-- Case [strided_qkv_slice_zero_copy] --")
    cu_s, max_s = _make_cu_seqlens([19, 65, 130], dev)
    total_s = int(cu_s[-1].item())
    h_q_s, h_kv_s = 8, 2
    qkv_packed = torch.randn((total_s, h_q_s + 2 * h_kv_s, head_dim), device=dev, dtype=torch.float16)
    qs = qkv_packed[:, :h_q_s, :]
    ks = qkv_packed[:, h_q_s : h_q_s + h_kv_s, :]
    vs = qkv_packed[:, h_q_s + h_kv_s :, :]
    assert not qs.is_contiguous() and not ks.is_contiguous() and not vs.is_contiguous()
    go_s = torch.randn((total_s, h_q_s, head_dim), device=dev, dtype=torch.float16) * 0.25
    rope_s = RotaryPositionalEmbedding(head_dim, max_seq_len=max_s + 32, device=dev)
    cos_s, sin_s = rope_s.get_cos_sin(device=dev, dtype=torch.float16)

    with torch.no_grad():
        os_raw, lses_raw = fused_mha_varlen_fwd(qs, ks, vs, cu_s, cu_s, max_s, cos=cos_s, sin=sin_s, is_causal=True)
        dqs_raw, dks_raw, dvs_raw = fused_mha_varlen_bwd(
            go_s, qs, ks, vs, os_raw, lses_raw, cu_s, cu_s, None, max_s, cos=cos_s, sin=sin_s, is_causal=True
        )
    qs_ref = qs.contiguous().detach().requires_grad_(True)
    ks_ref = ks.contiguous().detach().requires_grad_(True)
    vs_ref = vs.contiguous().detach().requires_grad_(True)
    os_ref, _ = varlen_attention_ref(qs_ref, ks_ref, vs_ref, cu_s, cu_s, cos=cos_s, sin=sin_s, is_causal=True)
    os_ref.backward(go_s)
    all_passed &= check_allclose("strided_qkv.fwd_o", os_raw, os_ref, atol=1.5e-2, rtol=1.5e-2)[0]
    all_passed &= check_allclose("strided_qkv.bwd_dq", dqs_raw, qs_ref.grad, atol=2.5e-2, rtol=2.5e-2)[0]
    all_passed &= check_allclose("strided_qkv.bwd_dk", dks_raw, ks_ref.grad, atol=2.5e-2, rtol=2.5e-2)[0]
    all_passed &= check_allclose("strided_qkv.bwd_dv", dvs_raw, vs_ref.grad, atol=2.0e-2, rtol=2.0e-2)[0]

    # Edge cases: empty sequences inside a batch and an empty batch
    print("\n-- Case [edge_zero_len_and_empty_batch] --")
    cu_q_edge, max_q_edge = _make_cu_seqlens([0, 17, 0, 65], dev)
    cu_k_edge, max_k_edge = _make_cu_seqlens([0, 17, 0, 65], dev)
    total_edge = int(cu_q_edge[-1].item())
    q_e = torch.randn((total_edge, 8, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
    k_e = torch.randn((total_edge, 2, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
    v_e = torch.randn((total_edge, 2, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
    q_e_ref = q_e.detach().clone().requires_grad_(True)
    k_e_ref = k_e.detach().clone().requires_grad_(True)
    v_e_ref = v_e.detach().clone().requires_grad_(True)
    go_e = torch.randn_like(q_e) * 0.25

    o_e = varlen_flash_attention(q_e, k_e, v_e, cu_q_edge, cu_k_edge, max_q_edge, max_k_edge, is_causal=True)
    o_e.backward(go_e)
    o_e_ref, _ = varlen_attention_ref(q_e_ref, k_e_ref, v_e_ref, cu_q_edge, cu_k_edge, is_causal=True)
    o_e_ref.backward(go_e)
    all_passed &= check_allclose("edge_zero_seq.fwd", o_e, o_e_ref, atol=1.5e-2, rtol=1.5e-2)[0]
    all_passed &= check_allclose("edge_zero_seq.dq", q_e.grad, q_e_ref.grad, atol=2.5e-2, rtol=2.5e-2)[0]
    all_passed &= check_allclose("edge_zero_seq.dk", k_e.grad, k_e_ref.grad, atol=2.5e-2, rtol=2.5e-2)[0]
    all_passed &= check_allclose("edge_zero_seq.dv", v_e.grad, v_e_ref.grad, atol=2.0e-2, rtol=2.0e-2)[0]

    cu_empty = torch.zeros((1,), dtype=torch.int32, device=dev)
    q_empty = torch.empty((0, 8, head_dim), device=dev, dtype=torch.float16)
    k_empty = torch.empty((0, 2, head_dim), device=dev, dtype=torch.float16)
    v_empty = torch.empty((0, 2, head_dim), device=dev, dtype=torch.float16)
    o_empty, lse_empty = fused_mha_varlen_fwd(q_empty, k_empty, v_empty, cu_empty, cu_empty, 0)
    empty_ok = (o_empty.shape == (0, 8, head_dim)) and (lse_empty.shape == (8, 0))
    print(f"[{'PASS' if empty_ok else 'FAIL'}] edge_empty_batch.shapes             o={tuple(o_empty.shape)} lse={tuple(lse_empty.shape)}")
    all_passed &= empty_ok

    return all_passed


def test_varlen_decode_kernels(quick: bool = False, device: Optional[torch.device] = None) -> bool:
    """Test Paged KV Cache ops (`prefill_to_paged_kv`, `paged_kv_append`) and `fused_mha_varlen_decode`."""
    print("\n" + "=" * 80)
    print("2. Testing Paged KV Cache & Split-K Flash Decoding vs Reference")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    head_dim = 64
    all_passed = True

    if quick:
        cases = [
            ("decode_short_rope", [1, 16, 17, 65], 8, 2, True, [0, 3, 9, 4], False),
            ("decode_multisplit_shuffled", [19, 257, 1025], 16, 4, True, None, True),
            ("decode_no_rope", [31, 513], 8, 8, False, None, False),
        ]
    else:
        cases = [
            ("decode_short_no_rope", [1, 7, 15, 16, 17], 8, 8, False, None, False),
            ("decode_gqa_rope_offsets_shuffled", [1, 15, 16, 17, 64, 127, 255], 8, 2, True, [2, 0, 11, 5, 19, 3, 8], True),
            ("decode_multisplit_long_rope", [1, 65, 257, 513, 1025], 16, 4, True, [0, 12, 4, 33, 7], False),
            ("decode_gqa_8_1_long", [129, 769, 1537], 8, 1, True, None, True),
            ("decode_edge_zero_seq_in_batch", [0, 33, 0, 260], 8, 2, True, [0, 5, 0, 10], False),
        ]

    for name, k_lens, h_q, h_kv, use_rope, offset_list, shuffle_pages in cases:
        cu_k, max_k = _make_cu_seqlens(k_lens, dev)
        B = len(k_lens)
        total_lk = int(cu_k[-1].item())

        q = torch.randn((B, h_q, head_dim), device=dev, dtype=torch.float16)
        k_raw = torch.randn((total_lk, h_kv, head_dim), device=dev, dtype=torch.float16)
        v_raw = torch.randn((total_lk, h_kv, head_dim), device=dev, dtype=torch.float16)

        offsets = torch.tensor(offset_list, dtype=torch.int32, device=dev) if offset_list is not None else None
        cos = sin = None
        if use_rope:
            max_off = max(offset_list) if offset_list is not None else 0
            rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_k + max_off + 64, device=dev)
            cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16)

        cache = VarlenKVCache.for_prefill(
            cu_seqlens_k=cu_k,
            num_heads_kv=h_kv,
            head_dim=head_dim,
            max_seqlen_k=max_k,
            offsets=offsets,
            device=dev,
            dtype=torch.float16,
        )
        if shuffle_pages and cache.num_pages > 1:
            perm = torch.randperm(cache.num_pages, device=dev, dtype=torch.int32)
            cache.block_table = perm.view(B, cache.max_pages_per_seq).contiguous()

        with torch.no_grad():
            prefill_to_paged_kv(
                k_raw, v_raw, cu_k,
                cache.k_cache, cache.v_cache, cache.block_table, cache.cache_seqlens,
                max_k,
            )
            o_raw = fused_mha_varlen_decode(
                q, cache.k_cache, cache.v_cache, cache.block_table, cache.cache_seqlens, max_k,
                cos=cos, sin=sin, offsets=offsets,
            )
            o_hi = varlen_flash_decode(
                q, cache.k_cache, cache.v_cache, cache.block_table, cache.cache_seqlens, max_k,
                cos=cos, sin=sin, offsets=offsets,
            )
            o_direct = varlen_flash_decode(
                q, cache.k_cache, cache.v_cache, cache.block_table, cache.cache_seqlens, max_k,
                cos=cos, sin=sin, offsets=offsets, use_splitk=False,
            )
            o_splitk = varlen_flash_decode(
                q, cache.k_cache, cache.v_cache, cache.block_table, cache.cache_seqlens, max_k,
                cos=cos, sin=sin, offsets=offsets, use_splitk=True,
            )
            o_ref = varlen_decode_ref(
                q, cache.k_cache, cache.v_cache, cache.block_table, cache.cache_seqlens,
                cos=cos, sin=sin, offsets=offsets,
            )

        no_nan = bool(
            torch.isfinite(o_raw).all().item()
            and torch.isfinite(o_direct).all().item()
            and torch.isfinite(o_splitk).all().item()
        )
        print(f"\n-- Case [{name}] k_lens={k_lens}, H=({h_q},{h_kv}), rope={use_rope}, shuffled={shuffle_pages}, finite={no_nan} --")
        all_passed &= no_nan
        all_passed &= check_allclose(f"{name}.decode_auto", o_raw, o_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.decode_hi", o_hi, o_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.decode_direct", o_direct, o_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.decode_splitk", o_splitk, o_ref, atol=1e-2, rtol=1e-2)[0]

    # Explicit page-boundary append check (14 -> 15 -> 16 -> 17 -> 18) with strided new_k/new_v
    print("\n-- Case [paged_kv_append_boundary_cross] --")
    init_lens = [14, 15, 16, 31]
    cu_b, max_b = _make_cu_seqlens(init_lens, dev)
    total_b = int(cu_b[-1].item())
    h_qb, h_kvb = 8, 2
    k_b = torch.randn((total_b, h_kvb, head_dim), device=dev, dtype=torch.float16)
    v_b = torch.randn((total_b, h_kvb, head_dim), device=dev, dtype=torch.float16)
    cache_b = VarlenKVCache.from_packed(k_b, v_b, cu_b, max_seqlen_k=max_b)
    rope_b = RotaryPositionalEmbedding(head_dim, max_seq_len=64, device=dev)
    cos_b, sin_b = rope_b.get_cos_sin(device=dev, dtype=torch.float16)

    for step in range(4):
        qkv_step = torch.randn((len(init_lens), h_qb + 2 * h_kvb, head_dim), device=dev, dtype=torch.float16)
        q_s = qkv_step[:, :h_qb, :]
        nk_s = qkv_step[:, h_qb : h_qb + h_kvb, :]
        nv_s = qkv_step[:, h_qb + h_kvb :, :]
        with torch.no_grad():
            cache_b.ensure_capacity_for_append()
            paged_kv_append(nk_s, nv_s, cache_b.k_cache, cache_b.v_cache, cache_b.block_table, cache_b.cache_seqlens)
            cache_b.advance_seqlens_cpu(1)
            out_s = fused_mha_varlen_decode(
                q_s, cache_b.k_cache, cache_b.v_cache, cache_b.block_table, cache_b.cache_seqlens,
                cache_b.max_seqlen_k, cos=cos_b, sin=sin_b,
            )
            ref_s = varlen_decode_ref(
                q_s, cache_b.k_cache, cache_b.v_cache, cache_b.block_table, cache_b.cache_seqlens,
                cos=cos_b, sin=sin_b,
            )
        all_passed &= check_allclose(f"append_cross_step_{step}", out_s, ref_s, atol=1e-2, rtol=1e-2)[0]

    return all_passed


def test_end_to_end_modules(device: Optional[torch.device] = None) -> bool:
    """Test `VarlenFlashGQA` (Self & Cross Attention, Prefill + Paged Decode) vs references."""
    print("\n" + "=" * 80)
    print("3. Testing Fused Eager GQA Modules (Self & Cross Train + Paged Prefill/Decode)")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    all_passed = True

    embed_dim, h_q, h_kv, head_dim = 512, 8, 2, 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=1024, device=dev)

    # Verify bias=True is rejected cleanly in v2
    bias_rejected = False
    try:
        VarlenFlashGQA(embed_dim, h_q, h_kv, rope=rope, bias=True)
    except ValueError:
        bias_rejected = True
    print(f"[{'PASS' if bias_rejected else 'FAIL'}] bias_free_enforcement                rejected_bias_true={bias_rejected}")
    all_passed &= bias_rejected

    mod_cuda = VarlenFlashGQA(embed_dim, h_q, h_kv, rope=rope, bias=False).to(device=dev, dtype=torch.float16)
    mod_vref = VarlenFlashGQABaselineCorrectness(embed_dim, h_q, h_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )
    mod_batched = BatchedFlashGQABaselineSpeed(embed_dim, h_q, h_kv, rope=rope, bias=False).to(
        device=dev, dtype=torch.float16
    )

    state = mod_cuda.state_dict()
    mod_vref.load_state_dict(state)
    mod_batched.load_state_dict(state)

    # --- 3A. Self-Attention Training Forward + Backward ---
    lengths = [19, 64, 130, 45]
    cu_seqlens, max_len = _make_cu_seqlens(lengths, dev)
    total_tokens = int(cu_seqlens[-1].item())

    x_cuda = torch.randn((total_tokens, embed_dim), device=dev, dtype=torch.float16, requires_grad=True)
    x_vref = x_cuda.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_cuda) * 0.1

    out_cuda = mod_cuda(
        x_cuda, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len, is_causal=True
    )
    out_cuda.backward(grad_out)

    out_vref = mod_vref(
        x_vref, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len, is_causal=True
    )
    out_vref.backward(grad_out)

    all_passed &= check_allclose("e2e_self_train.out", out_cuda, out_vref, atol=2e-2, rtol=2e-2)[0]
    all_passed &= check_allclose("e2e_self_train.dx", x_cuda.grad, x_vref.grad, atol=3e-2, rtol=3e-2)[0]
    all_passed &= check_allclose(
        "e2e_self_train.dqkv_weight", mod_cuda.qkv_proj.weight.grad, mod_vref.qkv_proj.weight.grad, atol=4e-2, rtol=4e-2
    )[0]
    all_passed &= check_allclose(
        "e2e_self_train.dout_weight", mod_cuda.out_proj.weight.grad, mod_vref.out_proj.weight.grad, atol=4e-2, rtol=4e-2
    )[0]

    # Cross-check Forward vs BatchedFlashGQABaselineSpeed (equal lengths check against SDPA)
    eq_len = 64
    B_eq = 4
    cu_eq, max_eq = _make_cu_seqlens([eq_len] * B_eq, dev)
    x_batched = torch.randn((B_eq, eq_len, embed_dim), device=dev, dtype=torch.float16)
    x_packed_eq = pack_varlen_tensor(x_batched, cu_eq)
    with torch.no_grad():
        o_k_eq = mod_cuda(x_packed_eq, cu_seqlens_q=cu_eq, max_seqlen_q=max_eq, is_causal=True)
        o_b_eq = mod_batched(x_batched, is_causal=True)
    all_passed &= check_allclose(
        "e2e_vs_batched_sdpa_causal", unpack_varlen_tensor(o_k_eq, cu_eq, max_eq), o_b_eq, atol=2e-2, rtol=2e-2
    )[0]

    # --- 3B. Cross-Attention Training Forward + Backward ---
    context_dim = 384
    cross_cuda = VarlenFlashGQA(
        embed_dim, h_q, h_kv, bias=False, is_cross_attention=True, context_dim=context_dim
    ).to(device=dev, dtype=torch.float16)
    cross_vref = VarlenFlashGQABaselineCorrectness(
        embed_dim, h_q, h_kv, bias=False, is_cross_attention=True, context_dim=context_dim
    ).to(device=dev, dtype=torch.float16)
    cross_vref.load_state_dict(cross_cuda.state_dict())

    q_lens_c = [17, 65, 29]
    k_lens_c = [48, 129, 35]
    cu_qc, max_qc = _make_cu_seqlens(q_lens_c, dev)
    cu_kc, max_kc = _make_cu_seqlens(k_lens_c, dev)
    tot_qc = int(cu_qc[-1].item())
    tot_kc = int(cu_kc[-1].item())

    xc_cuda = torch.randn((tot_qc, embed_dim), device=dev, dtype=torch.float16, requires_grad=True)
    ctxc_cuda = torch.randn((tot_kc, context_dim), device=dev, dtype=torch.float16, requires_grad=True)
    xc_vref = xc_cuda.detach().clone().requires_grad_(True)
    ctxc_vref = ctxc_cuda.detach().clone().requires_grad_(True)
    gc_out = torch.randn_like(xc_cuda) * 0.1

    out_cc = cross_cuda(
        xc_cuda, cu_seqlens_q=cu_qc, context=ctxc_cuda, cu_seqlens_k=cu_kc,
        max_seqlen_q=max_qc, max_seqlen_k=max_kc, is_causal=False,
    )
    out_cc.backward(gc_out)

    out_cr = cross_vref(
        xc_vref, cu_seqlens_q=cu_qc, context=ctxc_vref, cu_seqlens_k=cu_kc,
        max_seqlen_q=max_qc, max_seqlen_k=max_kc, is_causal=False,
    )
    out_cr.backward(gc_out)

    all_passed &= check_allclose("e2e_cross_train.out", out_cc, out_cr, atol=2e-2, rtol=2e-2)[0]
    all_passed &= check_allclose("e2e_cross_train.dx", xc_cuda.grad, xc_vref.grad, atol=3e-2, rtol=3e-2)[0]
    all_passed &= check_allclose("e2e_cross_train.dcontext", ctxc_cuda.grad, ctxc_vref.grad, atol=3e-2, rtol=3e-2)[0]
    all_passed &= check_allclose("e2e_cross_train.dw_q", cross_cuda.q_proj.weight.grad, cross_vref.q_proj.weight.grad, atol=4e-2, rtol=4e-2)[0]
    all_passed &= check_allclose("e2e_cross_train.dw_kv", cross_cuda.kv_proj.weight.grad, cross_vref.kv_proj.weight.grad, atol=4e-2, rtol=4e-2)[0]
    all_passed &= check_allclose("e2e_cross_train.dw_out", cross_cuda.out_proj.weight.grad, cross_vref.out_proj.weight.grad, atol=4e-2, rtol=4e-2)[0]

    # --- 3C. Prefill + 20-step Autoregressive Paged Decode (crosses PAGE_SIZE=16 boundaries) ---
    with torch.no_grad():
        x_prefill = torch.randn((total_tokens, embed_dim), device=dev, dtype=torch.float16)
        pre_cuda, cache_cuda = mod_cuda(
            x_prefill, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, is_causal=True, use_cache=True
        )
        pre_vref, cache_vref = mod_vref(
            x_prefill, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, is_causal=True, use_cache=True
        )
        all_passed &= check_allclose("e2e_prefill_with_cache", pre_cuda, pre_vref, atol=2e-2, rtol=2e-2)[0]

        for step in range(20):
            x_step = torch.randn((len(lengths), embed_dim), device=dev, dtype=torch.float16)
            dec_cuda = mod_cuda(x_step, kv_cache=cache_cuda, decode=True)
            dec_vref = mod_vref(x_step, kv_cache=cache_vref, decode=True)
            if step in (0, 1, 15, 16, 19):
                all_passed &= check_allclose(f"e2e_decode_step_{step}", dec_cuda, dec_vref, atol=2e-2, rtol=2e-2)[0]
            else:
                all_passed &= bool(torch.allclose(dec_cuda.float(), dec_vref.float(), atol=2e-2, rtol=2e-2))

    return all_passed


def test_faketensor_registrations(device: Optional[torch.device] = None) -> bool:
    """Verify FakeTensor / meta kernel shape inference for all TORCH_LIBRARY ops."""
    print("\n" + "=" * 80)
    print("4. Testing TORCH_LIBRARY FakeTensor Registrations (All 11 Ops)")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        print("[WARN] FakeTensorMode unavailable; skipping FakeTensor check.")
        return True

    ns = torch.ops.varlen_transformer
    all_passed = True
    with FakeTensorMode():
        B, total_l, total_lk, h_q, h_kv, d = 3, 96, 128, 8, 2, 64
        embed_dim, ctx_dim = h_q * d, 384
        cu_q = torch.empty((B + 1,), device=dev, dtype=torch.int32)
        cu_k = torch.empty((B + 1,), device=dev, dtype=torch.int32)
        cos = torch.empty((256, d), device=dev, dtype=torch.float16)
        sin = torch.empty((256, d), device=dev, dtype=torch.float16)
        offsets = torch.empty((B,), device=dev, dtype=torch.int32)

        q = torch.empty((total_l, h_q, d), device=dev, dtype=torch.float16)
        k = torch.empty((total_lk, h_kv, d), device=dev, dtype=torch.float16)
        v = torch.empty((total_lk, h_kv, d), device=dev, dtype=torch.float16)

        o, lse = ns.fused_mha_varlen_fwd(q, k, v, cu_q, cu_k, 48, cos, sin, False, offsets)
        dq, dk, dv = ns.fused_mha_varlen_bwd(o, q, k, v, o, lse, cu_q, cu_k, offsets, 64, cos, sin, False)
        ok_core = (o.shape == (total_l, h_q, d)) and (lse.shape == (h_q, total_l)) and (dq.shape == q.shape) and (dk.shape == k.shape) and (dv.shape == v.shape)
        print(f"[{'PASS' if ok_core else 'FAIL'}] faketensor.core_fwd_bwd              o={tuple(o.shape)} lse={tuple(lse.shape)}")
        all_passed &= ok_core

        k_cache = torch.empty((24, 16, h_kv, d), device=dev, dtype=torch.float16)
        v_cache = torch.empty((24, 16, h_kv, d), device=dev, dtype=torch.float16)
        bt = torch.empty((B, 8), device=dev, dtype=torch.int32)
        seqlens = torch.empty((B,), device=dev, dtype=torch.int32)
        qd = torch.empty((B, h_q, d), device=dev, dtype=torch.float16)
        nk = torch.empty((B, h_kv, d), device=dev, dtype=torch.float16)
        nv = torch.empty((B, h_kv, d), device=dev, dtype=torch.float16)

        ns.prefill_to_paged_kv(k, v, cu_k, k_cache, v_cache, bt, seqlens, 64)
        ns.paged_kv_append(nk, nv, k_cache, v_cache, bt, seqlens)
        od = ns.fused_mha_varlen_decode(qd, k_cache, v_cache, bt, seqlens, 64, cos, sin, offsets, None)
        ok_dec = od.shape == (B, h_q, d)
        print(f"[{'PASS' if ok_dec else 'FAIL'}] faketensor.paged_kv_decode           od={tuple(od.shape)}")
        all_passed &= ok_dec

        x = torch.empty((total_l, embed_dim), device=dev, dtype=torch.float16)
        ctx = torch.empty((total_lk, ctx_dim), device=dev, dtype=torch.float16)
        w_qkv = torch.empty(((h_q + 2 * h_kv) * d, embed_dim), device=dev, dtype=torch.float16)
        w_out = torch.empty((embed_dim, embed_dim), device=dev, dtype=torch.float16)
        w_q = torch.empty((embed_dim, embed_dim), device=dev, dtype=torch.float16)
        w_kv = torch.empty((2 * h_kv * d, ctx_dim), device=dev, dtype=torch.float16)
        xd = torch.empty((B, embed_dim), device=dev, dtype=torch.float16)

        g_out, g_qkv, g_ao, g_lse = ns.fused_gqa_varlen_fwd(x, w_qkv, w_out, cu_q, 48, h_q, h_kv, cos, sin, True, offsets, True)
        dx, dw_qkv, dw_out = ns.fused_gqa_varlen_bwd(g_out, x, g_qkv, g_ao, g_lse, w_qkv, w_out, cu_q, offsets, 48, h_q, h_kv, cos, sin, True, True, True, True)
        gp_out = ns.fused_gqa_varlen_prefill(x, w_qkv, w_out, cu_q, 48, h_q, h_kv, k_cache, v_cache, bt, seqlens, cos, sin, True, offsets)
        gd_out = ns.fused_gqa_varlen_decode(xd, w_qkv, w_out, k_cache, v_cache, bt, seqlens, 65, h_q, h_kv, cos, sin, offsets, None)
        c_out, c_q, c_kv, c_ao, c_lse = ns.fused_cross_gqa_varlen_fwd(x, ctx, w_q, w_kv, w_out, cu_q, cu_k, 48, h_q, h_kv, cos, sin, offsets, True)
        cdx, cdctx, cdwq, cdwkv, cdwout = ns.fused_cross_gqa_varlen_bwd(c_out, x, ctx, c_q, c_kv, c_ao, c_lse, w_q, w_kv, w_out, cu_q, cu_k, offsets, 64, h_q, h_kv, cos, sin, True, True, True, True, True)

        ok_gqa = (
            (g_out.shape == x.shape)
            and (dx.shape == x.shape)
            and (dw_qkv.shape == w_qkv.shape)
            and (dw_out.shape == w_out.shape)
            and (gp_out.shape == x.shape)
            and (gd_out.shape == xd.shape)
            and (c_out.shape == x.shape)
            and (cdx.shape == x.shape)
            and (cdctx.shape == ctx.shape)
            and (cdwq.shape == w_q.shape)
            and (cdwkv.shape == w_kv.shape)
            and (cdwout.shape == w_out.shape)
        )
        print(f"[{'PASS' if ok_gqa else 'FAIL'}] faketensor.fused_gqa_self_cross      all_shapes_ok={ok_gqa}")
        all_passed &= ok_gqa

    return all_passed


def run_all_tests(quick: bool = False, seed: int = 42) -> bool:
    """Run the full v2 correctness verification suite and return overall pass status."""
    seed_everything(seed)
    dev = _require_gpu_and_ext()

    ok_1 = test_varlen_fwd_bwd_kernels(quick=quick, device=dev)
    ok_2 = test_varlen_decode_kernels(quick=quick, device=dev)
    ok_3 = test_end_to_end_modules(device=dev)
    ok_4 = test_faketensor_registrations(device=dev)

    overall = ok_1 and ok_2 and ok_3 and ok_4
    print("\n" + "=" * 80)
    print(f"FINAL CORRECTNESS RESULT: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    print("=" * 80)
    if not overall:
        raise AssertionError("Correctness suite failed! See FAIL entries above.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 correctness checks for varlen_transformer.")
    parser.add_argument("--quick", action="store_true", help="Run compact fast subset of test cases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    run_all_tests(quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
