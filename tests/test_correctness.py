"""Numerical correctness tests comparing CUDA kernels vs PyTorch varlen reference."""

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tests.utils import (
    build_padding_mask,
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
    """Test raw `fused_mha_varlen_fwd`, `fused_mha_varlen_bwd`, and `varlen_flash_attention`."""
    print("\n" + "=" * 80)
    print("1. Testing Varlen Forward & Backward CUDA Kernels vs Pure-PyTorch Reference")
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

        q = torch.randn((h_q, total_lq, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
        k = torch.randn((h_kv, total_lk, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
        v = torch.randn((h_kv, total_lk, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
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

    # Edge cases: empty sequences inside a batch and an empty batch
    print("\n-- Case [edge_zero_len_and_empty_batch] --")
    cu_q_edge, max_q_edge = _make_cu_seqlens([0, 17, 0, 65], dev)
    cu_k_edge, max_k_edge = _make_cu_seqlens([0, 17, 0, 65], dev)
    total_edge = int(cu_q_edge[-1].item())
    q_e = torch.randn((8, total_edge, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
    k_e = torch.randn((2, total_edge, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
    v_e = torch.randn((2, total_edge, head_dim), device=dev, dtype=torch.float16, requires_grad=True)
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
    q_empty = torch.empty((8, 0, head_dim), device=dev, dtype=torch.float16)
    k_empty = torch.empty((2, 0, head_dim), device=dev, dtype=torch.float16)
    v_empty = torch.empty((2, 0, head_dim), device=dev, dtype=torch.float16)
    o_empty, lse_empty = fused_mha_varlen_fwd(q_empty, k_empty, v_empty, cu_empty, cu_empty, 0)
    empty_ok = (o_empty.shape == (8, 0, head_dim)) and (lse_empty.shape == (8, 0))
    print(f"[{'PASS' if empty_ok else 'FAIL'}] edge_empty_batch.shapes             o={tuple(o_empty.shape)} lse={tuple(lse_empty.shape)}")
    all_passed &= empty_ok

    return all_passed


def test_varlen_decode_kernels(quick: bool = False, device: Optional[torch.device] = None) -> bool:
    """Test Split-K `fused_mha_varlen_decode` & `varlen_flash_decode` vs `varlen_decode_ref`."""
    print("\n" + "=" * 80)
    print("2. Testing Varlen Split-K Flash Decoding (Method 2 Raw K-Cache) vs Reference")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    head_dim = 64
    all_passed = True

    if quick:
        cases = [
            ("decode_short_rope", [1, 16, 17, 65], 8, 2, True, [0, 3, 9, 4]),
            ("decode_multisplit_long", [19, 257, 1025], 16, 4, True, None),
            ("decode_no_rope", [31, 513], 8, 8, False, None),
        ]
    else:
        cases = [
            ("decode_short_no_rope", [1, 7, 15, 16, 17], 8, 8, False, None),
            ("decode_gqa_rope_offsets", [1, 15, 16, 17, 64, 127, 255], 8, 2, True, [2, 0, 11, 5, 19, 3, 8]),
            ("decode_multisplit_long_rope", [1, 65, 257, 513, 1025], 16, 4, True, [0, 12, 4, 33, 7]),
            ("decode_gqa_8_1_long", [129, 769, 1537], 8, 1, True, None),
            ("decode_edge_zero_seq_in_batch", [0, 33, 0, 260], 8, 2, True, [0, 5, 0, 10]),
        ]

    for name, k_lens, h_q, h_kv, use_rope, offset_list in cases:
        cu_k, max_k = _make_cu_seqlens(k_lens, dev)
        B = len(k_lens)
        total_lk = int(cu_k[-1].item())

        q = torch.randn((h_q, B, head_dim), device=dev, dtype=torch.float16)
        k_cache = torch.randn((h_kv, total_lk, head_dim), device=dev, dtype=torch.float16)
        v_cache = torch.randn((h_kv, total_lk, head_dim), device=dev, dtype=torch.float16)

        offsets = torch.tensor(offset_list, dtype=torch.int32, device=dev) if offset_list is not None else None
        cos = sin = None
        if use_rope:
            max_off = max(offset_list) if offset_list is not None else 0
            rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_k + max_off + 64, device=dev)
            cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16)

        with torch.no_grad():
            o_raw = fused_mha_varlen_decode(
                q, k_cache, v_cache, cu_k, max_k, cos=cos, sin=sin, offsets=offsets
            )
            o_hi = varlen_flash_decode(
                q, k_cache, v_cache, cu_k, max_k, cos=cos, sin=sin, offsets=offsets
            )
            o_direct = varlen_flash_decode(
                q, k_cache, v_cache, cu_k, max_k, cos=cos, sin=sin, offsets=offsets, use_splitk=False
            )
            o_splitk = varlen_flash_decode(
                q, k_cache, v_cache, cu_k, max_k, cos=cos, sin=sin, offsets=offsets, use_splitk=True
            )
            o_ref = varlen_decode_ref(
                q, k_cache, v_cache, cu_k, cos=cos, sin=sin, offsets=offsets
            )

        no_nan = bool(
            torch.isfinite(o_raw).all().item()
            and torch.isfinite(o_direct).all().item()
            and torch.isfinite(o_splitk).all().item()
        )
        print(f"\n-- Case [{name}] k_lens={k_lens}, H=({h_q},{h_kv}), rope={use_rope}, finite={no_nan} --")
        all_passed &= no_nan
        all_passed &= check_allclose(f"{name}.decode_auto", o_raw, o_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.decode_hi", o_hi, o_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.decode_direct", o_direct, o_ref, atol=1e-2, rtol=1e-2)[0]
        all_passed &= check_allclose(f"{name}.decode_splitk", o_splitk, o_ref, atol=1e-2, rtol=1e-2)[0]

    return all_passed


def test_end_to_end_modules(device: Optional[torch.device] = None) -> bool:
    """Test `VarlenFlashGQA` vs `VarlenFlashGQABaselineCorrectness` & `BatchedFlashGQABaselineSpeed`."""
    print("\n" + "=" * 80)
    print("3. Testing End-to-End GQA Modules (Training & Prefill + Autoregressive Decode)")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    all_passed = True

    embed_dim, h_q, h_kv, head_dim = 512, 8, 2, 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=1024, device=dev)

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

    # --- 3A. Training Forward + Backward (with variable lengths & causal mask) ---
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

    all_passed &= check_allclose("e2e_train.out", out_cuda, out_vref, atol=2e-2, rtol=2e-2)[0]
    all_passed &= check_allclose("e2e_train.dx", x_cuda.grad, x_vref.grad, atol=3e-2, rtol=3e-2)[0]
    all_passed &= check_allclose(
        "e2e_train.dqkv_weight", mod_cuda.qkv_proj.weight.grad, mod_vref.qkv_proj.weight.grad, atol=4e-2, rtol=4e-2
    )[0]
    all_passed &= check_allclose(
        "e2e_train.dout_weight", mod_cuda.out_proj.weight.grad, mod_vref.out_proj.weight.grad, atol=4e-2, rtol=4e-2
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

    # --- 3B. Prefill + 4-step Autoregressive Varlen Decode (VarlenKVCache Method 2) ---
    with torch.no_grad():
        x_prefill = torch.randn((total_tokens, embed_dim), device=dev, dtype=torch.float16)
        pre_cuda, cache_cuda = mod_cuda(
            x_prefill, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, is_causal=True, use_cache=True
        )
        pre_vref, cache_vref = mod_vref(
            x_prefill, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, is_causal=True, use_cache=True
        )
        all_passed &= check_allclose("e2e_prefill_with_cache", pre_cuda, pre_vref, atol=2e-2, rtol=2e-2)[0]

        for step in range(4):
            x_step = torch.randn((len(lengths), embed_dim), device=dev, dtype=torch.float16)
            dec_cuda = mod_cuda(x_step, kv_cache=cache_cuda, decode=True)
            dec_vref = mod_vref(x_step, kv_cache=cache_vref, decode=True)
            all_passed &= check_allclose(f"e2e_decode_step_{step}", dec_cuda, dec_vref, atol=2e-2, rtol=2e-2)[0]

    return all_passed


def test_torch_compile_inductor(device: Optional[torch.device] = None) -> bool:
    """Verify `torch.compile(..., backend='inductor')` compatibility for Training & Decode."""
    print("\n" + "=" * 80)
    print("4. Testing torch.compile (TorchInductor + FakeTensor + AOTAutograd)")
    print("=" * 80)
    dev = device or _require_gpu_and_ext()
    all_passed = True

    embed_dim, h_q, h_kv, head_dim = 512, 8, 2, 64
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=512, device=dev)
    eager_model = VarlenFlashGQA(embed_dim, h_q, h_kv, rope=rope, bias=False).to(device=dev, dtype=torch.float16)

    torch._dynamo.reset()
    compiled_model = torch.compile(eager_model, backend="inductor")

    # 4A. Compile Training Forward + Backward
    lengths = [24, 65, 97]
    cu_seqlens, max_len = _make_cu_seqlens(lengths, dev)
    total_tokens = int(cu_seqlens[-1].item())

    x_eager = torch.randn((total_tokens, embed_dim), device=dev, dtype=torch.float16, requires_grad=True)
    x_comp = x_eager.detach().clone().requires_grad_(True)
    go = torch.randn_like(x_eager) * 0.1

    out_eager = eager_model(
        x_eager, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len, is_causal=True
    )
    out_eager.backward(go)
    grads_eager = {n: p.grad.detach().clone() for n, p in eager_model.named_parameters()}
    eager_model.zero_grad(set_to_none=True)

    out_comp = compiled_model(
        x_comp, cu_seqlens_q=cu_seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len, is_causal=True
    )
    out_comp.backward(go)

    all_passed &= check_allclose("compile_train.out", out_comp, out_eager, atol=5e-3, rtol=5e-3)[0]
    all_passed &= check_allclose("compile_train.dx", x_comp.grad, x_eager.grad, atol=5e-3, rtol=5e-3)[0]
    for pname, param in eager_model.named_parameters():
        all_passed &= check_allclose(f"compile_train.{pname}", param.grad, grads_eager[pname], atol=5e-3, rtol=5e-3)[0]

    # 4B. Compile Decode (varlen_flash_decode op inside compiled function)
    cos, sin = rope.get_cos_sin(device=dev, dtype=torch.float16)
    B = len(lengths)
    k_cache = torch.randn((h_kv, total_tokens, head_dim), device=dev, dtype=torch.float16)
    v_cache = torch.randn((h_kv, total_tokens, head_dim), device=dev, dtype=torch.float16)
    q_dec = torch.randn((h_q, B, head_dim), device=dev, dtype=torch.float16)

    @torch.compile(backend="inductor")
    def _compiled_decode_step(q_in, k_in, v_in, cu_in, c_in, s_in):
        return varlen_flash_decode(q_in, k_in, v_in, cu_in, max_seqlen_k=max_len, cos=c_in, sin=s_in)

    with torch.no_grad():
        dec_eager = varlen_flash_decode(q_dec, k_cache, v_cache, cu_seqlens, max_seqlen_k=max_len, cos=cos, sin=sin)
        dec_comp = _compiled_decode_step(q_dec, k_cache, v_cache, cu_seqlens, cos, sin)
    all_passed &= check_allclose("compile_decode.out", dec_comp, dec_eager, atol=5e-3, rtol=5e-3)[0]

    return all_passed


def run_all_tests(quick: bool = False, seed: int = 42) -> bool:
    """Run the full correctness verification suite and return overall pass status."""
    seed_everything(seed)
    dev = _require_gpu_and_ext()

    ok_1 = test_varlen_fwd_bwd_kernels(quick=quick, device=dev)
    ok_2 = test_varlen_decode_kernels(quick=quick, device=dev)
    ok_3 = test_end_to_end_modules(device=dev)
    ok_4 = test_torch_compile_inductor(device=dev)

    overall = ok_1 and ok_2 and ok_3 and ok_4
    print("\n" + "=" * 80)
    print(f"FINAL CORRECTNESS RESULT: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    print("=" * 80)
    if not overall:
        raise AssertionError("Correctness suite failed! See FAIL entries above.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run correctness checks for varlen_transformer.")
    parser.add_argument("--quick", action="store_true", help="Run compact fast subset of test cases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    run_all_tests(quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
