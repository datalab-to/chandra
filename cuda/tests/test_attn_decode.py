"""attn_decode against an FP32 PyTorch reference: decode (n_q = 1) and causal verify (n_q 2-4)."""

import math

import pytest
import torch

chandra_cuda = pytest.importorskip("chandra_cuda")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

H, KVH, D, PAGE = 16, 4, 256, 64


def make_case(ctxs, n_q, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    b = len(ctxs)
    pages_per = [math.ceil(c / PAGE) for c in ctxs]
    n_pages = sum(pages_per) + 3
    # Pages in a shuffled order, plus garbage (NaN) in unused slots to check zero-fill.
    k_cache = torch.randn(n_pages, KVH, PAGE, D, device="cuda", generator=g).to(torch.bfloat16)
    v_cache = torch.randn(n_pages, KVH, PAGE, D, device="cuda", generator=g).to(torch.bfloat16)
    perm = torch.randperm(n_pages, device="cuda", generator=g)
    bt = torch.zeros(b, max(pages_per), dtype=torch.int32, device="cuda")
    i = 0
    for s, p in enumerate(pages_per):
        bt[s, :p] = perm[i : i + p].to(torch.int32)
        i += p
        last = int(bt[s, p - 1])
        used = ctxs[s] - (p - 1) * PAGE
        k_cache[last, :, used:] = float("nan")
        v_cache[last, :, used:] = float("nan")
    qg = torch.randn(b * n_q, H, 2 * D, device="cuda", generator=g).to(torch.bfloat16)
    ctx = torch.tensor(ctxs, dtype=torch.int32, device="cuda")
    return qg[..., :D], qg[..., D:], k_cache, v_cache, bt, ctx


def reference(q, gate, k_cache, v_cache, bt, ctxs, n_q):
    outs = []
    for s, c in enumerate(ctxs):
        p = math.ceil(c / PAGE)
        k = k_cache[bt[s, :p].long()].permute(1, 0, 2, 3).reshape(KVH, -1, D)[:, :c].float()
        v = v_cache[bt[s, :p].long()].permute(1, 0, 2, 3).reshape(KVH, -1, D)[:, :c].float()
        for t in range(n_q):
            seen = c - n_q + t + 1
            qt = q[s * n_q + t].float()  # [16, 256]
            kk = k.repeat_interleave(4, dim=0)[:, :seen]  # [16, seen, 256]
            vv = v.repeat_interleave(4, dim=0)[:, :seen]
            att = torch.softmax(torch.einsum("hd,hkd->hk", qt, kk) / 16.0, dim=-1)
            o = torch.einsum("hk,hkd->hd", att, vv).to(torch.bfloat16).float()
            outs.append((o * torch.sigmoid(gate[s * n_q + t].float())).to(torch.bfloat16))
    return torch.stack(outs)


@pytest.mark.parametrize("n_q", [1, 2, 3, 4])
@pytest.mark.parametrize("ctxs", [[1], [37], [64, 65, 130], [6700, 7123, 50, 2048], [9000] * 8])
def test_matches_reference(ctxs, n_q):
    if min(ctxs) < n_q:
        pytest.skip("context shorter than the verify window")
    q, gate, kc, vc, bt, ctx = make_case(ctxs, n_q, seed=len(ctxs) * 10 + n_q)
    got = chandra_cuda.attn_decode(q, gate, kc, vc, bt, ctx, n_q=n_q).float()
    ref = reference(q, gate, kc, vc, bt, ctxs, n_q).float()
    assert torch.isfinite(got).all()
    err = (got - ref).abs().max().item()
    assert err < 2e-2, err  # BF16 P and output rounding


def test_verify_rows_match_single_token_steps():
    # Token t of a verify step must equal a decode step on the prefix of length ctx - n_q + t + 1,
    # up to rounding (the two are different reduction orders).
    ctxs = [3000, 777]
    n_q = 4
    q, gate, kc, vc, bt, ctx = make_case(ctxs, n_q, seed=7)
    ver = chandra_cuda.attn_decode(q, gate, kc, vc, bt, ctx, n_q=n_q).float()
    for t in range(n_q):
        sub = ctx - n_q + t + 1
        one = chandra_cuda.attn_decode(
            q.reshape(2, n_q, H, D)[:, t], gate.reshape(2, n_q, H, D)[:, t], kc, vc, bt, sub.to(torch.int32), n_q=1
        ).float()
        assert (one - ver.reshape(2, n_q, H, D)[:, t]).abs().max().item() < 2e-2


def quantize_kv(cache):
    """Per (page, head, token) symmetric INT8, as the KV write does: x = int8 * scale."""
    scale = cache.float().abs().amax(dim=-1) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(cache.float() / scale[..., None]).clamp(-127, 127).to(torch.int8)
    return q, scale


@pytest.mark.parametrize("n_q", [1, 4])
@pytest.mark.parametrize("ctxs", [[37], [6700, 7123, 50, 2048], [9000] * 8])
def test_int8_kv_matches_reference_on_dequantized_cache(ctxs, n_q):
    if min(ctxs) < n_q:
        pytest.skip("context shorter than the verify window")
    q, gate, kc, vc, bt, ctx = make_case(ctxs, n_q, seed=99 + n_q)
    kc = torch.nan_to_num(kc, nan=3.0)  # INT8 cannot hold NaN; poison the scales instead
    vc = torch.nan_to_num(vc, nan=3.0)
    k8, ks = quantize_kv(kc)
    v8, vs = quantize_kv(vc)
    for s, c in enumerate(ctxs):  # NaN scales past each context must be ignored
        p = (c - 1) // PAGE
        last = int(bt[s, p])
        ks[last, :, c - p * PAGE:] = float("nan")
        vs[last, :, c - p * PAGE:] = float("nan")
    got = chandra_cuda.attn_decode(q, gate, k8, v8, bt, ctx, n_q=n_q, k_scale=ks, v_scale=vs).float()
    kd = (k8.float() * torch.nan_to_num(ks, nan=0.0)[..., None]).to(torch.bfloat16)
    vd = (v8.float() * torch.nan_to_num(vs, nan=0.0)[..., None]).to(torch.bfloat16)
    ref = reference(q, gate, kd, vd, bt, ctxs, n_q).float()
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max().item() < 3e-2  # FP16 tiles and BF16 rounding of the reference


def test_kv_write_int8():
    g = torch.Generator(device="cuda").manual_seed(8)
    m = 5
    qkv = torch.randn(m, 10240, device="cuda", generator=g).to(torch.bfloat16)
    qw = (0.2 * torch.randn(256, device="cuda", generator=g)).to(torch.bfloat16)
    kw = (0.2 * torch.randn(256, device="cuda", generator=g)).to(torch.bfloat16)
    pos = torch.randint(0, 9000, (m, 3), device="cuda", dtype=torch.int32, generator=g)
    page = torch.tensor([0, 1, 1, 2, 0], device="cuda", dtype=torch.int32)
    slot = torch.tensor([3, 0, 63, 17, 4], device="cuda", dtype=torch.int32)
    kb = torch.zeros(3, 4, 64, 256, device="cuda", dtype=torch.bfloat16)
    vb = torch.zeros_like(kb)
    chandra_cuda.qk_norm_rope_kv(qkv, qw, kw, pos, kb, vb, page, slot)
    k8 = torch.zeros(3, 4, 64, 256, device="cuda", dtype=torch.int8)
    v8 = torch.zeros_like(k8)
    ks = torch.zeros(3, 4, 64, device="cuda")
    vs = torch.zeros_like(ks)
    chandra_cuda.qk_norm_rope_kv(qkv, qw, kw, pos, k8, v8, page, slot, k_scale=ks, v_scale=vs)
    for i in range(m):
        p, s = int(page[i]), int(slot[i])
        for c8, cs, cb in ((k8, ks, kb), (v8, vs, vb)):
            deq = c8[p, :, s].float() * cs[p, :, s, None]
            ref = cb[p, :, s].float()
            assert ((deq - ref).abs() <= cs[p, :, s, None] * 0.51 + ref.abs() * 2**-7).all()
