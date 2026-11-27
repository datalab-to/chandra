"""rmsnorm_add, silu_mul and qk_norm_rope_kv against references written as transformers'
Qwen3.5 code computes them (FP32 math, BF16 at the model's rounding points)."""

import pytest
import torch
import torch.nn.functional as F

chandra_cuda = pytest.importorskip("chandra_cuda")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)


def ulp_close(a, b, ulps=2):
    """Elementwise within `ulps` BF16 steps of b (relative), for BF16 outputs."""
    a, b = a.float(), b.float()
    step = b.abs().clamp_min(1e-3) * 2**-7
    return bool(((a - b).abs() <= ulps * step).all())


@pytest.mark.parametrize("m", [1, 7, 64])
def test_rmsnorm_add(m):
    g = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, 2560, device="cuda", generator=g).to(torch.bfloat16)
    res = torch.randn(m, 2560, device="cuda", generator=g).to(torch.bfloat16)
    w = (0.1 * torch.randn(2560, device="cuda", generator=g)).to(torch.bfloat16)
    out, r = chandra_cuda.rmsnorm_add(res, w, x=x)
    ref_r = res + x  # BF16 add, as the model's residual stream
    assert torch.equal(r, ref_r)
    rf = ref_r.float()
    ref = (rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w.float())).to(torch.bfloat16)
    assert ulp_close(out, ref)
    out2, r2 = chandra_cuda.rmsnorm_add(res, w)  # no add
    rf = res.float()
    assert r2 is res
    assert ulp_close(out2, (rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w.float())).to(torch.bfloat16))


@pytest.mark.parametrize("m", [1, 33])
def test_silu_mul(m):
    gu = torch.randn(m, 2 * 9216, device="cuda").to(torch.bfloat16)
    out = chandra_cuda.silu_mul(gu)
    ref = (F.silu(gu[:, :9216].float()) * gu[:, 9216:].float()).to(torch.bfloat16)
    assert ulp_close(out, ref, ulps=1)


def rope_ref(x, pos, theta=1e7):
    """x: [M, H, 256] BF16 after the norm; pos: [M, 3]. Partial interleaved mRoPE as transformers."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float, device=x.device) / 64))  # [32]
    axis = torch.arange(32, device=x.device) % 3
    p = pos.float()[:, axis]  # [M, 32]
    freqs = p * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)  # [M, 64]
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]
    xf = x.float()
    rot, rest = xf[..., :64], xf[..., 64:]
    half = torch.cat([-rot[..., 32:], rot[..., :32]], dim=-1)
    return torch.cat([rot * cos + half * sin, rest], dim=-1).to(torch.bfloat16)


def norm_ref(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w.float())).to(torch.bfloat16)


@pytest.mark.parametrize("m", [1, 6])
def test_qk_norm_rope_kv(m):
    g = torch.Generator(device="cuda").manual_seed(40 + m)
    qkv = torch.randn(m, 10240, device="cuda", generator=g).to(torch.bfloat16)
    qw = (0.2 * torch.randn(256, device="cuda", generator=g)).to(torch.bfloat16)
    kw = (0.2 * torch.randn(256, device="cuda", generator=g)).to(torch.bfloat16)
    pos = torch.randint(0, 9000, (m, 3), device="cuda", dtype=torch.int32, generator=g)
    pos[0] = torch.tensor([7123, 7123, 7123], device="cuda")  # a text token: all axes equal
    kc = torch.zeros(5, 4, 64, 256, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    page = torch.randint(0, 5, (m,), device="cuda", dtype=torch.int32, generator=g)
    slot = torch.randperm(64, device="cuda", generator=g)[:m].to(torch.int32)
    q = chandra_cuda.qk_norm_rope_kv(qkv, qw, kw, pos, kc, vc, page, slot)
    heads = qkv[:, :8192].view(m, 16, 512)
    q_ref = rope_ref(norm_ref(heads[..., :256], qw), pos)
    k_ref = rope_ref(norm_ref(qkv[:, 8192:9216].view(m, 4, 256), kw), pos)
    v_ref = qkv[:, 9216:].view(m, 4, 256)
    assert ulp_close(q, q_ref, ulps=3)
    for i in range(m):
        assert ulp_close(kc[page[i], :, slot[i]], k_ref[i], ulps=3)
        assert torch.equal(vc[page[i], :, slot[i]], v_ref[i])


def test_conv_silu_prefill_matches_reference():
    torch.manual_seed(41)
    n = 97
    y = torch.randn(n, 12416, device="cuda").to(torch.bfloat16)
    x = y[:, :8192]
    w = (0.4 * torch.randn(8192, 4, device="cuda")).to(torch.bfloat16)
    got = chandra_cuda.conv_silu_prefill(x, w).float()
    conv = torch.nn.functional.conv1d(x.T.float()[None], w.float()[:, None, :], padding=3, groups=8192)[0, :, :n]
    ref = torch.nn.functional.silu(conv).T
    assert (got - ref).abs().max().item() <= 2e-2 * ref.abs().max().item()


def test_gated_rmsnorm_matches_reference():
    torch.manual_seed(42)
    n = 53
    o = torch.randn(n, 32, 128, device="cuda").to(torch.bfloat16)
    zz = torch.randn(n, 12416, device="cuda").to(torch.bfloat16)
    z = zz[:, 8192:12288]
    w = (1 + 0.1 * torch.randn(128, device="cuda")).to(torch.bfloat16)
    got = chandra_cuda.gated_rmsnorm(o, z, w).float()
    of = o.float()
    on = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (on * w.float() * torch.nn.functional.silu(z.float().reshape(n, 32, 128))).reshape(n, 4096)
    assert (got - ref).abs().max().item() <= 2e-2 * ref.abs().max().item()

