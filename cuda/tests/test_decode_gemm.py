"""DecodeLinear against an FP32 reference, and its batch invariance."""

import pytest
import torch

chandra_cuda = pytest.importorskip("chandra_cuda")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

# (N, K) of Chandra 2's decode GEMMs: merged GDN qkvz, merged GDN ba, GDN out_proj,
# merged full-attention qkv (with the output gate), o_proj, merged gate_up, down.
SHAPES = [(12288, 2560), (64, 2560), (2560, 4096), (10240, 2560), (18432, 2560), (2560, 9216), (1024, 2560)]
ROWS = [1, 2, 7, 8, 9, 16, 17, 31, 32, 33, 48, 63, 64]


@pytest.mark.parametrize("n,k", SHAPES)
def test_matches_fp32_reference(n, k):
    torch.manual_seed(n + k)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinear(w)
    for m in ROWS:
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ref = x.float() @ w.float().T
        got = lin(x).float()
        err = (got - ref).abs().max().item()
        scale = ref.abs().max().item()
        # BF16 output rounding (2^-8 relative) dominates; FP32 accumulation order differs.
        assert err <= scale * 2**-7 + 1e-3, (m, err, scale)


@pytest.mark.parametrize("n,k", [(2560, 9216), (64, 2560), (12288, 2560)])
def test_same_result_at_every_batch_size(n, k):
    torch.manual_seed(1)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinear(w)
    x = torch.randn(64, k, device="cuda").to(torch.bfloat16)
    full = lin(x)
    for m in ROWS:
        assert torch.equal(lin(x[:m]), full[:m]), m


def test_repeated_calls_reset_counters():
    w = (torch.randn(64, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinear(w)
    x = torch.randn(8, 2560, device="cuda").to(torch.bfloat16)
    first = lin(x)
    for _ in range(5):
        assert torch.equal(lin(x), first)
    assert int(lin.counters.abs().sum()) == 0


@pytest.mark.parametrize("n,k", [(248320, 2560), (4096, 2560)])
def test_lm_head_argmax_matches_its_logits(n, k):
    torch.manual_seed(5)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    head = chandra_cuda.LMHead(w)
    lin = chandra_cuda.DecodeLinear(w)
    agree = total = 0
    for m in ROWS:
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ids = head(x)
        # Same accumulation as DecodeLinear, so the same BF16 logits: an exact match.
        assert torch.equal(ids, lin(x).float().argmax(dim=-1)), m
        agree += int((ids == torch.nn.functional.linear(x, w).float().argmax(dim=-1)).sum())
        total += m
    assert agree >= 0.97 * total  # cuBLAS rounds some logits differently (near-ties)


def test_lm_head_ties_go_to_the_lowest_index():
    torch.manual_seed(6)
    w = (torch.randn(1024, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(8, 2560, device="cuda").to(torch.bfloat16)
    best = torch.nn.functional.linear(x, w).float().argmax(dim=-1)
    w2 = w.clone()
    w2[1000] = w[int(best[0])]  # a later copy of token 0's best row
    w2[3] = w[int(best[0])]  # and an earlier one
    ids = chandra_cuda.LMHead(w2)(x)
    assert int(ids[0]) == min(3, int(best[0]))


@pytest.mark.parametrize("n,k", [s for s in SHAPES if s[0] % 32 == 0])
def test_fp8_matches_its_dequantized_weight(n, k):
    torch.manual_seed(n * 7 + k)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinearFP8(w)
    wd = lin.dequantized()
    # The quantization itself: within e4m3's relative step (2^-3) of each weight.
    assert ((wd - w.float()).abs() <= w.float().abs() * 2**-3 + lin.scale[:, None] * 2**-9).all()
    for m in ROWS:
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ref = x.float() @ wd.T
        got = lin(x).float()
        assert (got - ref).abs().max().item() <= ref.abs().max().item() * 2**-7 + 1e-3, m


def test_fp8_same_result_at_every_batch_size():
    torch.manual_seed(2)
    w = (torch.randn(2560, 9216, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinearFP8(w)
    x = torch.randn(64, 9216, device="cuda").to(torch.bfloat16)
    full = lin(x)
    for m in ROWS:
        assert torch.equal(lin(x[:m]), full[:m]), m


def test_fp8_lm_head_matches_fp8_logits():
    torch.manual_seed(8)
    n, k = 248320, 2560
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    head = chandra_cuda.LMHeadFP8(w)
    lin = chandra_cuda.DecodeLinearFP8(w)  # one split at this N: the same accumulation
    for m in (1, 9, 33, 64):
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        assert torch.equal(head(x), lin(x).float().argmax(dim=-1)), m


def test_fp8_lm_head_subset_returns_vocabulary_ids():
    torch.manual_seed(9)
    w = (torch.randn(8192, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    rows = torch.randperm(8192, device="cuda")[:1000].sort().values  # not a multiple of 128
    head = chandra_cuda.LMHeadFP8(w, rows=rows)
    x = torch.randn(16, 2560, device="cuda").to(torch.bfloat16)
    got = head(x)
    assert bool(torch.isin(got, rows).all())
    ref = torch.nn.functional.linear(x, w[rows]).float()
    best = ref.max(dim=-1).values
    picked = ref.gather(1, torch.searchsorted(rows, got)[:, None])[:, 0]
    assert ((best - picked) <= 0.05 * best.abs() + 0.05).all()  # FP8 weights: near the BF16 best


@pytest.mark.parametrize("n,k", [s for s in SHAPES if s[0] % 64 == 0])
def test_fp4_matches_its_dequantized_weight(n, k):
    torch.manual_seed(n * 3 + k)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinearFP4(w, keep_codes=True)
    wd = lin.dequantized()
    rel = ((wd - w.float()).norm() / w.float().norm()).item()
    assert rel < 0.15, rel  # e2m1 with a shared power-of-two scale: ~9% RMS error on Gaussian weights
    for m in ROWS:
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ref = x.float() @ wd.T
        got = lin(x).float()
        assert (got - ref).abs().max().item() <= ref.abs().max().item() * 2**-7 + 1e-3, m


def test_fp4_same_result_at_every_batch_size():
    torch.manual_seed(4)
    w = (torch.randn(2560, 9216, device="cuda") * 0.02).to(torch.bfloat16)
    lin = chandra_cuda.DecodeLinearFP4(w)
    x = torch.randn(64, 9216, device="cuda").to(torch.bfloat16)
    full = lin(x)
    for m in (1, 5, 17, 40):
        assert torch.equal(lin(x[:m]), full[:m]), m


def test_mxfp4_quantize_matches_vllm():
    try:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize as vq
    except Exception:  # noqa: BLE001
        pytest.skip("vLLM not importable")
    torch.manual_seed(12)
    w = (torch.randn(512, 2560, device="cuda") * 0.03).to(torch.bfloat16)
    w[0, :32] = 0  # an all-zero block
    w[1, 5] = 6.5  # a large outlier
    codes, exps = chandra_cuda.mxfp4_quantize(w)
    vp, vs = vq(w.contiguous())
    vcodes = torch.stack([vp & 0xF, vp >> 4], dim=-1).reshape(512, 2560)
    assert torch.equal(exps, vs.view(torch.uint8).reshape(512, -1))
    assert torch.equal(codes, vcodes)


def test_exact_lm_head_equals_bf16_head():
    torch.manual_seed(13)
    n, k = 65536, 2560
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    head = chandra_cuda.LMHead(w)
    ex = chandra_cuda.LMHeadExact(w, head)
    cands = []
    for m in (1, 7, 16, 33, 64):
        for scale in (1.0, 4.0):  # flat logits (many near-ties) and peaked ones
            x = (torch.randn(m, k, device="cuda") * scale).to(torch.bfloat16)
            assert torch.equal(ex(x), head(x)), (m, scale)
            cands.append(ex.candidate_tiles(x).float().mean().item())
    # Rows that tie in BF16 with a later copy: the re-check keeps the lowest index.
    x = torch.randn(4, k, device="cuda").to(torch.bfloat16)
    best = head(x)
    w2 = w.clone()
    w2[n - 1] = w[int(best[0])]
    w2[100] = w[int(best[0])]
    h2 = chandra_cuda.LMHead(w2)
    assert torch.equal(chandra_cuda.LMHeadExact(w2, h2)(x), h2(x))
    print("mean candidate tiles per token:", [round(c, 1) for c in cands])


@pytest.mark.parametrize("fmt", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("f,k", [(9216, 2560), (1024, 2560)])  # the MLP; a narrow one, split along K
def test_silu_mul_epilogue_equals_separate_ops(fmt, f, k):
    torch.manual_seed(f + k)
    w = (torch.randn(2 * f, k, device="cuda") * 0.02).to(torch.bfloat16)
    cls = {"bf16": chandra_cuda.DecodeLinear, "fp8": chandra_cuda.DecodeLinearFP8,
           "fp4": chandra_cuda.DecodeLinearFP4}[fmt]
    plain, fused = cls(w), cls(w, epilogue="silu_mul")
    for m in (1, 9, 33, 64):
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        assert torch.equal(fused(x), chandra_cuda.silu_mul(plain(x))), (fmt, m)

