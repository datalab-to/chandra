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
