"""vit_attn against PyTorch scaled_dot_product_attention (FP32) on packed images."""

import pytest
import torch
import torch.nn.functional as F

chandra_cuda = pytest.importorskip("chandra_cuda")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)


@pytest.mark.parametrize("version", [1, 2, 4])
@pytest.mark.parametrize("lens", [[1], [64], [300], [1000, 77, 4096], [24552]])
def test_matches_sdpa(lens, version):
    g = torch.Generator(device="cuda").manual_seed(sum(lens))
    n = sum(lens)
    qkv = torch.randn(n, 3, 16, 64, device="cuda", generator=g).to(torch.bfloat16)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    got = chandra_cuda.vit_attn(q, k, v, cu, max(lens), version=version).float()
    for i, ln in enumerate(lens):
        s = int(cu[i])
        qi, ki, vi = (x[s : s + ln].transpose(0, 1).float()[None] for x in (q, k, v))  # [1, 16, L, 64]
        if ln > 4096:  # check a slice of the queries of the long image (the reference is O(L^2))
            qi = qi[:, :, :: ln // 1024]
            ref = F.scaled_dot_product_attention(qi, ki, vi)[0].transpose(0, 1)
            sub = got[s : s + ln : ln // 1024]
        else:
            ref = F.scaled_dot_product_attention(qi, ki, vi)[0].transpose(0, 1)
            sub = got[s : s + ln]
        assert (sub - ref).abs().max().item() < 1e-2, (ln, (sub - ref).abs().max().item())


@pytest.mark.parametrize("version", [3, 5])
@pytest.mark.parametrize("lens", [[300], [1000, 77, 4096], [24552]])
def test_int8_qk_close_to_sdpa(lens, version):
    """Approximate (INT8 Q and K): relative L1 error against FP32, as SageAttention reports it."""
    g = torch.Generator(device="cuda").manual_seed(sum(lens) + 1)
    n = sum(lens)
    qkv = torch.randn(n, 3, 16, 64, device="cuda", generator=g).to(torch.bfloat16)
    qkv[:, 1] += 0.7  # a K mean, which the smoothing removes
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    got = chandra_cuda.vit_attn(q, k, v, cu, max(lens), version=version).float()
    for i, ln in enumerate(lens):
        s = int(cu[i])
        step = max(1, ln // 1024)
        qi, ki, vi = (x[s : s + ln].transpose(0, 1).float()[None] for x in (q, k, v))
        ref = F.scaled_dot_product_attention(qi[:, :, ::step], ki, vi)[0].transpose(0, 1)
        sub = got[s : s + ln : step]
        rel = ((sub - ref).abs().sum() / ref.abs().sum()).item()
        assert rel < 0.02, (ln, rel)

