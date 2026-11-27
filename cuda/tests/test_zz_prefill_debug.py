import pytest
import torch

from test_engine_spec import ckpt, VOCAB  # noqa: F401
from chandra_cuda.engine import Decoder


def test_prefill_cache_diff(ckpt):  # noqa: F811
    torch.manual_seed(21)
    n = 150
    prompt = torch.randint(0, VOCAB, (n,), device="cuda")
    pos3 = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].expand(n, 3).contiguous()
    a = Decoder(ckpt, max_seqs=1, max_ctx=512)
    for t in prompt:
        a.step(t[None], [0])
    b = Decoder(ckpt, max_seqs=1, max_ctx=512)
    b.prefill(0, b.embed[prompt], pos3, 0)
    for i, (La, Lb) in enumerate(zip(a.layers, b.layers)):
        if La["type"] == "linear_attention":
            sa, sb = La["state"][1].float(), Lb["state"][1].float()
            st = ((sa - sb).norm() / sa.norm()).item()
            stT = ((sa - sb.transpose(-1, -2)).norm() / sa.norm()).item()
            ring = [(La["ring"][1, p % 4].float() - Lb["ring"][1, p % 4].float()).abs().max().item() for p in range(n - 3, n)]
            print(f"layer {i} GDN: state rel {st:.2e} (transposed {stT:.2e}); ring max diff {ring}")
        else:
            ka = La["k_cache"][: 3].float()
            kb = Lb["k_cache"][: 3].float()
            va, vb = La["v_cache"][:3].float(), Lb["v_cache"][:3].float()
            print(f"layer {i} attn: K rel {((ka - kb).norm() / ka.norm()).item():.2e}, V rel {((va - vb).norm() / va.norm()).item():.2e}")
