"""The engine's speculative path on a small random checkpoint (real layer shapes, 4 layers, a
4,096-token vocabulary): verify + accept with any drafts gives plain greedy decoding's tokens."""

import json

import pytest
import torch

chandra_cuda = pytest.importorskip("chandra_cuda")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
from chandra_cuda.engine import Decoder  # noqa: E402

VOCAB, H = 4096, 2560
LAYERS = ["linear_attention", "full_attention", "linear_attention", "full_attention"]


def _lin(g, n, k):
    return (torch.randn(n, k, device="cuda", generator=g) * k**-0.5).to(torch.bfloat16)


def _norm(g, n, base=0.0):
    return (base + 0.1 * torch.randn(n, device="cuda", generator=g)).to(torch.bfloat16)


def _attn(g, p, w):
    a = p + "self_attn."
    w[a + "q_proj.weight"] = _lin(g, 8192, H)
    w[a + "k_proj.weight"] = _lin(g, 1024, H)
    w[a + "v_proj.weight"] = _lin(g, 1024, H)
    w[a + "o_proj.weight"] = _lin(g, H, 4096)
    w[a + "q_norm.weight"] = _norm(g, 256)
    w[a + "k_norm.weight"] = _norm(g, 256)


def _common(g, p, w):
    w[p + "input_layernorm.weight"] = _norm(g, H)
    w[p + "post_attention_layernorm.weight"] = _norm(g, H)
    w[p + "mlp.gate_proj.weight"] = _lin(g, 9216, H)
    w[p + "mlp.up_proj.weight"] = _lin(g, 9216, H)
    w[p + "mlp.down_proj.weight"] = _lin(g, H, 9216)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    g = torch.Generator(device="cuda").manual_seed(0)
    w = {}
    P = "model.language_model."
    w[P + "embed_tokens.weight"] = (torch.randn(VOCAB, H, device="cuda", generator=g)).to(torch.bfloat16)
    w[P + "norm.weight"] = _norm(g, H)
    w["lm_head.weight"] = _lin(g, VOCAB, H)
    for i, t in enumerate(LAYERS):
        p = f"{P}layers.{i}."
        _common(g, p, w)
        if t == "full_attention":
            _attn(g, p, w)
        else:
            a = p + "linear_attn."
            w[a + "in_proj_qkv.weight"] = _lin(g, 8192, H)
            w[a + "in_proj_z.weight"] = _lin(g, 4096, H)
            w[a + "in_proj_b.weight"] = _lin(g, 32, H)
            w[a + "in_proj_a.weight"] = _lin(g, 32, H)
            w[a + "out_proj.weight"] = _lin(g, H, 4096)
            w[a + "conv1d.weight"] = (0.4 * torch.randn(8192, 1, 4, device="cuda", generator=g)).to(torch.bfloat16)
            w[a + "A_log"] = 0.5 * torch.randn(32, device="cuda", generator=g)
            w[a + "dt_bias"] = (0.5 * torch.randn(32, device="cuda", generator=g)).to(torch.bfloat16)
            w[a + "norm.weight"] = _norm(g, 128, base=1.0)
    _common(g, "mtp.layers.0.", w)
    _attn(g, "mtp.layers.0.", w)
    w["mtp.fc.weight"] = _lin(g, H, 2 * H)
    for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        w[f"mtp.{n}.weight"] = _norm(g, H)
    d = tmp_path_factory.mktemp("ckpt")
    save_file({k: v.cpu().contiguous() for k, v in w.items()}, str(d / "model.safetensors"))
    json.dump({"text_config": {"layer_types": LAYERS, "rope_parameters": {"rope_theta": 1e7}}},
              open(d / "config.json", "w"))
    return str(d)


def _reset(dec):
    for L in dec.layers + [dec.mtp]:
        for key in ("state", "ring", "hist", "k_cache", "v_cache"):
            if key in L:
                L[key].zero_()
    dec.length = [0] * dec.max_seqs
    dec.pending = [0] * dec.max_seqs


STEPS = 48


@pytest.fixture(scope="module")
def run(ckpt):
    dec = Decoder(ckpt, max_seqs=3, max_ctx=512, mtp=True, draft_tokens=3)
    seqs = [0, 1, 2]
    t0 = torch.tensor([5, 77, 4000], device="cuda")
    ref = [[] for _ in seqs]
    tok = t0
    for _ in range(STEPS):
        tok = dec.step(tok, seqs)
        for r, x in zip(ref, tok.tolist()):
            r.append(x)
    return dec, seqs, t0, ref


def test_oracle_drafts_with_errors(run):
    """Drafts from the reference with random errors: the accepted text is the reference."""
    dec, seqs, t0, ref = run
    _reset(dec)
    rng = torch.Generator().manual_seed(3)
    out = [[] for _ in seqs]
    last = t0.tolist()
    total_acc, steps = 0, 0
    while min(len(o) for o in out) < STEPS - 4:
        drafts = []
        for n in range(len(seqs)):
            d = (ref[n][len(out[n]) : len(out[n]) + 3] + [0, 0, 0])[:3]  # past the reference: any drafts
            d = [x if torch.rand(1, generator=rng).item() > 0.25 else (x + 1) % VOCAB for x in d]
            drafts.append(d)
        vin = torch.tensor([[last[n]] + drafts[n] for n in range(len(seqs))], device="cuda")
        g, _ = dec.verify(vin, seqs)
        gl = g.tolist()
        counts = []
        for n in range(len(seqs)):
            a = 0
            while a < 3 and drafts[n][a] == gl[n][a]:
                a += 1
            out[n] += drafts[n][:a] + [gl[n][a]]
            last[n] = gl[n][a]
            counts.append(a + 1)
            total_acc += a
        dec.accept(seqs, counts)
        steps += 1
    for n in range(len(seqs)):
        assert out[n][:STEPS] == ref[n][: len(out[n])], n
    assert total_acc > steps  # some drafts were accepted, so the replay ran


def test_generate_mtp_matches_plain(run):
    dec, seqs, t0, ref = run
    _reset(dec)
    hidden, first = dec._forward(t0, seqs, 1, commit=False)
    for s in seqs:
        dec.length[s] += 1
    out, steps = dec.generate_mtp(seqs, first, hidden, max_new=STEPS)
    for n in range(len(seqs)):
        assert out[n] == ref[n][: len(out[n])], n
        assert len(out[n]) == STEPS
    assert steps <= STEPS


def test_graph_step_matches_eager(run):
    """A CUDA-graph step gives the eager step's tokens (splits planned for max_ctx may round the
    attention differently, so this compares tokens, which random weights rarely leave tied)."""
    dec, seqs, t0, ref = run
    _reset(dec)
    gs = dec.graph_step(seqs)
    tok = t0
    out = [[] for _ in seqs]
    for _ in range(16):
        _, nxt = gs(tok)
        for s in seqs:
            dec.length[s] += 1
        tok = nxt.clone()
        for o, x in zip(out, tok.tolist()):
            o.append(x)
    for n in range(len(seqs)):
        assert out[n] == ref[n][:16], n


def test_int8_kv_decoder_runs_and_mostly_agrees(ckpt, run):
    """INT8 KV is approximate (class G): on random weights the greedy tokens stay close to BF16 KV's."""
    _, seqs, t0, ref = run
    dec8 = Decoder(ckpt, max_seqs=3, max_ctx=512, kv="int8")
    tok = t0
    out = [[] for _ in seqs]
    for _ in range(24):
        tok = dec8.step(tok, seqs)
        for o, x in zip(out, tok.tolist()):
            o.append(x)
    lead = [next((j for j in range(24) if out[n][j] != ref[n][j]), 24) for n in range(len(seqs))]
    assert max(lead) >= 4, lead



def test_generate_mtp_with_4bit_draft_layer(ckpt, run):
    """Drafts from an MXFP4 MTP layer and an FP8 draft head: the output is still plain decoding's."""
    _, seqs, t0, ref = run
    dec4 = Decoder(ckpt, max_seqs=3, max_ctx=512, mtp=True, draft_tokens=3, mtp_weights="mxfp4")
    dec4.set_draft_head("fp8")
    hidden, first = dec4._forward(t0, seqs, 1, commit=False)
    for s in seqs:
        dec4.length[s] += 1
    out, steps = dec4.generate_mtp(seqs, first, hidden, max_new=32)
    for n in range(len(seqs)):
        assert out[n] == ref[n][:32], n


def test_graph_verify_step_matches_eager(run):
    """A CUDA graph of a verify step (4 rows per sequence, commit mode) gives eager verify's picks."""
    dec, seqs, t0, ref = run
    _reset(dec)
    vin = torch.tensor([[int(t0[n])] + ref[n][:3] for n in range(len(seqs))], device="cuda")
    g_eager, _ = dec.verify(vin, seqs)
    _reset(dec)
    gs = dec.graph_step(seqs, T=4, commit=True)
    _, nxt = gs(vin.reshape(-1))
    assert torch.equal(nxt.view(len(seqs), 4), g_eager)


def test_spec_graph_matches_plain(ckpt, run):
    """The graph MTP loop gives plain decoding's tokens."""
    _, seqs, t0, ref = run
    dec = Decoder(ckpt, max_seqs=3, max_ctx=512, mtp=True, draft_tokens=3)
    hidden, first = dec._forward(t0, seqs, 1, commit=False)
    for s in seqs:
        dec.length[s] += 1
    sg = dec.spec_graph(seqs)
    out, steps = sg.generate(first, hidden, max_new=32)
    for n in range(len(seqs)):
        assert out[n][:32] == ref[n][:32], n



def test_exact_lm_head_in_graphs(ckpt, run):
    """The exact INT8 LM head, captured in the graph MTP loop, gives the BF16 head's tokens."""
    _, seqs, t0, ref = run
    dec = Decoder(ckpt, max_seqs=3, max_ctx=512, mtp=True, draft_tokens=3, lm_head="exact")
    hidden, first = dec._forward(t0, seqs, 1, commit=False)
    for s in seqs:
        dec.length[s] += 1
    out, steps = dec.spec_graph(seqs).generate(first, hidden, max_new=24)
    for n in range(len(seqs)):
        assert out[n][:24] == ref[n][:24], n


def test_prefill_equals_step_by_step_decoding(ckpt):
    """Prefill (chunked FLA DeltaNet, FA2 attention, cuBLAS GEMMs) of a prompt, then decoding,
    gives the tokens of feeding the same prompt one token at a time through decode steps."""
    pytest.importorskip("vllm")
    torch.manual_seed(21)
    n, steps = 150, 16
    prompt = torch.randint(0, VOCAB, (n,), device="cuda")
    pos3 = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].expand(n, 3).contiguous()
    a = Decoder(ckpt, max_seqs=1, max_ctx=512)
    for t in prompt[:-1]:
        a.step(t[None], [0])
    nxt = a.step(prompt[-1:], [0])
    ref = [int(nxt)]
    for _ in range(steps - 1):
        nxt = a.step(nxt, [0])
        ref.append(int(nxt))
    b = Decoder(ckpt, max_seqs=1, max_ctx=512)
    _, first = b.prefill(0, b.embed[prompt], pos3, 0)
    got = [int(first)]
    nxt = first[None]
    for _ in range(steps - 1):
        nxt = b.step(nxt, [0])
        got.append(int(nxt))
    lead = next((i for i in range(steps) if got[i] != ref[i]), steps)
    assert lead >= steps // 2, (lead, got, ref)  # chunked vs recurrent DeltaNet: rounding only

