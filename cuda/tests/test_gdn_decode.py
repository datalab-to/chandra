"""gdn_decode against an FP32 PyTorch reference of Chandra 2's Gated DeltaNet decode step,
over several consecutive steps from position 0 (which exercises the conv ring)."""

import pytest
import torch
import torch.nn.functional as F

chandra_cuda = pytest.importorskip("chandra_cuda")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

HK, HV, D, C = 16, 32, 128, 8192


def bf(x):
    return x.to(torch.bfloat16).float()


def reference_step(qkv, z, b, a, A_log, dt_bias, conv_w, norm_w, S, hist):
    """hist: [B, 3, 8192] raw inputs of the 3 previous positions (oldest first, zeros before 0)."""
    win = torch.cat([hist, qkv.float()[:, None]], dim=1)  # [B, 4, C]
    x = bf(F.silu((win * conv_w.float().T[None]).sum(1)))
    q = x[:, :2048].view(-1, HK, D)
    k = x[:, 2048:4096].view(-1, HK, D)
    v = x[:, 4096:].view(-1, HV, D)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k = q.repeat_interleave(2, dim=1), k.repeat_interleave(2, dim=1)
    decay = torch.exp(-torch.exp(A_log)[None] * F.softplus(a.float() + dt_bias.float()[None]))
    beta = torch.sigmoid(b.float())
    S = decay[..., None, None] * S
    vp = beta[..., None] * (v - torch.einsum("bhvk,bhk->bhv", S, k))
    S = S + vp[..., None] * k[:, :, None, :]
    o = bf(torch.einsum("bhvk,bhk->bhv", S, q))
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6) * norm_w.float() * F.silu(z.float().view(-1, HV, D))
    return o.reshape(-1, HV * D).to(torch.bfloat16), S, torch.cat([hist[:, 1:], qkv.float()[:, None]], dim=1)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch", [1, 5])
def test_steps_match_reference(state_dtype, batch):
    g = torch.Generator(device="cuda").manual_seed(batch)
    A_log = torch.randn(HV, device="cuda", generator=g) * 0.5
    dt_bias = (torch.randn(HV, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    conv_w = (torch.randn(C, 4, device="cuda", generator=g) * 0.4).to(torch.bfloat16)
    norm_w = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    slots = batch + 2
    state = torch.zeros(slots, HV, D, D, device="cuda", dtype=state_dtype)
    ring = torch.zeros(slots, 4, C, device="cuda", dtype=torch.bfloat16)
    slot = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    S_ref = torch.zeros(batch, HV, D, D, device="cuda")
    hist = torch.zeros(batch, 3, C, device="cuda")
    for step in range(6):
        qkv = (torch.randn(batch, C, device="cuda", generator=g)).to(torch.bfloat16)
        z = torch.randn(batch, HV * D, device="cuda", generator=g).to(torch.bfloat16)
        ba = torch.randn(batch, 2 * HV, device="cuda", generator=g).to(torch.bfloat16)
        b, a = ba[:, :HV], ba[:, HV:]
        pos = torch.full((batch,), step, dtype=torch.int32, device="cuda")
        out = chandra_cuda.gdn_decode(qkv, z, b, a, A_log, dt_bias, conv_w, norm_w, state, slot, slot, ring, slot, pos)
        ref, S_ref, hist = reference_step(qkv, z, b, a, A_log, dt_bias, conv_w, norm_w, S_ref, hist)
        tol = 2e-2 if state_dtype == torch.float32 else 4e-2
        err = (out.float() - ref.float()).abs().max().item()
        assert err < tol * max(1.0, ref.float().abs().max().item()), (step, err)
        # The conv output is rounded to BF16 on both sides; the kernel's fast exp in SiLU can put
        # a value on the other side of a rounding boundary (one BF16 ulp, ~0.4%), so the bound
        # is relative to the state's size. Wrong heads or layouts give O(1) errors.
        s_err = (state[1 : batch + 1].float() - S_ref).abs().max().item()
        s_tol = (5e-3 if state_dtype == torch.float32 else 1e-2) * S_ref.abs().max().item() + 1e-5
        assert s_err < s_tol, (step, s_err, s_tol)
    assert state[0].abs().sum() == 0 and state[-1].abs().sum() == 0  # other slots untouched


def test_separate_in_and_out_slots():
    g = torch.Generator(device="cuda").manual_seed(3)
    A_log = torch.zeros(HV, device="cuda")
    dt_bias = torch.zeros(HV, device="cuda", dtype=torch.bfloat16)
    conv_w = (torch.randn(C, 4, device="cuda", generator=g) * 0.4).to(torch.bfloat16)
    norm_w = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(3, HV, D, D, device="cuda", generator=g)
    before = state[1].clone()
    ring = torch.zeros(3, 4, C, device="cuda", dtype=torch.bfloat16)
    qkv = torch.randn(1, C, device="cuda", generator=g).to(torch.bfloat16)
    z = torch.randn(1, HV * D, device="cuda", generator=g).to(torch.bfloat16)
    ba = torch.randn(1, 2 * HV, device="cuda", generator=g).to(torch.bfloat16)
    one = torch.tensor([1], dtype=torch.int32, device="cuda")
    two = torch.tensor([2], dtype=torch.int32, device="cuda")
    zero = torch.tensor([0], dtype=torch.int32, device="cuda")
    chandra_cuda.gdn_decode(qkv, z, ba[:, :HV], ba[:, HV:], A_log, dt_bias, conv_w, norm_w, state, one, two, ring, one, zero)
    assert torch.equal(state[1], before)  # the input slot is only read
    assert not torch.equal(state[2], before)


def _layer(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        A_log=torch.randn(HV, device="cuda", generator=g) * 0.5,
        dt_bias=(torch.randn(HV, device="cuda", generator=g) * 0.5).to(torch.bfloat16),
        conv_w=(torch.randn(C, 4, device="cuda", generator=g) * 0.4).to(torch.bfloat16),
        norm_w=(1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16),
    ), g


def _tokens(g, rows):
    qkv = torch.randn(rows, C, device="cuda", generator=g).to(torch.bfloat16)
    z = torch.randn(rows, HV * D, device="cuda", generator=g).to(torch.bfloat16)
    ba = torch.randn(rows, 2 * HV, device="cuda", generator=g).to(torch.bfloat16)
    return qkv, z, ba


def _i32(*v):
    return torch.tensor(v, dtype=torch.int32, device="cuda")


def test_verify_equals_sequential_steps():
    lay, g = _layer(11)
    batch, T, p0 = 3, 4, 9
    state0 = torch.randn(batch + 1, HV, D, D, device="cuda", generator=g) * 0.1
    ring0 = torch.randn(batch + 1, 7, C, device="cuda", generator=g).to(torch.bfloat16)
    qkv, z, ba = _tokens(g, batch * T)
    slot = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    pos = torch.full((batch,), p0, dtype=torch.int32, device="cuda")
    # One verify step over the 4 tokens.
    st_v, ring_v = state0.clone(), ring0.clone()
    out_v = chandra_cuda.gdn_decode(qkv, z, ba[:, :HV], ba[:, HV:], lay["A_log"], lay["dt_bias"], lay["conv_w"],
                                    lay["norm_w"], st_v, slot, slot, ring_v, slot, pos, T=T)
    # Four decode steps.
    st_s, ring_s = state0.clone(), ring0.clone()
    rows = lambda t_: torch.arange(batch, device="cuda") * T + t_
    for t in range(T):
        r = rows(t)
        out_t = chandra_cuda.gdn_decode(qkv[r], z[r], ba[r, :HV], ba[r, HV:], lay["A_log"], lay["dt_bias"],
                                        lay["conv_w"], lay["norm_w"], st_s, slot, slot, ring_s, slot, pos + t)
        assert torch.equal(out_t, out_v[r]), t
    assert torch.equal(st_s, st_v)
    assert torch.equal(ring_s, ring_v)


@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_replay_gives_the_state_after_the_accepted_tokens(accepted):
    lay, g = _layer(20 + accepted)
    T, p0 = 4, 5
    state = torch.randn(3, HV, D, D, device="cuda", generator=g) * 0.1  # slot 1: start, slot 2: verify output
    ring0 = torch.randn(2, 8, C, device="cuda", generator=g).to(torch.bfloat16)
    hist = torch.zeros(2, T, HV, chandra_cuda._core.gdn_hist_floats, device="cuda")
    qkv, z, ba = _tokens(g, T)
    nxt_qkv, nxt_z, nxt_ba = _tokens(g, 1)  # the token after the accepted prefix
    start = state[1].clone()
    ring = ring0.clone()
    chandra_cuda.gdn_decode(qkv, z, ba[:, :HV], ba[:, HV:], lay["A_log"], lay["dt_bias"], lay["conv_w"],
                            lay["norm_w"], state, _i32(1), _i32(2), ring, _i32(1), _i32(p0), T=T, hist_out=hist)
    assert torch.equal(state[1], start)
    # Next step: start from slot 1 again, replay the accepted tokens, then the new token.
    out = chandra_cuda.gdn_decode(nxt_qkv, nxt_z, nxt_ba[:, :HV], nxt_ba[:, HV:], lay["A_log"], lay["dt_bias"],
                                  lay["conv_w"], lay["norm_w"], state, _i32(1), _i32(2), ring, _i32(1),
                                  _i32(p0 + accepted), n_replay=_i32(accepted), hist_in=hist)
    # Reference: the accepted tokens and the new one as plain decode steps.
    ref_state = torch.zeros_like(state)
    ref_state[1] = start
    ref_ring = ring0.clone()
    for t in range(accepted):
        chandra_cuda.gdn_decode(qkv[t : t + 1], z[t : t + 1], ba[t : t + 1, :HV], ba[t : t + 1, HV:], lay["A_log"],
                                lay["dt_bias"], lay["conv_w"], lay["norm_w"], ref_state, _i32(1), _i32(1), ref_ring,
                                _i32(1), _i32(p0 + t))
    ref_out = chandra_cuda.gdn_decode(nxt_qkv, nxt_z, nxt_ba[:, :HV], nxt_ba[:, HV:], lay["A_log"], lay["dt_bias"],
                                      lay["conv_w"], lay["norm_w"], ref_state, _i32(1), _i32(1), ref_ring, _i32(1),
                                      _i32(p0 + accepted))
    assert torch.equal(state[2], ref_state[1])
    assert torch.equal(out, ref_out)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_commit_mode_rounds_equal_plain_decode(state_dtype):
    """Verify rounds in commit mode (one slot, history in place, random acceptance) give the
    outputs of plain one-token steps, and the slot holds the committed state: bit for bit with an
    FP32 state. An FP16 state is rounded once per step here and once per token in plain steps."""
    lay, g = _layer(40)
    batch, T, rounds = 3, 4, 7
    state = (torch.randn(batch + 1, HV, D, D, device="cuda", generator=g) * 0.1).to(state_dtype)
    ring = torch.randn(batch + 1, 8, C, device="cuda", generator=g).to(torch.bfloat16)
    ref_state, ref_ring = state.clone(), ring.clone()
    hist = torch.zeros(batch + 1, T, HV, chandra_cuda._core.gdn_hist_floats, device="cuda")
    slot = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    pos = [5, 9, 2]
    n_rep = [0] * batch
    acc_g = torch.Generator().manual_seed(7)

    def same(got, ref, where):
        if state_dtype == torch.float32:
            assert torch.equal(got, ref), where
        else:
            err = (got.float() - ref.float()).abs().max().item()
            assert err <= 1e-2 * ref.float().abs().max().item() + 1e-3, (where, err)
    for r in range(rounds):
        qkv, z, ba = _tokens(g, batch * T)
        out = chandra_cuda.gdn_decode(qkv, z, ba[:, :HV], ba[:, HV:], lay["A_log"], lay["dt_bias"],
                                      lay["conv_w"], lay["norm_w"], state, slot, slot, ring, slot,
                                      _i32(*pos), T=T, n_replay=_i32(*n_rep), hist_in=hist, hist_out=hist,
                                      commit=True)
        # The slot now holds the committed state: the reference's, before this round's tokens.
        same(state[1:], ref_state[1:], (r,))
        accepted = [int(torch.randint(1, T + 1, (1,), generator=acc_g)) for _ in range(batch)]
        for n in range(batch):
            sl = _i32(n + 1)
            for t in range(T):  # every token's output, from plain steps on the reference
                row = n * T + t
                o = chandra_cuda.gdn_decode(qkv[row : row + 1], z[row : row + 1], ba[row : row + 1, :HV],
                                            ba[row : row + 1, HV:], lay["A_log"], lay["dt_bias"], lay["conv_w"],
                                            lay["norm_w"], ref_state, sl, sl, ref_ring, sl, _i32(pos[n] + t))
                same(out[row : row + 1], o, (r, n, t))
                if t + 1 == accepted[n]:  # keep the state after the accepted tokens
                    keep_s, keep_r = ref_state[n + 1].clone(), ref_ring[n + 1].clone()
            ref_state[n + 1], ref_ring[n + 1] = keep_s, keep_r
            pos[n] += accepted[n]
        n_rep = accepted

