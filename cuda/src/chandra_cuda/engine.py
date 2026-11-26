"""A minimal decode engine for Chandra 2 built only from chandra_cuda's kernels.

It loads the language model's weights from the checkpoint into packed, merged
layouts (qkvz, ba, qkv with the output gate, gate_up), owns a paged KV cache for
the 8 full-attention layers and state slots for the 24 Gated DeltaNet layers, and
runs greedy decode steps for a batch of sequences. Prefill is not here yet: a
sequence's state is loaded from transformers' cache after its prefill
(`load_from_hf_cache`), which is how the kernels are validated end to end.

Speculative decoding with the checkpoint's MTP layer (`mtp=True`): `generate_mtp`
drafts k tokens with the MTP layer, verifies them in one step of the main model
(k + 1 query rows per sequence: attention reads each sequence's KV once, the
DeltaNet layers read and write their state once in commit mode), and keeps the
longest prefix the main model agrees with, plus the main model's next token.
"""

from __future__ import annotations

import json
import math
import os

import torch

from . import (_core, DecodeLinear, LMHead, LMHeadFP8, attn_decode, gdn_decode, qk_norm_rope_kv, rmsnorm_add,
               silu_mul)

PAGE = 64
ROWS = 64  # the decode GEMMs take up to 64 rows per call


def _rows(fn, x: torch.Tensor) -> torch.Tensor:
    """fn over row blocks of at most 64 (the decode kernels are batch-invariant, so this is exact)."""
    if x.shape[0] <= ROWS:
        return fn(x)
    return torch.cat([fn(x[i : i + ROWS]) for i in range(0, x.shape[0], ROWS)])


class Decoder:
    def __init__(self, ckpt_dir: str, device: str = "cuda", max_seqs: int = 8, max_ctx: int = 18432,
                 state_dtype: torch.dtype = torch.float32, mtp: bool = False, draft_tokens: int = 3):
        from safetensors import safe_open

        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))["text_config"]
        self.layer_types = cfg["layer_types"]
        self.rope_theta = float(cfg["rope_parameters"]["rope_theta"])
        self.device = torch.device(device)
        self.ckpt_dir = ckpt_dir
        self.max_seqs, self.max_ctx = max_seqs, max_ctx
        self.k = draft_tokens if mtp else 0
        self.t_max = self.k + 1  # query rows per sequence in a verify step
        f = safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt", device=str(self.device))
        g = f.get_tensor
        P = "model.language_model."
        self.embed = g(P + "embed_tokens.weight")
        self.final_w = g(P + "norm.weight")
        self.lm_head = LMHead(g("lm_head.weight"))
        self.draft_head = self.lm_head

        def attn_layer(p):
            a = p + "self_attn."
            return dict(
                type="full_attention",
                w_in=g(p + "input_layernorm.weight"),
                w_post=g(p + "post_attention_layernorm.weight"),
                gate_up=DecodeLinear(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])),
                down=DecodeLinear(g(p + "mlp.down_proj.weight")),
                qkv=DecodeLinear(torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")])),
                o=DecodeLinear(g(a + "o_proj.weight")),
                q_norm=g(a + "q_norm.weight"),
                k_norm=g(a + "k_norm.weight"),
            )

        self.layers = []
        for i, t in enumerate(self.layer_types):
            p = f"{P}layers.{i}."
            if t != "linear_attention":
                self.layers.append(attn_layer(p))
                continue
            a = p + "linear_attn."
            self.layers.append(dict(
                type=t,
                w_in=g(p + "input_layernorm.weight"),
                w_post=g(p + "post_attention_layernorm.weight"),
                gate_up=DecodeLinear(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])),
                down=DecodeLinear(g(p + "mlp.down_proj.weight")),
                qkvz=DecodeLinear(torch.cat([g(a + "in_proj_qkv.weight"), g(a + "in_proj_z.weight")])),
                ba=DecodeLinear(torch.cat([g(a + "in_proj_b.weight"), g(a + "in_proj_a.weight")])),
                out=DecodeLinear(g(a + "out_proj.weight")),
                conv_w=g(a + "conv1d.weight").reshape(8192, 4).contiguous(),
                A_log=g(a + "A_log").float(),
                dt_bias=g(a + "dt_bias"),
                norm_w=g(a + "norm.weight"),
            ))
        self.mtp = None
        if mtp:
            self.mtp = attn_layer("mtp.layers.0.")
            self.mtp.update(
                fc=DecodeLinear(g("mtp.fc.weight")),
                w_emb=g("mtp.pre_fc_norm_embedding.weight"),
                w_hid=g("mtp.pre_fc_norm_hidden.weight"),
                w_norm=g("mtp.norm.weight"),
            )
        torch.cuda.empty_cache()

        # Caches. Sequence s owns pages [s * pps, (s + 1) * pps) and state / ring slot s + 1.
        dev = self.device
        self.pps = math.ceil(max_ctx / PAGE)
        pages = max_seqs * self.pps
        self.ring = 4 if self.t_max == 1 else 8  # >= 3 + rows per step
        self.block_table = torch.arange(pages, dtype=torch.int32, device=dev).reshape(max_seqs, self.pps)
        attn_layers = [L for L in self.layers if L["type"] != "linear_attention"] + ([self.mtp] if mtp else [])
        for L in self.layers:
            if L["type"] == "linear_attention":
                L["state"] = torch.zeros(max_seqs + 1, 32, 128, 128, dtype=state_dtype, device=dev)
                L["ring"] = torch.zeros(max_seqs + 1, self.ring, 8192, dtype=torch.bfloat16, device=dev)
                if self.t_max > 1:  # the accepted tokens' updates, replayed by the next step
                    L["hist"] = torch.zeros(max_seqs + 1, self.t_max, 32, _core.gdn_hist_floats,
                                            dtype=torch.float32, device=dev)
        for L in attn_layers:
            L["k_cache"] = torch.zeros(pages, 4, PAGE, 256, dtype=torch.bfloat16, device=dev)
            L["v_cache"] = torch.zeros_like(L["k_cache"])
        self.length = [0] * max_seqs  # tokens in the cache per sequence
        self.rope_delta = [0] * max_seqs
        self.pending = [0] * max_seqs  # DeltaNet updates to replay at the next step (commit mode)

    @torch.no_grad()
    def load_from_hf_cache(self, seq: int, cache, batch_index: int, length: int, rope_delta: int):
        """Copy one sequence's prefill state out of a transformers Qwen3.5 cache."""
        for i, L in enumerate(self.layers):
            if L["type"] == "linear_attention":
                rec = cache.get_recurrent_state(i)[batch_index]  # [32, K, V]
                L["state"][seq + 1] = rec.transpose(-1, -2).to(L["state"].dtype)  # ours: [v][k]
                conv = cache.get_conv_state(i)[batch_index]  # [8192, w]: last w raw inputs, oldest first
                w = conv.shape[-1]
                for j in range(1, 4):  # positions length - j
                    pos = length - j
                    if pos >= 0:
                        L["ring"][seq + 1, pos % self.ring] = conv[:, w - j].to(torch.bfloat16)
            else:
                lay = cache.layers[i]
                k = lay.keys[batch_index]  # [4, L, 256]
                v = lay.values[batch_index]
                for p in range(math.ceil(length / PAGE)):
                    page = int(self.block_table[seq, p])
                    n = min(PAGE, length - p * PAGE)
                    L["k_cache"][page, :, :n] = k[:, p * PAGE : p * PAGE + n]
                    L["v_cache"][page, :, :n] = v[:, p * PAGE : p * PAGE + n]
        self.length[seq] = length
        self.rope_delta[seq] = int(rope_delta)
        self.pending[seq] = 0

    def _positions(self, seqs, start, T):
        """Per-row tensors for T tokens of each sequence from position start[n]."""
        dev = self.device
        seq_t = torch.tensor(seqs, dtype=torch.int64, device=dev)
        st = torch.tensor(start, dtype=torch.int32, device=dev)
        pos = (st[:, None] + torch.arange(T, dtype=torch.int32, device=dev)).reshape(-1)  # [B * T]
        delta = torch.tensor([self.rope_delta[s] for s in seqs], dtype=torch.int32, device=dev)
        rope = pos + delta.repeat_interleave(T)
        pos3 = rope[:, None].expand(-1, 3).contiguous()
        page = self.block_table[seq_t.repeat_interleave(T), (pos // PAGE).long()].contiguous()
        in_page = (pos % PAGE).to(torch.int32)
        bt = self.block_table[seq_t].contiguous()
        return st, pos3, page, in_page, bt

    def _attention(self, L, xn, pos3, page, in_page, bt, ctx, T, max_ctx):
        m = xn.shape[0]
        qkv = _rows(L["qkv"], xn)
        q = qk_norm_rope_kv(qkv, L["q_norm"], L["k_norm"], pos3, L["k_cache"], L["v_cache"], page, in_page,
                            rope_theta=self.rope_theta)
        gate = qkv[:, :8192].view(m, 16, 512)[..., 256:]
        att = attn_decode(q, gate, L["k_cache"], L["v_cache"], bt, ctx, n_q=T, max_ctx=max_ctx)
        return _rows(L["o"], att.reshape(m, 4096))

    @torch.no_grad()
    def _forward(self, tokens: torch.Tensor, seqs: list[int], T: int, commit: bool):
        """The main model on T tokens per sequence at positions length .. length + T - 1 (rows n * T + t).
        Returns the final-norm hidden states [B * T, 2560] and the greedy next tokens [B * T]."""
        dev = self.device
        b = len(seqs)
        lens, pos3, page, in_page, bt = self._positions(seqs, [self.length[s] for s in seqs], T)
        slot = torch.tensor([s + 1 for s in seqs], dtype=torch.int32, device=dev)
        ctx = lens + T
        max_ctx = max(self.length[s] for s in seqs) + T
        replay = any(self.pending[s] for s in seqs)
        n_rep = torch.tensor([self.pending[s] for s in seqs], dtype=torch.int32, device=dev) if replay or commit else None

        res = self.embed[tokens.to(dev)]
        prev = None
        for L in self.layers:
            xn, res = rmsnorm_add(res, L["w_in"], x=prev)
            if L["type"] == "linear_attention":
                qkvz = _rows(L["qkvz"], xn)
                ba = _rows(L["ba"], xn)
                hist = L.get("hist")
                o = gdn_decode(qkvz[:, :8192], qkvz[:, 8192:], ba[:, :32], ba[:, 32:], L["A_log"], L["dt_bias"],
                               L["conv_w"], L["norm_w"], L["state"], slot, slot, L["ring"], slot, lens, T=T,
                               n_replay=n_rep, hist_in=hist if n_rep is not None else None,
                               hist_out=hist if commit else None, commit=commit)
                y = _rows(L["out"], o)
            else:
                y = self._attention(L, xn, pos3, page, in_page, bt, ctx, T, max_ctx)
            xn, res = rmsnorm_add(res, L["w_post"], x=y)
            prev = _rows(L["down"], silu_mul(_rows(L["gate_up"], xn)))
        hidden, _ = rmsnorm_add(res, self.final_w, x=prev)
        for s in seqs:
            self.pending[s] = 0
        return hidden, _rows(self.lm_head, hidden)

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, seqs: list[int]) -> torch.Tensor:
        """One greedy decode step: tokens [B] (the last token of each sequence) -> next tokens [B]."""
        _, nxt = self._forward(tokens, seqs, 1, commit=False)
        for s in seqs:
            self.length[s] += 1
        return nxt

    @torch.no_grad()
    def verify(self, tokens: torch.Tensor, seqs: list[int]):
        """tokens [B, T]: each sequence's last token and its drafts. Returns (greedy tokens [B, T],
        hidden [B, T, 2560]); the cache keeps them until `accept`."""
        b, T = tokens.shape
        hidden, nxt = self._forward(tokens.reshape(-1), seqs, T, commit=True)
        return nxt.view(b, T), hidden.view(b, T, -1)

    def accept(self, seqs: list[int], counts: list[int]):
        """After `verify`: the first counts[n] input tokens of sequence n are kept."""
        for s, c in zip(seqs, counts):
            self.length[s] += c
            self.pending[s] = c

    # ---- the MTP draft layer ----------------------------------------------------------------

    def _mtp_input(self, tokens, hidden):
        M = self.mtp
        en, _ = rmsnorm_add(self.embed[tokens.to(self.device)], M["w_emb"])
        hn, _ = rmsnorm_add(hidden.contiguous(), M["w_hid"])
        return _rows(M["fc"], torch.cat([en, hn], dim=1))

    @torch.no_grad()
    def mtp_prefill(self, seq: int, next_tokens: torch.Tensor, hidden: torch.Tensor, pos3: torch.Tensor):
        """The MTP layer's KV over a prompt: row i = (token i + 1, the main model's final-norm hidden
        state at position i, its (t, h, w) position). Rows 0 .. n - 2 of an n-token prompt."""
        M = self.mtp
        dev = self.device
        n = next_tokens.shape[0]
        for i in range(0, n, 1024):
            j = min(n, i + 1024)
            x = self._mtp_input(next_tokens[i:j], hidden[i:j])
            xn, _ = rmsnorm_add(x, M["w_in"])
            qkv = _rows(M["qkv"], xn)
            pos = torch.arange(i, j, dtype=torch.int64, device=dev)
            page = self.block_table[seq, pos // PAGE].contiguous()
            in_page = (pos % PAGE).to(torch.int32)
            qk_norm_rope_kv(qkv, M["q_norm"], M["k_norm"], pos3[i:j].to(torch.int32), M["k_cache"], M["v_cache"],
                            page, in_page, rope_theta=self.rope_theta)

    @torch.no_grad()
    def mtp_forward(self, tokens: torch.Tensor, hidden: torch.Tensor, seqs: list[int], start: list[int], T: int):
        """The MTP layer on T rows per sequence at positions start[n] .. + T - 1 (row = token at the
        next position and the hidden state at this one). Returns (its final-norm hidden states, drafts)."""
        M = self.mtp
        st, pos3, page, in_page, bt = self._positions(seqs, start, T)
        x = self._mtp_input(tokens, hidden)
        xn, res = rmsnorm_add(x, M["w_in"])
        y = self._attention(M, xn, pos3, page, in_page, bt, st + T, T, max(start) + T)
        xn, res = rmsnorm_add(res, M["w_post"], x=y)
        z = _rows(M["down"], silu_mul(_rows(M["gate_up"], xn)))
        h, _ = rmsnorm_add(res, M["w_norm"], x=z)
        return h, _rows(self.draft_head, h)

    def set_draft_head(self, kind: str = "bf16", rows: torch.Tensor | None = None):
        """The LM head the MTP drafts use: "bf16" (the main model's), or "fp8" with an optional
        draft vocabulary (token ids). Drafts only change how many tokens a step keeps, never which."""
        if kind == "bf16":
            self.draft_head = self.lm_head
        elif kind == "fp8":
            from safetensors import safe_open

            f = safe_open(os.path.join(self.ckpt_dir, "model.safetensors"), "pt", device=str(self.device))
            self.draft_head = LMHeadFP8(f.get_tensor("lm_head.weight"), rows=rows)
            torch.cuda.empty_cache()
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def generate_mtp(self, seqs: list[int], first: torch.Tensor, h_last: torch.Tensor, max_new: int,
                     eos: set[int] = frozenset()):
        """Greedy decoding with MTP drafts. first [B]: each sequence's first generated token (from the
        prefill); h_last [B, 2560]: the main model's final-norm hidden state at the last prompt
        position, whose MTP KV rows before it are in place (`mtp_prefill`). Returns the token lists
        (starting with first, at most max_new) and the number of verify steps."""
        dev = self.device
        k = self.k
        out = [[int(t)] for t in first.tolist()]
        act = [n for n in range(len(seqs)) if out[n][0] not in eos and max_new > 1]
        if not act:
            return out, 0
        sel = torch.tensor(act, device=dev)
        sq = [seqs[n] for n in act]
        last = first.to(dev)[sel]
        start = [self.length[s] - 1 for s in sq]
        hd, e = self.mtp_forward(last, h_last[sel], sq, start, 1)
        drafts = [e]
        for j in range(1, k):
            hd, e = self.mtp_forward(e, hd, sq, [p + j for p in start], 1)
            drafts.append(e)
        steps = 0
        while act:
            sq = [seqs[n] for n in act]
            vin = torch.stack([last] + drafts, dim=1)  # [B, k + 1]
            g, H = self.verify(vin, sq)
            steps += 1
            gl, dl = g.tolist(), vin.tolist()
            acc, keep = [], []
            for i, n in enumerate(act):
                a = 0
                while a < k and dl[i][a + 1] == gl[i][a]:
                    a += 1
                acc.append(a)
                live = True
                for t in dl[i][1 : a + 1] + [gl[i][a]]:
                    out[n].append(t)
                    if t in eos or len(out[n]) >= max_new:
                        live = False
                        break
                if live:
                    keep.append(i)
            p0 = [self.length[s] for s in sq]
            self.accept(sq, [a + 1 for a in acc])
            if not keep:
                break
            # MTP rows at p0 .. p0 + k: (token at the next position, the verified hidden state); row a
            # (the last accepted position) gives the first draft of the next step.
            act = [act[i] for i in keep]
            sq = [sq[i] for i in keep]
            acc = [acc[i] for i in keep]
            p0 = [p0[i] for i in keep]
            gl = [gl[i] for i in keep]
            dl = [dl[i] for i in keep]
            b = len(act)
            mt = torch.tensor([[dl[n][i + 1] if i < acc[n] else gl[n][acc[n]] for i in range(k + 1)]
                               for n in range(b)], device=dev)
            hk = H[torch.tensor(keep, device=dev)].reshape(b * (k + 1), -1)
            hd_all, e_all = self.mtp_forward(mt.reshape(-1), hk, sq, p0, k + 1)
            row = torch.tensor([n * (k + 1) + acc[n] for n in range(b)], device=dev)
            hd, e = hd_all[row], e_all[row]
            drafts = [e]
            for j in range(1, k):
                hd, e = self.mtp_forward(e, hd, sq, [p0[n] + acc[n] + j for n in range(b)], 1)
                drafts.append(e)
            last = torch.tensor([gl[n][acc[n]] for n in range(b)], device=dev)
        return out, steps
