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

from . import (_core, DecodeLinear, DecodeLinearFP4, DecodeLinearFP8, DecodeLinearINT4, LMHead, LMHeadFP8,
               attn_decode, gdn_decode, qk_norm_rope_kv, rmsnorm_add)

WEIGHT_FORMATS = {"bf16": DecodeLinear, "fp8": DecodeLinearFP8, "mxfp4": DecodeLinearFP4, "int4": DecodeLinearINT4}

PAGE = 64
ROWS = 64  # the decode GEMMs take up to 64 rows per call


def _quant_kv(x: torch.Tensor):
    """[heads, L, 256] -> (INT8 values, FP32 scales [heads, L]): the INT8 KV write's rounding."""
    xf = x.float()
    mx = xf.abs().amax(dim=-1)
    scale = torch.where(mx > 0, mx / 127.0, torch.ones_like(mx))
    q = torch.round(xf * (1.0 / scale)[..., None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def _rows(fn, x: torch.Tensor) -> torch.Tensor:
    """fn over row blocks of at most 64 (the decode kernels are batch-invariant, so this is exact)."""
    if x.shape[0] <= ROWS:
        return fn(x)
    return torch.cat([fn(x[i : i + ROWS]) for i in range(0, x.shape[0], ROWS)])


class Decoder:
    def __init__(self, ckpt_dir: str, device: str = "cuda", max_seqs: int = 8, max_ctx: int = 18432,
                 state_dtype: torch.dtype = torch.float32, mtp: bool = False, draft_tokens: int = 3,
                 weights: str = "bf16", lm_head: str = "bf16", kv: str = "bf16", attn_split_keys: int | None = 1024,
                 mtp_weights: str = "bf16", autotune: bool = True):
        """weights: the format of the decoder layers' linear weights (bf16; fp8, mxfp4, int4 are
        approximate, class G). lm_head: "bf16", or "exact" (INT8 pass + BF16 re-check, the same picks).
        mtp_weights: the MTP layer's GEMM format (drafts are verified, so any format changes only how
        many tokens a step keeps). kv: "bf16", or "int8" (symmetric per token and head, class G)."""
        if kv not in ("bf16", "int8"):
            raise ValueError(kv)
        self.kv = kv
        # Fixed attention chunks: results independent of the batch, and MTP verify steps round
        # exactly as plain steps (None: chunks sized to fill the GPU, faster at batch 1-4).
        self.attn_split_keys = attn_split_keys
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
        if lm_head == "exact":
            from . import LMHeadExact

            self.lm_head = LMHeadExact(g("lm_head.weight"), self.draft_head)
        elif lm_head != "bf16":
            raise ValueError(lm_head)
        Lin = WEIGHT_FORMATS[weights]

        def attn_layer(p, Lin=Lin):
            a = p + "self_attn."
            return dict(
                type="full_attention",
                w_in=g(p + "input_layernorm.weight"),
                w_post=g(p + "post_attention_layernorm.weight"),
                gate_up=Lin(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]),
                            epilogue="silu_mul"),
                down=Lin(g(p + "mlp.down_proj.weight")),
                qkv=Lin(torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")])),
                o=Lin(g(a + "o_proj.weight")),
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
                gate_up=Lin(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]),
                            epilogue="silu_mul"),
                down=Lin(g(p + "mlp.down_proj.weight")),
                # One GEMM for qkv | z | b | a, padded to 12,416 rows (a whole number of 128-row tiles).
                qkvzba=Lin(torch.cat([g(a + "in_proj_qkv.weight"), g(a + "in_proj_z.weight"),
                                      g(a + "in_proj_b.weight"), g(a + "in_proj_a.weight"),
                                      torch.zeros(64, 2560, dtype=torch.bfloat16, device=self.device)])),
                out=Lin(g(a + "out_proj.weight")),
                conv_w=g(a + "conv1d.weight").reshape(8192, 4).contiguous(),
                A_log=g(a + "A_log").float(),
                dt_bias=g(a + "dt_bias"),
                norm_w=g(a + "norm.weight"),
            ))
        self.mtp = None
        if mtp:
            self.mtp = attn_layer("mtp.layers.0.", Lin=WEIGHT_FORMATS[mtp_weights])
            self.mtp.update(
                fc=WEIGHT_FORMATS[mtp_weights](g("mtp.fc.weight")),
                w_emb=g("mtp.pre_fc_norm_embedding.weight"),
                w_hid=g("mtp.pre_fc_norm_hidden.weight"),
                w_norm=g("mtp.norm.weight"),
            )
        torch.cuda.empty_cache()
        if autotune:  # each decode GEMM's plan, once per GPU and shape (cached on disk)
            from . import autotune_plans

            lins = [v for L in self.layers + ([self.mtp] if self.mtp else []) for v in L.values()
                    if isinstance(v, DecodeLinear)]
            self.plans = autotune_plans(lins)

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
            dt = torch.int8 if kv == "int8" else torch.bfloat16
            L["k_cache"] = torch.zeros(pages, 4, PAGE, 256, dtype=dt, device=dev)
            L["v_cache"] = torch.zeros_like(L["k_cache"])
            if kv == "int8":
                L["k_scale"] = torch.ones(pages, 4, PAGE, dtype=torch.float32, device=dev)
                L["v_scale"] = torch.ones_like(L["k_scale"])
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
                if self.kv == "int8":  # as the INT8 KV write: amax / 127 per (head, token), round to nearest
                    qk, sk = _quant_kv(k[:, :length])
                    qv, sv = _quant_kv(v[:, :length])
                for p in range(math.ceil(length / PAGE)):
                    page = int(self.block_table[seq, p])
                    n = min(PAGE, length - p * PAGE)
                    if self.kv == "int8":
                        L["k_cache"][page, :, :n] = qk[:, p * PAGE : p * PAGE + n]
                        L["v_cache"][page, :, :n] = qv[:, p * PAGE : p * PAGE + n]
                        L["k_scale"][page, :, :n] = sk[:, p * PAGE : p * PAGE + n]
                        L["v_scale"][page, :, :n] = sv[:, p * PAGE : p * PAGE + n]
                    else:
                        L["k_cache"][page, :, :n] = k[:, p * PAGE : p * PAGE + n]
                        L["v_cache"][page, :, :n] = v[:, p * PAGE : p * PAGE + n]
        self.length[seq] = length
        self.rope_delta[seq] = int(rope_delta)
        self.pending[seq] = 0

    @torch.no_grad()
    def prefill(self, seq: int, embeds: torch.Tensor, pos3: torch.Tensor, rope_delta: int):
        """Run a prompt through the language model into sequence seq's caches (its KV pages, DeltaNet
        state and conv ring), as transformers' prefill does: embeds [n, 2560] BF16 (text embeddings
        with the image features in place), pos3 [n, 3] int32 (t, h, w) positions, rope_delta for the
        positions after the prompt. GEMMs run in cuBLAS on each layer's weight unpacked for the
        call (BF16 or FP8 layers, the same weights as the decode kernels); the DeltaNet layers use vLLM's chunked FLA kernel, attention FA2.
        Returns (final-norm hidden states [n, 2560], the greedy next token)."""
        import torch.nn.functional as F

        from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from . import silu_mul

        dev = self.device
        n = embeds.shape[0]
        pos = torch.arange(n, dtype=torch.int64, device=dev)
        page = self.block_table[seq, pos // PAGE].contiguous()
        in_page = (pos % PAGE).to(torch.int32)
        pos3 = pos3.to(dev, torch.int32).contiguous()
        cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
        npg = math.ceil(n / PAGE)
        pages = self.block_table[seq, :npg].long()

        def lin(L, key, x):
            return L[key].prefill(x)

        res = embeds.to(dev, torch.bfloat16).contiguous()
        prev = None
        for L in self.layers:
            xn, res = rmsnorm_add(res, L["w_in"], x=prev)
            if L["type"] == "linear_attention":
                y = lin(L, "qkvzba", xn)
                raw = y[:, :8192]
                z, b, a = y[:, 8192:12288], y[:, 12288:12320], y[:, 12320:12352]
                conv = F.conv1d(raw.T.float()[None], L["conv_w"].float()[:, None, :], padding=3, groups=8192)
                mixed = F.silu(conv[0, :, :n]).to(torch.bfloat16).T  # [n, 8192], BF16 as vLLM's conv output
                q = mixed[:, :2048].reshape(1, n, 16, 128).repeat_interleave(2, dim=2)
                k = mixed[:, 2048:4096].reshape(1, n, 16, 128).repeat_interleave(2, dim=2)
                v = mixed[:, 4096:].reshape(1, n, 32, 128)
                g = -torch.exp(L["A_log"]) * F.softplus(a.float() + L["dt_bias"].float())
                beta = torch.sigmoid(b.float()).to(torch.bfloat16)
                o, st = chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(), g=g[None],
                                               beta=beta[None], initial_state=None, output_final_state=True,
                                               use_qk_l2norm_in_kernel=True)
                L["state"][seq + 1] = st[0].transpose(-1, -2).to(L["state"].dtype)  # [K, V] -> ours [v][k]
                for j in range(1, 4):
                    if n - j >= 0:
                        L["ring"][seq + 1, (n - j) % self.ring] = raw[n - j]
                of = o[0].to(torch.bfloat16).float()  # vLLM rounds o to BF16 before the norm
                on = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + 1e-6)
                gated = (on * L["norm_w"].float()) * F.silu(z.float().reshape(n, 32, 128))
                y = lin(L, "out", gated.to(torch.bfloat16).reshape(n, 4096))
            else:
                qkv = lin(L, "qkv", xn).contiguous()
                q = qk_norm_rope_kv(qkv, L["q_norm"], L["k_norm"], pos3, L["k_cache"], L["v_cache"], page,
                                    in_page, rope_theta=self.rope_theta, k_scale=L.get("k_scale"),
                                    v_scale=L.get("v_scale"))

                def gather(c, sc):
                    x = c[pages]  # [npg, 4, 64, 256]
                    if sc is not None:
                        x = (x.float() * sc[pages][..., None]).to(torch.bfloat16)
                    return x.permute(0, 2, 1, 3).reshape(npg * PAGE, 4, 256)[:n].contiguous()

                kk, vv = gather(L["k_cache"], L.get("k_scale")), gather(L["v_cache"], L.get("v_scale"))
                att = flash_attn_varlen_func(q=q, k=kk, v=vv, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=n,
                                             max_seqlen_k=n, softmax_scale=1.0 / 16.0, causal=True, fa_version=2)
                gate = qkv[:, :8192].view(n, 16, 512)[..., 256:]
                out = (att.to(torch.bfloat16).float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
                y = lin(L, "o", out.reshape(n, 4096))
            xn, res = rmsnorm_add(res, L["w_post"], x=y)
            prev = lin(L, "down", silu_mul(lin(L, "gate_up", xn).contiguous()))
        hidden, _ = rmsnorm_add(res, self.final_w, x=prev)
        self.length[seq] = n
        self.rope_delta[seq] = int(rope_delta)
        self.pending[seq] = 0
        return hidden, _rows(self.lm_head, hidden[-1:])[0]

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
                            rope_theta=self.rope_theta, k_scale=L.get("k_scale"), v_scale=L.get("v_scale"))
        gate = qkv[:, :8192].view(m, 16, 512)[..., 256:]
        att = attn_decode(q, gate, L["k_cache"], L["v_cache"], bt, ctx, n_q=T, max_ctx=max_ctx,
                          k_scale=L.get("k_scale"), v_scale=L.get("v_scale"), split_keys=self.attn_split_keys)
        return _rows(L["o"], att.reshape(m, 4096))

    @torch.no_grad()
    def _forward(self, tokens: torch.Tensor, seqs: list[int], T: int, commit: bool):
        """The main model on T tokens per sequence at positions length .. length + T - 1 (rows n * T + t).
        Returns the final-norm hidden states [B * T, 2560] and the greedy next tokens [B * T]."""
        dev = self.device
        lens, pos3, page, in_page, bt = self._positions(seqs, [self.length[s] for s in seqs], T)
        slot = torch.tensor([s + 1 for s in seqs], dtype=torch.int32, device=dev)
        max_ctx = max(self.length[s] for s in seqs) + T
        replay = any(self.pending[s] for s in seqs)
        n_rep = torch.tensor([self.pending[s] for s in seqs], dtype=torch.int32, device=dev) if replay or commit else None
        out = self._forward_dev(tokens.to(dev), slot, lens, pos3, page, in_page, bt, T, commit, n_rep, max_ctx)
        for s in seqs:
            self.pending[s] = 0
        return out

    def _forward_dev(self, tokens, slot, lens, pos3, page, in_page, bt, T, commit, n_rep, max_ctx):
        """_forward on device tensors only (no host reads): what a CUDA graph captures."""
        ctx = lens + T
        res = self.embed[tokens]
        prev = None
        for L in self.layers:
            xn, res = rmsnorm_add(res, L["w_in"], x=prev)
            if L["type"] == "linear_attention":
                y4 = _rows(L["qkvzba"], xn)
                hist = L.get("hist")
                o = gdn_decode(y4[:, :8192], y4[:, 8192:12288], y4[:, 12288:12320], y4[:, 12320:12352],
                               L["A_log"], L["dt_bias"],
                               L["conv_w"], L["norm_w"], L["state"], slot, slot, L["ring"], slot, lens, T=T,
                               n_replay=n_rep, hist_in=hist if n_rep is not None else None,
                               hist_out=hist if commit else None, commit=commit)
                y = _rows(L["out"], o)
            else:
                y = self._attention(L, xn, pos3, page, in_page, bt, ctx, T, max_ctx)
            xn, res = rmsnorm_add(res, L["w_post"], x=y)
            prev = _rows(L["down"], _rows(L["gate_up"], xn))
        hidden, _ = rmsnorm_add(res, self.final_w, x=prev)
        return hidden, _rows(self.lm_head, hidden)

    def positions_dev(self, lens, delta, bt, T):
        """Device-side per-row positions (as _positions) from lens [B], rope deltas [B], block table rows."""
        pos = (lens[:, None] + torch.arange(T, dtype=torch.int32, device=lens.device)).reshape(-1)
        rope = pos + delta.repeat_interleave(T)
        pos3 = rope[:, None].expand(-1, 3).contiguous()
        page = bt.repeat_interleave(T, dim=0).gather(1, (pos // PAGE).long()[:, None])[:, 0].contiguous()
        in_page = (pos % PAGE).to(torch.int32)
        return pos3, page, in_page

    def graph_step(self, seqs: list[int], T: int = 1, commit: bool = False) -> "GraphStep":
        """A CUDA graph of one forward step (T rows per sequence) for these sequences."""
        return GraphStep(self, seqs, T, commit)

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
                            page, in_page, rope_theta=self.rope_theta, k_scale=M.get("k_scale"),
                            v_scale=M.get("v_scale"))

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
        z = _rows(M["down"], _rows(M["gate_up"], xn))
        h, _ = rmsnorm_add(res, M["w_norm"], x=z)
        return h, _rows(self.draft_head, h)

    def mtp_forward_dev(self, tokens, hidden, start, delta, bt, T):
        """mtp_forward on device tensors only (start [B] int32 positions, rope deltas, block-table rows)."""
        M = self.mtp
        pos3, page, in_page = self.positions_dev(start, delta, bt, T)
        x = self._mtp_input(tokens, hidden)
        xn, res = rmsnorm_add(x, M["w_in"])
        y = self._attention(M, xn, pos3, page, in_page, bt, start + T, T, self.max_ctx)
        xn, res = rmsnorm_add(res, M["w_post"], x=y)
        z = _rows(M["down"], _rows(M["gate_up"], xn))
        h, _ = rmsnorm_add(res, M["w_norm"], x=z)
        return h, _rows(self.draft_head, h)

    def spec_graph(self, seqs: list[int]) -> "SpecGraph":
        """CUDA graphs of an MTP verify step and of the k draft passes, for these sequences."""
        return SpecGraph(self, seqs)

    def exact_head(self):
        """An LMHeadExact (INT8 pass + BF16 re-check, the same picks as the BF16 head) sharing this
        decoder's BF16 head."""
        from safetensors import safe_open

        from . import LMHeadExact

        f = safe_open(os.path.join(self.ckpt_dir, "model.safetensors"), "pt", device=str(self.device))
        head = LMHeadExact(f.get_tensor("lm_head.weight"), self.lm_head)
        torch.cuda.empty_cache()
        return head

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


class GraphStep:
    """One main-model step for a fixed set of sequences as a CUDA graph. Inputs live in static
    device buffers; positions, pages and the attention split come from the device lengths, so
    one capture serves every step until the set of sequences changes. Attention splits are
    planned for the decoder's full context (max_ctx). Warm-up and capture run on slot 0 (no
    sequence's state) and KV positions at or after each sequence's length (not yet valid)."""

    def __init__(self, dec: Decoder, seqs: list[int], T: int = 1, commit: bool = False):
        dev = dec.device
        b = len(seqs)
        self.dec, self.seqs, self.T, self.commit = dec, list(seqs), T, commit
        self.tok = torch.zeros(b * T, dtype=torch.int64, device=dev)
        self.lens = torch.tensor([dec.length[s] for s in seqs], dtype=torch.int32, device=dev)
        self.delta = torch.tensor([dec.rope_delta[s] for s in seqs], dtype=torch.int32, device=dev)
        self.slot = torch.zeros(b, dtype=torch.int32, device=dev)  # slot 0 while warming up and capturing
        self.bt = dec.block_table[torch.tensor(seqs, device=dev)].contiguous()
        self.n_rep = torch.zeros(b, dtype=torch.int32, device=dev) if (commit or dec.t_max > 1) else None

        def run():
            pos3, page, in_page = dec.positions_dev(self.lens, self.delta, self.bt, T)
            return dec._forward_dev(self.tok, self.slot, self.lens, pos3, page, in_page, self.bt, T, commit,
                                    self.n_rep, dec.max_ctx)

        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                run()
        torch.cuda.current_stream(dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.hidden, self.next = run()
        self.slot.copy_(torch.tensor([s + 1 for s in seqs], dtype=torch.int32, device=dev))

    def __call__(self, tokens: torch.Tensor):
        """Run the step on tokens [B * T] at the sequences' current lengths (and pending replays).
        Returns (hidden [B * T, 2560], next tokens [B * T]); the outputs are the graph's buffers."""
        d = self.dec
        self.tok.copy_(tokens, non_blocking=True)
        self.lens.copy_(torch.tensor([d.length[s] for s in self.seqs], dtype=torch.int32), non_blocking=True)
        if self.n_rep is not None:
            self.n_rep.copy_(torch.tensor([d.pending[s] for s in self.seqs], dtype=torch.int32), non_blocking=True)
        self.graph.replay()
        for s in self.seqs:
            d.pending[s] = 0
        return self.hidden, self.next


class SpecGraph:
    """MTP decoding with two CUDA graphs per step: the verify step (k + 1 rows per sequence, commit
    mode) and the draft passes (the MTP layer on the k + 1 verified rows, then k - 1 single-row
    passes, each with the draft head). Between them the host reads the verify picks (one sync per
    step), accepts, and writes the next step's small inputs. Every sequence stays in the graphs
    until generate() ends (a finished one keeps decoding; its tokens are not kept)."""

    def __init__(self, dec: Decoder, seqs: list[int]):
        if dec.mtp is None:
            raise ValueError("needs Decoder(mtp=True)")
        dev, b, k = dec.device, len(seqs), dec.k
        T = k + 1
        self.dec, self.seqs, self.k = dec, list(seqs), k
        self.verify = GraphStep(dec, seqs, T=T, commit=True)
        self.mt = torch.zeros(b * T, dtype=torch.int64, device=dev)
        self.p0 = torch.tensor([dec.length[s] for s in seqs], dtype=torch.int32, device=dev)
        self.acc = torch.zeros(b, dtype=torch.int32, device=dev)
        self.sel = torch.arange(b, dtype=torch.int64, device=dev) * T
        delta, bt = self.verify.delta, self.verify.bt

        def run():
            hd_all, e_all = dec.mtp_forward_dev(self.mt, self.verify.hidden, self.p0, delta, bt, T)
            hd, e = hd_all.index_select(0, self.sel), e_all.index_select(0, self.sel)
            pos = self.p0 + self.acc + 1
            ds = [e]
            for j in range(1, k):
                hd, e = dec.mtp_forward_dev(e, hd, pos + (j - 1), delta, bt, 1)
                ds.append(e)
            return torch.stack(ds, dim=1)  # [B, k]

        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side), torch.no_grad():  # MTP KV writes at positions >= each length
            for _ in range(2):
                run()
        torch.cuda.current_stream(dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.drafts = run()

    @torch.no_grad()
    def generate(self, first: torch.Tensor, h_last: torch.Tensor, max_new: int, eos=frozenset()):
        """As Decoder.generate_mtp (same tokens), with the steps as graph replays. Returns the token
        lists and the number of verify steps."""
        d, dev, k = self.dec, self.dec.device, self.k
        seqs, b = self.seqs, len(self.seqs)
        out = [[int(t)] for t in first.tolist()]
        live = [out[n][0] not in eos and max_new > 1 for n in range(b)]
        start = [d.length[s] - 1 for s in seqs]  # the first drafts, eagerly (once)
        hd, e = d.mtp_forward(first.to(dev), h_last, seqs, start, 1)
        ds = [e]
        for j in range(1, k):
            hd, e = d.mtp_forward(e, hd, seqs, [p + j for p in start], 1)
            ds.append(e)
        drafts = torch.stack(ds, dim=1)
        last = first.to(dev)
        steps = 0
        while any(live):
            vin = torch.cat([last[:, None], drafts], dim=1)  # [B, k + 1]
            _, nxt = self.verify(vin.reshape(-1))
            steps += 1
            gl, dl = nxt.view(b, k + 1).tolist(), vin.tolist()
            acc = []
            for n in range(b):
                a = 0
                while a < k and dl[n][a + 1] == gl[n][a]:
                    a += 1
                acc.append(a)
                if live[n]:
                    for t in dl[n][1 : a + 1] + [gl[n][a]]:
                        out[n].append(t)
                        if t in eos or len(out[n]) >= max_new:
                            live[n] = False
                            break
            p0 = [d.length[s] for s in seqs]
            d.accept(seqs, [a + 1 for a in acc])
            self.mt.copy_(torch.tensor([[dl[n][i + 1] if i < acc[n] else gl[n][acc[n]] for i in range(k + 1)]
                                        for n in range(b)]).reshape(-1), non_blocking=True)
            self.p0.copy_(torch.tensor(p0, dtype=torch.int32), non_blocking=True)
            self.acc.copy_(torch.tensor(acc, dtype=torch.int32), non_blocking=True)
            self.sel.copy_(torch.tensor([n * (k + 1) + acc[n] for n in range(b)]), non_blocking=True)
            self.graph.replay()
            drafts = self.drafts
            last = torch.tensor([gl[n][acc[n]] for n in range(b)], device=dev)
        return out, steps

