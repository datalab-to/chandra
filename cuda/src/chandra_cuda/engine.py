"""A minimal decode engine for Chandra 2 built only from chandra_cuda's kernels.

It loads the language model's weights from the checkpoint into packed, merged
layouts (qkvz, ba, qkv with the output gate, gate_up), owns a paged KV cache for
the 8 full-attention layers and state slots for the 24 Gated DeltaNet layers, and
runs greedy decode steps for a batch of sequences. Prefill is not here yet: a
sequence's state is loaded from transformers' cache after its prefill
(`load_from_hf_cache`), which is how the kernels are validated end to end.
"""

from __future__ import annotations

import json
import math
import os

import torch

from . import DecodeLinear, LMHead, attn_decode, gdn_decode, qk_norm_rope_kv, rmsnorm_add, silu_mul

PAGE = 64


class Decoder:
    def __init__(self, ckpt_dir: str, device: str = "cuda", max_seqs: int = 8, max_ctx: int = 18432,
                 state_dtype: torch.dtype = torch.float32):
        from safetensors import safe_open

        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))["text_config"]
        self.layer_types = cfg["layer_types"]
        self.rope_theta = float(cfg["rope_parameters"]["rope_theta"])
        self.device = torch.device(device)
        self.max_seqs, self.max_ctx = max_seqs, max_ctx
        f = safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt", device=str(self.device))
        g = f.get_tensor
        P = "model.language_model."
        self.embed = g(P + "embed_tokens.weight")
        self.final_w = g(P + "norm.weight")
        self.lm_head = LMHead(g("lm_head.weight"))
        self.layers = []
        for i, t in enumerate(self.layer_types):
            p = f"{P}layers.{i}."
            L = dict(
                type=t,
                w_in=g(p + "input_layernorm.weight"),
                w_post=g(p + "post_attention_layernorm.weight"),
                gate_up=DecodeLinear(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])),
                down=DecodeLinear(g(p + "mlp.down_proj.weight")),
            )
            if t == "linear_attention":
                a = p + "linear_attn."
                L["qkvz"] = DecodeLinear(torch.cat([g(a + "in_proj_qkv.weight"), g(a + "in_proj_z.weight")]))
                L["ba"] = DecodeLinear(torch.cat([g(a + "in_proj_b.weight"), g(a + "in_proj_a.weight")]))
                L["out"] = DecodeLinear(g(a + "out_proj.weight"))
                L["conv_w"] = g(a + "conv1d.weight").reshape(8192, 4).contiguous()
                L["A_log"] = g(a + "A_log").float()
                L["dt_bias"] = g(a + "dt_bias")
                L["norm_w"] = g(a + "norm.weight")
            else:
                a = p + "self_attn."
                L["qkv"] = DecodeLinear(torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"),
                                                   g(a + "v_proj.weight")]))
                L["o"] = DecodeLinear(g(a + "o_proj.weight"))
                L["q_norm"], L["k_norm"] = g(a + "q_norm.weight"), g(a + "k_norm.weight")
            self.layers.append(L)
        torch.cuda.empty_cache()

        # Caches. Sequence s owns pages [s * pps, (s + 1) * pps) and state / ring slot s + 1.
        dev = self.device
        self.pps = math.ceil(max_ctx / PAGE)
        pages = max_seqs * self.pps
        self.block_table = torch.arange(pages, dtype=torch.int32, device=dev).reshape(max_seqs, self.pps)
        for L in self.layers:
            if L["type"] == "linear_attention":
                L["state"] = torch.zeros(max_seqs + 1, 32, 128, 128, dtype=state_dtype, device=dev)
                L["ring"] = torch.zeros(max_seqs + 1, 4, 8192, dtype=torch.bfloat16, device=dev)
            else:
                L["k_cache"] = torch.zeros(pages, 4, PAGE, 256, dtype=torch.bfloat16, device=dev)
                L["v_cache"] = torch.zeros_like(L["k_cache"])
        self.length = [0] * max_seqs  # tokens in the cache per sequence
        self.rope_delta = [0] * max_seqs

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
                        L["ring"][seq + 1, pos % 4] = conv[:, w - j].to(torch.bfloat16)
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

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, seqs: list[int]) -> torch.Tensor:
        """One greedy decode step: tokens [B] (the last token of each sequence) -> next tokens [B]."""
        dev = self.device
        b = len(seqs)
        lens = torch.tensor([self.length[s] for s in seqs], dtype=torch.int32, device=dev)
        rope = torch.tensor([self.length[s] + self.rope_delta[s] for s in seqs], dtype=torch.int32, device=dev)
        pos3 = rope[:, None].expand(b, 3).contiguous()
        seq_t = torch.tensor(seqs, dtype=torch.int64, device=dev)
        slot = (seq_t + 1).to(torch.int32)
        page = self.block_table[seq_t, (lens // PAGE).long()].contiguous()
        in_page = (lens % PAGE).to(torch.int32)
        bt = self.block_table[seq_t].contiguous()
        ctx = lens + 1
        max_ctx = max(self.length[s] for s in seqs) + 1

        res = self.embed[tokens.to(dev)]
        prev = None
        for L in self.layers:
            xn, res = rmsnorm_add(res, L["w_in"], x=prev)
            if L["type"] == "linear_attention":
                qkvz = L["qkvz"](xn)
                ba = L["ba"](xn)
                o = gdn_decode(qkvz[:, :8192], qkvz[:, 8192:], ba[:, :32], ba[:, 32:], L["A_log"], L["dt_bias"],
                               L["conv_w"], L["norm_w"], L["state"], slot, slot, L["ring"], slot, lens)
                y = L["out"](o)
            else:
                qkv = L["qkv"](xn)
                q = qk_norm_rope_kv(qkv, L["q_norm"], L["k_norm"], pos3, L["k_cache"], L["v_cache"], page, in_page,
                                    rope_theta=self.rope_theta)
                gate = qkv[:, :8192].view(b, 16, 512)[..., 256:]
                att = attn_decode(q, gate, L["k_cache"], L["v_cache"], bt, ctx, max_ctx=max_ctx)
                y = L["o"](att.reshape(b, 4096))
            xn, res = rmsnorm_add(res, L["w_post"], x=y)
            prev = L["down"](silu_mul(L["gate_up"](xn)))
        xn, _ = rmsnorm_add(res, self.final_w, x=prev)
        for s in seqs:
            self.length[s] += 1
        return self.lm_head(xn)
