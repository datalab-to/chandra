"""transformers' Qwen3.5 vision tower with chandra_cuda's fused rotary.

patch_vision(model) replaces each vision attention module's forward (on that model only) with one that
computes q and k's rotary in one kernel from the qkv projection's output (vit_rope) instead of
transformers' six FP32 elementwise passes, and then calls the same attention function on the same
tensor layouts. The image features are bitwise unchanged.
"""

from __future__ import annotations

import types

import torch

from . import vit_rope


def _attention_forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    n = hidden_states.shape[0]
    qkv = self.qkv(hidden_states)
    cos, sin = position_embeddings
    q, k = vit_rope(qkv, cos, sin)
    v = qkv.reshape(n, 3, self.num_heads, -1)[:, 2]  # transformers' value view
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    attn = mq.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, mq.eager_attention_forward)
    if mq.is_flash_attention_requested(self.config):
        max_seqlen = mq.get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
        out, _ = attn(self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0,
                      cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens, max_length_q=max_seqlen,
                      max_length_k=max_seqlen, is_causal=False, **kwargs)
    else:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        parts = zip(*(torch.split(t, lengths, dim=2) for t in (q, k, v)))
        out = torch.cat([attn(self, a, b, c, attention_mask=None, scaling=self.scaling, dropout=0.0,
                              is_causal=False, **kwargs)[0] for a, b, c in parts], dim=1)
    return self.proj(out.reshape(n, -1).contiguous())


def patch_vision(model) -> int:
    """Patch every Qwen3.5 vision attention module under `model`; returns how many were patched."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    count = 0
    for mod in model.modules():
        if isinstance(mod, mq.Qwen3_5VisionAttention) and mod.head_dim == 64 and mod.num_heads == 16:
            mod.forward = types.MethodType(_attention_forward, mod)
            count += 1
    return count
