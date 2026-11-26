"""chandra_cuda.vit_attn inside transformers' vision tower on real pages.

The vision attention's forward is replaced by one that calls vit_attn on the rotary-embedded
q, k and v ([N, 16, 64]) with the images' cu_seqlens. Per page: time of the whole vision tower
with SDPA (transformers' default; FlashAttention-2 kernels on sm_80+) and with vit_attn v1 (BF16)
and v3 (INT8 QK^T), the relative L2 difference of the image embeddings, and for v3 the greedy
tokens of a short decode against SDPA's.

usage: python vit_in_hf.py SAMPLE_DIR --pages 4 [--decode 128]
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

import chandra_cuda
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings

ORIG = mq.Qwen3_5VisionAttention.forward
VERSION = {"v": 1}


def patched(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    n = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = mq.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cu = cu_seqlens.to(torch.int32)
    mx = int((cu[1:] - cu[:-1]).max())
    out = chandra_cuda.vit_attn(q, k, v, cu, mx, version=VERSION["v"])
    return self.proj(out.reshape(n, -1))


def vision(model, inputs):
    return model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / reps * 1e3


def flat(x):
    if isinstance(x, (list, tuple)):
        return torch.cat([flat(t) for t in x])
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return flat(x.pooler_output)
    if hasattr(x, "last_hidden_state"):
        return flat(x.last_hidden_state)
    return x.reshape(-1, x.shape[-1]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--decode", type=int, default=128)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.sample_dir, "manifest.json")))
    pick = [m for m in man[:: max(1, len(man) // (a.pages + 1))] if "old_scans" not in m["png"]][: a.pages]
    proc = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model = AutoModelForImageTextToText.from_pretrained(
        settings.MODEL_CHECKPOINT, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa"
    ).eval()
    tot = {"sdpa": 0.0, "v1": 0.0, "v3": 0.0}
    with torch.no_grad():
        for m in pick:
            img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
            conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
            inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            patches = int(inputs["image_grid_thw"].prod(dim=-1).sum())
            mq.Qwen3_5VisionAttention.forward = ORIG
            ref, t_ref = timed(lambda: vision(model, inputs))
            ref = flat(ref)
            mq.Qwen3_5VisionAttention.forward = patched
            res = {}
            for ver in (1, 3):
                VERSION["v"] = ver
                out, t = timed(lambda: vision(model, inputs))
                res[ver] = ((flat(out) - ref).norm() / ref.norm()).item(), t
            tot["sdpa"] += t_ref
            tot["v1"] += res[1][1]
            tot["v3"] += res[3][1]
            line = (f"{m['png']}: {patches} patches; vision tower SDPA {t_ref:.0f} ms, vit_attn v1 {res[1][1]:.0f} ms "
                    f"(rel L2 {res[1][0]:.1e}), v3 INT8 QK {res[3][1]:.0f} ms (rel L2 {res[3][0]:.1e})")
            if a.decode:
                n = inputs["input_ids"].shape[1]
                mq.Qwen3_5VisionAttention.forward = ORIG
                g0 = model.generate(**inputs, max_new_tokens=a.decode, do_sample=False)[0, n:].tolist()
                mq.Qwen3_5VisionAttention.forward = patched
                VERSION["v"] = 3
                g3 = model.generate(**inputs, max_new_tokens=a.decode, do_sample=False)[0, n:].tolist()
                k = next((j for j in range(min(len(g0), len(g3))) if g0[j] != g3[j]), min(len(g0), len(g3)))
                line += f"; greedy tokens with v3 = SDPA's for {k}/{min(len(g0), len(g3))}"
            mq.Qwen3_5VisionAttention.forward = ORIG
            print(line, flush=True)
    print(f"total vision tower: SDPA {tot['sdpa']:.0f} ms, v1 {tot['v1']:.0f} ms ({tot['sdpa'] / tot['v1']:.2f}x), "
          f"v3 {tot['v3']:.0f} ms ({tot['sdpa'] / tot['v3']:.2f}x)")


if __name__ == "__main__":
    main()
