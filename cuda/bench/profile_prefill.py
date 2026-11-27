"""Where one page's engine prefill spends its GPU time (torch.profiler, CUDA kernels by total time).

usage: python profile_prefill.py SAMPLE_DIR [--weights bf16|fp8|int8] [--kv bf16|int8]
"""

import argparse
import glob
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings
from chandra_cuda.engine import Decoder


def flat(x):
    if isinstance(x, (list, tuple)):
        return torch.cat([flat(t) for t in x])
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return flat(x.pooler_output)
    if hasattr(x, "last_hidden_state"):
        return flat(x.last_hidden_state)
    return x.reshape(-1, x.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--weights", default="bf16")
    ap.add_argument("--kv", default="int8")
    a = ap.parse_args()
    ckpt = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/snapshots/*/"))[0]
    m = json.load(open(os.path.join(a.sample_dir, "manifest.json")))[0]
    proc = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model = AutoModelForImageTextToText.from_pretrained(settings.MODEL_CHECKPOINT, dtype=torch.bfloat16,
                                                        device_map={"": "cpu"}).eval()
    model.model.visual.to("cuda")
    dec = Decoder(ckpt, max_seqs=1, max_ctx=8192 + 64, weights=a.weights, kv=a.kv)
    img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
    conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
    inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt")
    ids = inputs["input_ids"].to("cuda")
    with torch.no_grad():
        feats = flat(model.model.get_image_features(inputs["pixel_values"].to("cuda", torch.bfloat16),
                                                    inputs["image_grid_thw"].to("cuda")))
        emb = dec.embed[ids[0]].clone()
        emb[ids[0] == model.config.image_token_id] = feats.to(emb.dtype)
        pos, delta = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                                inputs["image_grid_thw"], None, inputs["attention_mask"])
        pos3 = pos[:, 0].transpose(0, 1).contiguous().to("cuda")
        for _ in range(2):  # warm-up (Triton compiles)
            dec.prefill(0, emb, pos3, int(delta[0]))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            dec.prefill(0, emb, pos3, int(delta[0]))
            torch.cuda.synchronize()
    print(f"{ids.shape[1]} tokens, weights {a.weights}, KV {a.kv}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))


if __name__ == "__main__":
    main()
