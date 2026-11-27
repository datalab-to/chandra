"""Where one page's vision tower (transformers' Qwen3.5 ViT, as chandra_cuda.ocr runs it) spends its GPU
time: the tower's wall time (CUDA events) and torch.profiler's CUDA kernels by total time.

usage: python profile_vision.py SAMPLE_DIR [--page 0] [--attn sdpa|flash_attention_2] [--fused]
"""

import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--fused", action="store_true", help="chandra_cuda.vision.patch_vision (fused rotary)")
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.sample_dir, "manifest.json")))[a.page]
    proc = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    kw = {"attn_implementation": a.attn} if a.attn else {}
    model = AutoModelForImageTextToText.from_pretrained(settings.MODEL_CHECKPOINT, dtype=torch.bfloat16,
                                                        device_map={"": "cpu"}, **kw).eval()
    vis = model.model.visual.to("cuda")
    if a.fused:
        from chandra_cuda.vision import patch_vision

        patch_vision(vis)
    img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
    conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
    inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt")
    pv = inputs["pixel_values"].to("cuda", torch.bfloat16)
    thw = inputs["image_grid_thw"].to("cuda")
    with torch.no_grad():
        for _ in range(2):
            model.model.get_image_features(pv, thw)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            model.model.get_image_features(pv, thw)
        e1.record()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            model.model.get_image_features(pv, thw)
            torch.cuda.synchronize()
    print(f"{pv.shape[0]} patches, grid {thw.tolist()}, attention {vis.config._attn_implementation}{', fused rotary' if a.fused else ''}: "
          f"vision tower {e0.elapsed_time(e1) / 3:.0f} ms")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=80))


if __name__ == "__main__":
    main()
