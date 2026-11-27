"""The engine's own prefill on real pages against transformers' (whose cache validate_decode.py
loads): first token, the decode that follows, and prefill time.

transformers computes the inputs (text embeddings with the vision tower's image features in place)
and the (t, h, w) positions; Decoder.prefill runs the language model.

usage: python validate_prefill.py SAMPLE_DIR --pages 4 --steps 128 [--weights bf16|fp8] [--kv bf16|int8]
"""

import argparse
import glob
import json
import os
import time

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
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--weights", default="bf16")
    ap.add_argument("--kv", default="bf16")
    a = ap.parse_args()
    ckpt = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/snapshots/*/"))[0]
    man = json.load(open(os.path.join(a.sample_dir, "manifest.json")))
    pick = [m for m in man[:: max(1, len(man) // (a.pages + 1))] if "old_scans" not in m["png"]][: a.pages]
    proc = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model = AutoModelForImageTextToText.from_pretrained(
        settings.MODEL_CHECKPOINT, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa"
    ).eval()
    img_tok = model.config.image_token_id
    dec = Decoder(ckpt, max_seqs=2, max_ctx=8192 + a.steps + 64, weights=a.weights, kv=a.kv)
    tot = {"hf": 0.0, "ours": 0.0}
    with torch.no_grad():
        for m in pick:
            img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
            conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
            inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            ids = inputs["input_ids"]
            n = ids.shape[1]
            emb = model.model.get_input_embeddings()(ids)[0]
            feats = flat(model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]))
            emb[ids[0] == img_tok] = feats.to(emb.dtype)
            pos, delta = model.model.get_rope_index(ids, inputs["mm_token_type_ids"], inputs["image_grid_thw"],
                                                    None, inputs["attention_mask"])
            pos3 = pos[:, 0].transpose(0, 1).contiguous()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.model.language_model(inputs_embeds=emb[None], position_ids=pos, use_cache=True)
            torch.cuda.synchronize()
            t_hf = time.perf_counter() - t0
            hs = out.last_hidden_state[0]
            first_hf = int(model.lm_head(hs[-1:]).float().argmax())
            dec.load_from_hf_cache(1, out.past_key_values, 0, n, int(delta[0]))
            del out
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hid, first = dec.prefill(0, emb, pos3, int(delta[0]))
            torch.cuda.synchronize()
            t_ours = time.perf_counter() - t0
            rel = ((hid.float() - hs.float()).norm() / hs.float().norm()).item()
            toks = {0: [int(first)], 1: [first_hf]}
            for s in (0, 1):
                nxt = torch.tensor([toks[s][0]], device="cuda")
                for _ in range(a.steps - 1):
                    nxt = dec.step(nxt, [s])
                    toks[s].append(int(nxt))
            k = next((j for j in range(a.steps) if toks[0][j] != toks[1][j]), a.steps)
            tot["hf"] += t_hf
            tot["ours"] += t_ours
            print(f"{m['png']}: {n} tokens; prefill transformers {t_hf * 1e3:.0f} ms, ours {t_ours * 1e3:.0f} ms; "
                  f"final hidden rel L2 {rel:.1e}; first token ours/transformers {int(first)}/{first_hf}; decode "
                  f"after our prefill = after theirs for {k}/{a.steps} tokens", flush=True)
            torch.cuda.empty_cache()
    print(f"prefill total: transformers {tot['hf']:.2f} s, ours {tot['ours']:.2f} s "
          f"(weights {a.weights}, KV {a.kv}; GEMMs on weights unpacked per call)")


if __name__ == "__main__":
    main()
