"""End-to-end check of chandra_cuda's decode kernels on the real model.

For each page: transformers runs the prefill and its own greedy decode (the reference);
its cache after the prefill is loaded into chandra_cuda.engine.Decoder, which then decodes
all pages together in one batch, using only chandra_cuda's kernels. Reports, per page,
how many leading tokens match the reference and the text around the first difference.

usage: python validate_decode.py SAMPLE_DIR --pages 4 --steps 256
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ckpt = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/snapshots/*/"))[0]
    man = json.load(open(os.path.join(a.sample_dir, "manifest.json")))
    pick = [m for m in man[:: max(1, len(man) // (a.pages + 1))] if "old_scans" not in m["png"]][: a.pages]
    proc = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model = AutoModelForImageTextToText.from_pretrained(
        settings.MODEL_CHECKPOINT, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa"
    ).eval()
    tok = proc.tokenizer
    dec = Decoder(ckpt, max_seqs=len(pick), max_ctx=8192 + a.steps)
    refs, starts = [], []
    with torch.no_grad():
        for i, m in enumerate(pick):
            img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
            conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
            inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            n = inputs["input_ids"].shape[1]
            ref = model.generate(**inputs, max_new_tokens=a.steps, do_sample=False)[0, n:]
            refs.append(ref.tolist())
            out = model(**inputs, use_cache=True)
            first = int(out.logits[0, -1].float().argmax())
            dec.load_from_hf_cache(i, out.past_key_values, 0, n, int(model.model.rope_deltas[0]))
            starts.append(first)
            print(f"page {i} {m['png']}: prompt {n} tokens, reference {len(ref)} tokens, "
                  f"first token ours/ref {first}/{refs[-1][0]}", flush=True)
            del out
            torch.cuda.empty_cache()
        seqs = list(range(len(pick)))
        toks = [[s] for s in starts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps - 1):
            nxt = dec.step(torch.tensor([t[-1] for t in toks], device="cuda"), seqs)
            for t, x in zip(toks, nxt.tolist()):
                t.append(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    report = []
    for i, (ours, ref) in enumerate(zip(toks, refs)):
        k = next((j for j in range(min(len(ours), len(ref))) if ours[j] != ref[j]), min(len(ours), len(ref)))
        ctx = tok.decode(ref[max(0, k - 12):k])
        report.append(dict(page=pick[i]["png"], matched=k, compared=min(len(ours), len(ref)),
                           ours=tok.decode(ours[k:k + 8]), ref=tok.decode(ref[k:k + 8]), before=ctx))
        print(f"page {i}: {k} of {min(len(ours), len(ref))} tokens match"
              + ("" if k == min(len(ours), len(ref)) else
                 f"; after {ctx!r}: ours {tok.decode(ours[k:k + 8])!r}, ref {tok.decode(ref[k:k + 8])!r}"))
    print(f"engine decode: {a.steps - 1} steps x {len(pick)} sequences in {dt:.2f} s "
          f"({dt / (a.steps - 1) * 1e3:.1f} ms per step, eager Python, no CUDA graph)")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
