"""Decode tokens/s on real pages with CUDA graphs: plain steps against the MTP loop (SpecGraph), in
one engine configuration, and whether the two give the same tokens.

transformers runs each page's prefill; the engine loads its cache (and builds the MTP layer's KV
over the prompt), then decodes all pages together.

usage: python bench_spec_graph.py SAMPLE_DIR --pages 8 --steps 256 [--weights mxfp4] [--kv int8]
       [--state fp16] [--mtp-weights mxfp4] [--draft-vocab 32768]
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
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--weights", default="mxfp4")
    ap.add_argument("--kv", default="int8")
    ap.add_argument("--state", default="fp16")
    ap.add_argument("--mtp-weights", default="mxfp4")
    ap.add_argument("--draft-vocab", type=int, default=32768, help="0: the full vocabulary")
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
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    dec = Decoder(ckpt, max_seqs=len(pick), max_ctx=8192 + 4 * a.steps + 64, mtp=True, draft_tokens=3,
                  weights=a.weights, kv=a.kv, state_dtype=torch.float16 if a.state == "fp16" else torch.float32,
                  mtp_weights=a.mtp_weights)
    if a.draft_vocab:
        added = sorted(int(i) for i in tok.added_tokens_decoder)
        rows = torch.tensor(sorted(set(range(a.draft_vocab)) | set(added) | eos), device="cuda")
        dec.set_draft_head("fp8", rows=rows)
    else:
        dec.set_draft_head("fp8")
    saved, firsts, h_last = [], [], []
    with torch.no_grad():
        for i, m in enumerate(pick):
            img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
            conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
            inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            ids = inputs["input_ids"]
            n = ids.shape[1]
            out = model.model(**inputs, use_cache=True)
            hs = out.last_hidden_state[0]
            first = int(model.lm_head(hs[-1:]).float().argmax())
            pos, delta = model.model.get_rope_index(ids, inputs["mm_token_type_ids"], inputs["image_grid_thw"],
                                                    None, inputs["attention_mask"])
            pos3 = pos[:, 0].transpose(0, 1).contiguous()
            saved.append((out.past_key_values, n, int(delta[0])))
            dec.load_from_hf_cache(i, out.past_key_values, 0, n, int(delta[0]))
            nxt = torch.cat([ids[0, 1:], torch.tensor([first], device="cuda")])
            dec.mtp_prefill(i, nxt[: n - 1], hs[: n - 1], pos3[: n - 1])
            firsts.append(first)
            h_last.append(hs[-1].clone())
            del out, hs
            torch.cuda.empty_cache()
        seqs = list(range(len(pick)))
        b = len(seqs)

        # Plain steps as a graph.
        gs = dec.graph_step(seqs)
        plain = [[f] for f in firsts]
        tokv = torch.tensor(firsts, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps - 1):
            _, nxt = gs(tokv)
            for s in seqs:
                dec.length[s] += 1
            tokv = nxt.clone()
            for t, x in zip(plain, tokv.tolist()):
                t.append(x)
        torch.cuda.synchronize()
        t_plain = time.perf_counter() - t0
        del gs

        for i, (cache, n, delta) in enumerate(saved):
            dec.load_from_hf_cache(i, cache, 0, n, delta)
        sg = dec.spec_graph(seqs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec, steps = sg.generate(torch.tensor(firsts, device="cuda"), torch.stack(h_last), a.steps, eos=eos)
        torch.cuda.synchronize()
        t_spec = time.perf_counter() - t0

    # Plain decoding does not stop at EOS; compare up to each MTP output's length.
    same = sum(plain[n][: len(spec[n])] == spec[n] for n in range(b))
    gen_plain = (a.steps - 1) * b
    gen_spec = sum(len(s) - 1 for s in spec)
    print(f"{b} pages, weights {a.weights}, KV {a.kv}, state {a.state}, MTP layer {a.mtp_weights}, draft "
          f"vocabulary {a.draft_vocab or 'full'}:")
    print(f"  plain graph steps: {t_plain / (a.steps - 1) * 1e3:.2f} ms per step, {gen_plain / t_plain:.0f} tokens/s")
    print(f"  MTP-3 graphs: {steps} verify steps, {gen_spec / max(1, steps * b):.2f} tokens per sequence-step, "
          f"{t_spec / max(1, steps) * 1e3:.2f} ms per step, {gen_spec / t_spec:.0f} tokens/s "
          f"({(gen_spec / t_spec) / (gen_plain / t_plain):.2f}x); {same}/{b} pages identical to plain")


if __name__ == "__main__":
    main()
