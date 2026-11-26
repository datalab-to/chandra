"""Speculative decoding with the MTP layer on the real model, against plain decoding.

For each page, transformers runs the prefill (and its own greedy decode, the reference). Its
cache is loaded into chandra_cuda.engine.Decoder, and the MTP layer's KV over the prompt is
built from transformers' final hidden states. Then the engine decodes all pages together
twice: plain greedy steps, and MTP drafts (k per step) verified by the main model. Reports
leading-token agreement (spec vs plain, spec vs transformers), tokens per verify step and
time per token (eager Python, no CUDA graph).

usage: python validate_spec.py SAMPLE_DIR --pages 4 --steps 256 --k 3
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


def lead(a, b):
    n = min(len(a), len(b))
    return next((j for j in range(n) if a[j] != b[j]), n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--draft-heads", default="bf16,fp8,fp8:32768,fp8:65536",
                    help="draft LM heads to try: bf16, fp8, or fp8:N (token ids below N plus the added tokens)")
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
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    dec = Decoder(ckpt, max_seqs=len(pick), max_ctx=8192 + a.steps + 64, mtp=True, draft_tokens=a.k)
    refs, firsts, h_last, saved = [], [], [], []
    with torch.no_grad():
        for i, m in enumerate(pick):
            img = Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")
            conv = [[{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}]}]]
            inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to("cuda")
            ids = inputs["input_ids"]
            n = ids.shape[1]
            ref = model.generate(**inputs, max_new_tokens=a.steps, do_sample=False)[0, n:]
            refs.append(ref.tolist())
            out = model.model(**inputs, use_cache=True)
            hs = out.last_hidden_state[0]  # [n, 2560], after the final norm
            first = int(model.lm_head(hs[-1:]).float().argmax())
            pos, delta = model.model.get_rope_index(ids, inputs["mm_token_type_ids"], inputs["image_grid_thw"],
                                                    None, inputs["attention_mask"])
            pos3 = pos[:, 0].transpose(0, 1).contiguous()  # [n, 3]
            saved.append((out.past_key_values, n, int(delta[0])))
            dec.load_from_hf_cache(i, out.past_key_values, 0, n, int(delta[0]))
            nxt = torch.cat([ids[0, 1:], torch.tensor([first], device="cuda")])
            dec.mtp_prefill(i, nxt[: n - 1], hs[: n - 1], pos3[: n - 1])
            firsts.append(first)
            h_last.append(hs[-1])
            print(f"page {i} {m['png']}: prompt {n} tokens, reference {len(ref)} tokens, "
                  f"first token ours/ref {first}/{refs[-1][0]}", flush=True)
            del out, hs
            torch.cuda.empty_cache()
        seqs = list(range(len(pick)))

        # Plain greedy steps.
        plain = [[f] for f in firsts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps - 1):
            nxt = dec.step(torch.tensor([t[-1] for t in plain], device="cuda"), seqs)
            for t, x in zip(plain, nxt.tolist()):
                t.append(x)
        torch.cuda.synchronize()
        t_plain = time.perf_counter() - t0

        # MTP drafts + verify, from the same prefill state (the MTP KV over the prompt is unchanged),
        # once per draft head; the first is reported in full.
        added = sorted(int(i) for i in tok.added_tokens_decoder)
        runs = []
        for head in a.draft_heads.split(","):
            if head == "bf16":
                dec.set_draft_head("bf16")
            elif head == "fp8":
                dec.set_draft_head("fp8")
            else:
                lim = int(head.split(":")[1])
                rows = torch.tensor(sorted(set(range(lim)) | set(added) | eos), device="cuda")
                dec.set_draft_head("fp8", rows=rows)
            for i, (cache, n, delta) in enumerate(saved):
                dec.load_from_hf_cache(i, cache, 0, n, delta)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, st = dec.generate_mtp(seqs, torch.tensor(firsts, device="cuda"), torch.stack(h_last), a.steps,
                                       eos=eos)
            torch.cuda.synchronize()
            runs.append((head, out, st, time.perf_counter() - t0))
        spec, steps, t_spec = runs[0][1], runs[0][2], runs[0][3]

    report = []
    for i in range(len(pick)):
        p_s, n_s = lead(plain[i][: len(spec[i])], spec[i])
        r_s, n_r = lead(refs[i], spec[i])
        r_p, n_p = lead(refs[i], plain[i])
        report.append(dict(page=pick[i]["png"], spec_vs_plain=[p_s, n_s], spec_vs_ref=[r_s, n_r],
                           plain_vs_ref=[r_p, n_p], spec_tokens=len(spec[i])))
        print(f"page {i}: spec = plain for {p_s}/{n_s} leading tokens; vs transformers: spec {r_s}/{n_r}, "
              f"plain {r_p}/{n_p}" + ("" if p_s == n_s else
                                      f"; after {tok.decode(spec[i][max(0, p_s - 10):p_s])!r}: spec "
                                      f"{tok.decode(spec[i][p_s:p_s + 6])!r}, plain {tok.decode(plain[i][p_s:p_s + 6])!r}"))
    gen = sum(len(s) - 1 for s in spec)
    print(f"MTP k={a.k}: {steps} verify steps for {gen} tokens over {len(pick)} sequences: "
          f"{gen / max(1, steps * len(pick)):.2f} tokens per sequence-step (upper bound {a.k + 1}; "
          f"sequences that end early lower it)")
    tps_plain = (a.steps - 1) * len(pick) / t_plain
    tps_spec = gen / max(1e-9, t_spec)
    print(f"plain: {t_plain / (a.steps - 1) * 1e3:.1f} ms per step, {tps_plain:.0f} tokens/s; spec: "
          f"{t_spec / max(1, steps) * 1e3:.1f} ms per verify step incl. {a.k} draft passes, {tps_spec:.0f} tokens/s "
          f"({tps_spec / tps_plain:.2f}x, eager Python, no CUDA graph)")
    for head, out, st, t in runs:
        g = sum(len(s) - 1 for s in out)
        same = sum(lead(plain[i][: len(out[i])], out[i])[0] == len(out[i]) for i in range(len(pick)))
        print(f"draft head {head}: {g / max(1, st * len(pick)):.2f} tokens per sequence-step, {st} steps, "
              f"{g / t:.0f} tokens/s, {same}/{len(pick)} pages identical to plain")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
