"""End-to-end OCR of sample pages with chandra_cuda's engine (CudaGenerator), timed, against the
markdown of an earlier vLLM run of the same pages (mean character similarity, identical pages).

usage: python ocr_e2e.py SAMPLE_DIR --ref-candidate DIR [--pages 141] [--slots 16] [--weights fp8]
       [--kv int8] [--state fp16] [--no-mtp]
"""

import argparse
import difflib
import json
import os
import time

from PIL import Image

from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
from chandra_cuda.ocr import CudaGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--ref-candidate", required=True, help="bench_data/<candidate> of a vLLM run (markdown per page)")
    ap.add_argument("--pages", type=int, default=141)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--weights", default="fp8")
    ap.add_argument("--kv", default="int8")
    ap.add_argument("--state", default="fp16")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.sample_dir, "manifest.json")))[: a.pages]
    items = [BatchInputItem(image=Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB")) for m in man]
    gen = CudaGenerator(slots=a.slots, weights=a.weights, kv=a.kv, state=a.state, mtp=not a.no_mtp,
                        max_output_tokens=a.max_tokens)
    t0 = time.perf_counter()
    res = gen.generate(items)
    dt = time.perf_counter() - t0
    sims, same, toks = [], 0, 0
    for m, r in zip(man, res):
        md = parse_markdown(r.raw)
        toks += r.token_count
        stem = os.path.splitext(m["pdf"])[0]
        cands = [os.path.join(a.ref_candidate, stem + "_pg1_repeat1.md"), os.path.join(a.ref_candidate, stem + ".md")]
        ref = next((open(c).read() for c in cands if os.path.exists(c)), None)
        if ref is None:
            continue
        sims.append(difflib.SequenceMatcher(None, md, ref, autojunk=False).ratio())
        same += md.strip() == ref.strip()
    print(f"{len(items)} pages in {dt:.1f} s ({len(items) / dt:.3f} pages/s, {toks / dt:.0f} tokens/s incl. prefill "
          f"and vision); weights {a.weights}, KV {a.kv}, state {a.state}, MTP {not a.no_mtp}, {a.slots} slots")
    if sims:
        print(f"vs the reference run: {len(sims)} pages compared, mean character similarity "
              f"{sum(sims) / len(sims):.3f}, {same} identical")


if __name__ == "__main__":
    main()
