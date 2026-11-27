"""Where a decode step's GPU time goes: 16 real pages prefilled into slots, then MTP steps (verify graph
+ draft graph) under torch.profiler, CUDA kernels by total time, and the mean step time.

usage: python profile_step.py SAMPLE_DIR [--weights int8] [--kv int8] [--slots 16] [--steps 30]
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

from chandra.model.schema import BatchInputItem
from chandra_cuda.ocr import CudaGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--weights", default="int8")
    ap.add_argument("--kv", default="int8")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.sample_dir, "manifest.json")))[: a.slots]
    gen = CudaGenerator(slots=a.slots, weights=a.weights, kv=a.kv, state="fp16", overlap=False)
    d = gen.dec
    with torch.no_grad():
        firsts, hs = [], []
        for s, m in enumerate(man):
            item = BatchInputItem(image=Image.open(os.path.join(a.sample_dir, m["png"])).convert("RGB"))
            f, h = gen._prefill(s, gen._prepare(item))
            firsts.append(f)
            hs.append(h)
        seqs = list(range(len(man)))
        sg = gen._graph_for(seqs)
        last = torch.tensor(firsts, device="cuda")
        drafts = d.first_drafts(last, torch.stack(hs), seqs)
        for _ in range(5):
            _, last, drafts = sg.step(last, drafts)
            drafts = drafts.clone()
        torch.cuda.synchronize()
        t = time.perf_counter()
        n_tok = 0
        for _ in range(a.steps):
            emit, last, drafts = sg.step(last, drafts)
            drafts = drafts.clone()
            n_tok += sum(len(e) for e in emit)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / a.steps
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                _, last, drafts = sg.step(last, drafts)
                drafts = drafts.clone()
            torch.cuda.synchronize()
    ctx = sum(d.length[s] for s in seqs) / len(seqs)
    print(f"{len(seqs)} pages (mean context {ctx:.0f}), weights {a.weights}, KV {a.kv}: step {dt * 1e3:.1f} ms, "
          f"{n_tok / a.steps / len(seqs):.2f} tokens per page per step")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=90))


if __name__ == "__main__":
    main()
