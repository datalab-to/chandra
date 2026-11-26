"""Time the engine's decode step as a CUDA graph on the real checkpoint (weights only matter for
bytes; the caches hold junk at the given context length).

usage: python bench_engine_step.py [--ctx 7000] [--batch 1 8 16 32 64] [--weights bf16 fp8 mxfp4 int4]
       [--lm-head bf16]
"""

import argparse
import glob
import os

import torch

from chandra_cuda.engine import Decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=7000)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ap.add_argument("--weights", nargs="+", default=["bf16", "fp8", "mxfp4", "int4"])
    ap.add_argument("--lm-head", default="bf16")
    ap.add_argument("--state", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--verify", type=int, default=1, help="rows per sequence: 1 = plain decode, k + 1 = an MTP verify step")
    ap.add_argument("--kv", default="bf16", choices=["bf16", "int8"])
    a = ap.parse_args()
    ckpt = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/snapshots/*/"))[0]
    bmax = max(a.batch)
    for wf in a.weights:
        dec = Decoder(ckpt, max_seqs=bmax, max_ctx=a.ctx + 64, weights=wf, lm_head=a.lm_head, kv=a.kv,
                      state_dtype=torch.float16 if a.state == "fp16" else torch.float32,
                      draft_tokens=max(1, a.verify - 1), mtp=a.verify > 1)
        for b in a.batch:
            seqs = list(range(b))
            for s in seqs:
                dec.length[s] = a.ctx
            if b * a.verify > 64 * 4:
                continue
            gs = dec.graph_step(seqs, T=a.verify, commit=a.verify > 1)
            tok = torch.zeros(b * a.verify, dtype=torch.int64, device="cuda")
            for _ in range(3):
                gs(tok)
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(a.reps):
                gs.graph.replay()
            en.record()
            torch.cuda.synchronize()
            ms = st.elapsed_time(en) / a.reps
            what = "step" if a.verify == 1 else f"verify step ({a.verify} rows/seq)"
            print(f"weights {wf:6s} lm_head {a.lm_head:5s} state {a.state} kv {a.kv} B={b:3d} ctx {a.ctx}: "
                  f"{ms:7.2f} ms per {what}, {b / ms * 1e3:7.0f} sequence-steps/s", flush=True)
            del gs
        del dec
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
