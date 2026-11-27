"""olmOCR-bench through chandra_cuda's engine: every bench PDF's first page (rendered as chandra's
olmocr_bench.py does, at 300 dpi, post-processed the same way) into a candidate directory, timed;
then score it with olmocr.bench.benchmark.

usage: python engine_olmocr.py BENCH_DATA_DIR --candidate NAME [--slots 16] [--weights int8] [--kv int8]
       [--state fp16] [--no-mtp]
"""

import argparse
import glob
import os
import time

from chandra.input import load_pdf_images
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
from chandra.scripts.olmocr_bench import postprocess
from chandra_cuda.ocr import CudaGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--weights", default="int8")
    ap.add_argument("--kv", default="int8")
    ap.add_argument("--state", default="fp16")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    pdf_dir = os.path.join(a.bench_dir, "pdfs")
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    cand = os.path.join(a.bench_dir, a.candidate)

    def out_path(pdf):
        return os.path.join(cand, os.path.splitext(os.path.relpath(pdf, pdf_dir))[0] + "_pg1_repeat1.md")

    todo = [p for p in pdfs if not os.path.exists(out_path(p))]
    gen = CudaGenerator(slots=a.slots, weights=a.weights, kv=a.kv, state=a.state, mtp=not a.no_mtp)

    def loader(pdf):
        def load():
            return BatchInputItem(image=load_pdf_images(pdf, page_range=[0], image_dpi=a.dpi)[0], prompt_type="ocr_layout")
        return load

    done = [0]

    def write(i, r):
        op = out_path(todo[i])
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(op, "w", encoding="utf-8") as f:
            f.write(postprocess(parse_markdown(r.raw) or ""))
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"  {done[0]}/{len(todo)} {time.perf_counter() - t0:.0f} s", flush=True)

    print(f"{len(todo)} of {len(pdfs)} pages to OCR; weights {a.weights}, KV {a.kv}, state {a.state}, "
          f"MTP {not a.no_mtp}, {a.slots} slots", flush=True)
    t0 = time.perf_counter()
    gen.generate([loader(p) for p in todo], on_result=write)
    dt = time.perf_counter() - t0
    print(f"ENGINE_SECONDS {dt:.0f} ({len(todo) / dt:.3f} pages/s)")


if __name__ == "__main__":
    main()
