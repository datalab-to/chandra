"""
Reproduce Chandra-OCR-2's score on the upstream olmOCR-bench
(github.com/allenai/olmocr), end to end:

  1. download the olmOCR-bench dataset (allenai/olmOCR-bench) from HuggingFace,
  2. OCR every bench page with Chandra (this repo) against a running vLLM server,
  3. apply postprocessing to correct Chandra markdown to more standard markdown.
  4. score with `olmocr.bench.benchmark`.

Reference result: ~85.8% overall.

Usage:
    # 1) in one terminal, serve the model (see OLMOCR_BENCH.md):
    chandra_vllm
    # 2) in another:
    python -m chandra.scripts.olmocr_bench --bench-dir ./olmOCR-bench/bench_data
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import html as _html
import os
import re
import subprocess
import sys
import threading

import unicodeit

from chandra.input import load_pdf_images
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.settings import settings


# --------------------------- output-only postprocess ---------------------------
def _latex_to_unicode(s: str) -> str:
    """LaTeX -> Unicode for table-cell math, via the `unicodeit` library
    (Greek, symbols, ^/_).  We pre-strip \\text/\\mathrm and \\frac (which
    unicodeit leaves as-is), and post-strip any commands it didn't recognize."""
    s = _html.unescape(s)
    s = re.sub(r"\\(?:text|mathrm)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    try:
        s = unicodeit.replace(s)
    except Exception:  # noqa: BLE001 - never let a stray token kill the doc
        pass
    s = (
        re.sub(r"\\[a-zA-Z]+", "", s)
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
    )
    return re.sub(r"\s+", " ", s).strip()


# Unicode sub/superscript maps for <sub>/<sup> digit+operator content.
_SUP = {c: u for c, u in zip("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")}
_SUB = {c: u for c, u in zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")}


def _subsup_to_unicode(s: str) -> str:
    """<sub>2</sub> -> ₂, <sup>2</sup> -> ²"""
    s = re.sub(
        r"<sub>(.*?)</sub>",
        lambda m: "".join(_SUB.get(c, c) for c in m.group(1)),
        s,
        flags=re.S | re.I,
    )
    s = re.sub(
        r"<sup>(.*?)</sup>",
        lambda m: "".join(_SUP.get(c, c) for c in m.group(1)),
        s,
        flags=re.S | re.I,
    )
    return re.sub(r"</?su[bp]>", "", s)  # drop any leftover unmapped tags


def _strip_escapes_outside_math(s: str) -> str:
    """Drop the stray backslash-escapes chandra markdown conversion leaves behind (\\_ \\* \\$ ...)"""
    parts = re.split(r"(\$\$.*?\$\$|\$[^$\n]+\$)", s, flags=re.S)
    for i in range(0, len(parts), 2):  # even indices are non-math
        parts[i] = re.sub(r"\\([_*$%&#.+()\[\]!>~^{}])", r"\1", parts[i])
    return "".join(parts)


def postprocess(md: str) -> str:
    """Reformat Chandra markdown for olmoCR scoring (output-only).

    * table-cell <math>..</math> -> Unicode (chandra has latex in tables)
    * HTML-unescape inside prose math spans (\\&amp;c. -> \\&c.)
    * <sub>/<sup> -> Unicode (chandra has sup tags in output)
    * drop stray escape backslashes outside math spans (introduced by Chandra markdown renderer)
    * drop synthesized figure captions (chandra-specific synth captions)
    """
    # only <math> tags Chandra leaves are inside tables -> convert to Unicode
    s = re.sub(
        r"<math\b[^>]*>(.*?)</math>",
        lambda m: _latex_to_unicode(m.group(1)),
        md,
        flags=re.S | re.I,
    )
    s = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: "$$" + _html.unescape(m.group(1)) + "$$",
        s,
        flags=re.S,
    )
    s = re.sub(r"\$([^$\n]+)\$", lambda m: "$" + _html.unescape(m.group(1)) + "$", s)
    s = _subsup_to_unicode(s)
    s = _strip_escapes_outside_math(s)
    # Drop image markdown (Chandra emits synthesized figure descriptions as alt
    # text)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    return s


def download_bench(bench_dir: str) -> str:
    """Ensure the olmOCR-bench data is present; return the bench_data dir."""
    if os.path.isdir(os.path.join(bench_dir, "pdfs")):
        return bench_dir
    from huggingface_hub import snapshot_download

    print("Downloading allenai/olmOCR-bench from HuggingFace ...")
    local = snapshot_download(
        repo_id="allenai/olmOCR-bench",
        repo_type="dataset",
        allow_patterns=["bench_data/**"],
    )
    src = os.path.join(local, "bench_data")
    os.makedirs(os.path.dirname(bench_dir) or ".", exist_ok=True)
    import shutil

    shutil.copytree(src, bench_dir, dirs_exist_ok=True)
    return bench_dir


def run_inference(
    bench_dir, candidate, image_dpi, workers, vllm_api_base, apply_postprocess=True
):
    pdf_dir = os.path.join(bench_dir, "pdfs")
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    cand_dir = os.path.join(bench_dir, candidate)
    print(f"OCR'ing {len(pdfs)} bench pages with Chandra (dpi={image_dpi}) ...")
    mgr = InferenceManager(method="vllm")

    def out_path(pdf):
        rel = os.path.relpath(pdf, pdf_dir)
        return os.path.join(cand_dir, f"{os.path.splitext(rel)[0]}_pg1_repeat1.md")

    todo = [pdf for pdf in pdfs if not os.path.exists(out_path(pdf))]  # resumable
    lock = threading.Lock()
    done = len(pdfs) - len(todo)
    # Keep `workers` pages in flight: a new page is sent as soon as one finishes,
    # instead of waiting for a whole batch. With batches, the server ran each
    # batch's slowest page (and its retries) nearly alone before the next batch.
    slots = threading.BoundedSemaphore(workers)

    def ocr(pdf, image):
        nonlocal done
        try:
            item = BatchInputItem(image=image, prompt_type="ocr_layout")
            out = mgr.generate([item], vllm_api_base=vllm_api_base, max_workers=1)[0]
            op = out_path(pdf)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            md = out.markdown or ""
            with open(op, "w", encoding="utf-8") as f:
                f.write(postprocess(md) if apply_postprocess else md)
            with lock:
                done += 1
                print(f"  {done}/{len(pdfs)}", flush=True)
        finally:
            slots.release()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for pdf in todo:
            # pdfium is not thread-safe, so pages are rendered here, one at a
            # time, while up to `workers` earlier pages are being OCR'd.
            try:
                imgs = load_pdf_images(pdf, page_range=[0], image_dpi=image_dpi)
            except Exception as e:
                print(f"  render failed {pdf}: {e}", flush=True)
                continue
            if not imgs:
                continue
            slots.acquire()
            futures.append(pool.submit(ocr, pdf, imgs[0]))
        for fut in futures:
            fut.result()
    return cand_dir


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark Chandra on upstream olmOCR-bench."
    )
    ap.add_argument(
        "--bench-dir",
        default="./olmOCR-bench/bench_data",
        help="Path to olmOCR-bench bench_data (downloaded if missing).",
    )
    ap.add_argument("--candidate", default="chandra", help="Candidate subdir name.")
    ap.add_argument(
        "--image-dpi", type=int, default=300, help="Render DPI (300 recommended)."
    )
    ap.add_argument("--workers", type=int, default=32, help="Concurrent vLLM requests.")
    ap.add_argument(
        "--vllm-api-base",
        default=settings.VLLM_API_BASE,
        help="vLLM OpenAI base URL (default from settings).",
    )
    ap.add_argument(
        "--skip-inference",
        action="store_true",
        help="Reuse existing candidate markdown; only score.",
    )
    ap.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Only produce candidate markdown; don't run olmocr.bench.",
    )
    ap.add_argument(
        "--stock",
        action="store_true",
        help="Stock baseline: render at DPI-192 and apply NO output postprocess",
    )
    args = ap.parse_args()

    bench_dir = download_bench(args.bench_dir)
    image_dpi = 192 if args.stock else args.image_dpi
    if not args.skip_inference:
        run_inference(
            bench_dir,
            args.candidate,
            image_dpi,
            args.workers,
            args.vllm_api_base,
            apply_postprocess=not args.stock,
        )

    if args.skip_scoring:
        print(f"\nCandidate written to {os.path.join(bench_dir, args.candidate)}")
        print(
            "Score it with:  python -m olmocr.bench.benchmark "
            f"--dir {bench_dir} --candidate {args.candidate}"
        )
        return

    cmd = [
        sys.executable,
        "-m",
        "olmocr.bench.benchmark",
        "--dir",
        bench_dir,
        "--candidate",
        args.candidate,
    ]
    print("\nScoring with upstream olmocr.bench:\n  " + " ".join(cmd) + "\n")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "olmocr not installed. Install with: pip install olmocr[bench] && playwright install chromium"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
