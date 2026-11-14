"""Self-contained interactive web viewer for Chandra OCR 2 (`chandra_app`).

Zero extra dependencies beyond the base `chandra-ocr` install (`http.server` +
standard library). Supports PDF and image uploads, live switching between
`vllm` and `hf` backends, `ocr_layout` and `ocr` prompts, interactive SVG
layout/label/reading-order overlays, KaTeX math rendering, cropped image
embedding, and Markdown / Blocks / HTML / JSON inspection.
"""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import html as _html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List
import urllib.parse
import uuid

from PIL import Image
from bs4 import BeautifulSoup
import pypdfium2 as pdfium

from chandra import __version__
from chandra.input import (
    SUPPORTED_FILE_EXTENSIONS,
    flatten,
    load_image,
    parse_range_str,
)
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.output import sanitize_html
from chandra.settings import settings

CHANDRA_VERSION = __version__
MAX_SIDE = 1600
MAX_STORED_DOCS = 8
DEFAULT_MAX_UPLOAD_BYTES = int(
    os.environ.get("CHANDRA_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))
)

LOGO = """<svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="6" fill="#18181B"/><path d="M8 8h6.5a5.5 5.5 0 0 1 0 11H8V8Zm2.4 2.2v6.6h4.1a3.3 3.3 0 0 0 0-6.6h-4.1Z" fill="#FAFAFA"/><circle cx="20.2" cy="19.2" r="2.2" fill="#10B981"/></svg>"""


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _encode_preview_jpg(img: Image.Image) -> tuple[str, int, int]:
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", img.width, img.height


def _embed_images_in_html(raw_html: str, images: Dict[str, Image.Image]) -> str:
    safe = sanitize_html(raw_html)
    if not images:
        return safe
    soup = BeautifulSoup(safe, "html.parser")
    for img_name, pil_img in images.items():
        data_url = _pil_to_data_url(pil_img, "PNG")
        for tag in soup.find_all("img", src=img_name):
            tag["src"] = data_url
            tag["style"] = "max-width:100%;height:auto;border-radius:4px;"
    return str(soup)


class ViewerEngine:
    """Thread-safe document store and per-page Chandra OCR runner."""

    def __init__(
        self,
        model_supplier: Callable[[str], InferenceManager] | None = None,
        defaults: Dict[str, Any] | None = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ):
        self._model_supplier = model_supplier
        self._managers: Dict[str, InferenceManager] = {}
        self.docs_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.docs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_upload_bytes = max_upload_bytes
        self.defaults: Dict[str, Any] = {
            "method": "vllm",
            "prompt_type": "ocr_layout",
            "include_images": True,
            "include_headers_footers": False,
            "dpi": settings.IMAGE_DPI,
            "max_pages": 20,
            "vllm_api_base": settings.VLLM_API_BASE,
            "vllm_model_name": settings.VLLM_MODEL_NAME,
            **(defaults or {}),
        }

    def _get_manager(self, method: str) -> InferenceManager:
        method = (method or "vllm").lower()
        if method not in ("vllm", "hf"):
            raise ValueError(f"Unsupported method: {method!r}")
        if self._model_supplier is not None:
            return self._model_supplier(method)
        if method not in self._managers:
            self._managers[method] = InferenceManager(method=method)
        return self._managers[method]

    def info(self) -> Dict[str, Any]:
        return {
            "version": CHANDRA_VERSION,
            "model_checkpoint": settings.MODEL_CHECKPOINT,
            "methods": ["vllm", "hf"],
            "prompt_types": ["ocr_layout", "ocr"],
            "defaults": self.defaults,
        }

    def upload(self, data: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        if not data:
            raise ValueError("Uploaded file is empty.")
        if len(data) > self.max_upload_bytes:
            raise ValueError("Uploaded file exceeds maximum allowed size.")

        safe_name = os.path.basename(filename or "document.pdf")
        ext = os.path.splitext(safe_name)[1].lower() or ".pdf"
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
            )

        fd, tmp_path = tempfile.mkstemp(prefix="chandra_viewer_", suffix=ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            if ext == ".pdf":
                doc = pdfium.PdfDocument(tmp_path)
                try:
                    n_pages = len(doc)
                finally:
                    doc.close()
            else:
                with Image.open(tmp_path) as im:
                    im.verify()
                n_pages = 1
        except Exception as exc:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise ValueError(f"Could not open document: {exc}") from exc

        doc_id = uuid.uuid4().hex[:12]
        with self.docs_lock:
            self.docs[doc_id] = {
                "path": tmp_path,
                "filename": safe_name,
                "ext": ext,
                "pages": max(1, int(n_pages)),
            }
            while len(self.docs) > MAX_STORED_DOCS:
                _, old = self.docs.popitem(last=False)
                old_path = old.get("path")
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass

        return {
            "id": doc_id,
            "filename": safe_name,
            "pages": max(1, int(n_pages)),
        }

    def _render_page(self, doc_entry: Dict[str, Any], page_idx: int, dpi: int) -> Image.Image:
        path = doc_entry["path"]
        ext = doc_entry["ext"]
        if ext == ".pdf":
            pdf = pdfium.PdfDocument(path)
            try:
                if page_idx < 0 or page_idx >= len(pdf):
                    raise ValueError(
                        f"Page index {page_idx} out of range (0..{len(pdf) - 1})."
                    )
                pdf.init_forms()
                page_obj = pdf[page_idx]
                min_page_dim = max(1.0, min(page_obj.get_width(), page_obj.get_height()))
                scale_dpi = max(
                    (settings.MIN_PDF_IMAGE_DIM / min_page_dim) * 72,
                    float(dpi),
                )
                flatten(page_obj)
                page_obj = pdf[page_idx]
                return page_obj.render(scale=scale_dpi / 72.0).to_pil().convert("RGB")
            finally:
                pdf.close()
        return load_image(path)

    def page(
        self, doc_id: str, page_idx: int, opts: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        opts = opts or {}
        with self.docs_lock:
            doc_entry = self.docs.get(doc_id)
        if not doc_entry:
            raise ValueError("Document session not found. Please re-upload the file.")

        total_pages = doc_entry["pages"]
        if page_idx < 0 or page_idx >= total_pages:
            raise ValueError(f"Page {page_idx} out of range (0..{total_pages - 1}).")

        method = str(opts.get("method") or self.defaults["method"]).lower()
        prompt_type = str(opts.get("prompt_type") or self.defaults["prompt_type"]).lower()
        if prompt_type not in ("ocr_layout", "ocr"):
            prompt_type = "ocr_layout"
        include_images = bool(opts.get("include_images", self.defaults["include_images"]))
        include_headers_footers = bool(
            opts.get("include_headers_footers", self.defaults["include_headers_footers"])
        )
        dpi = max(72, min(400, int(opts.get("dpi") or self.defaults["dpi"])))
        vllm_api_base = str(
            opts.get("vllm_api_base") or self.defaults["vllm_api_base"]
        ).strip()
        vllm_model_name = str(
            opts.get("vllm_model_name") or self.defaults["vllm_model_name"]
        ).strip()

        t0 = time.perf_counter()
        pil_img = self._render_page(doc_entry, page_idx, dpi)
        render_ms = round((time.perf_counter() - t0) * 1000, 1)

        t1 = time.perf_counter()
        with self.model_lock:
            manager = self._get_manager(method)
            item = BatchInputItem(image=pil_img, prompt_type=prompt_type)
            gen_kwargs: Dict[str, Any] = {
                "include_images": include_images,
                "include_headers_footers": include_headers_footers,
            }
            if method == "vllm":
                if vllm_api_base:
                    gen_kwargs["vllm_api_base"] = vllm_api_base
                if vllm_model_name:
                    gen_kwargs["vllm_model_name"] = vllm_model_name
            res = manager.generate([item], **gen_kwargs)[0]
        ocr_ms = round((time.perf_counter() - t1) * 1000, 1)

        t2 = time.perf_counter()
        data_url, disp_w, disp_h = _encode_preview_jpg(pil_img)
        sx = disp_w / max(1, pil_img.width)
        sy = disp_h / max(1, pil_img.height)

        blocks_out: List[Dict[str, Any]] = []
        for idx, ch in enumerate(res.chunks):
            bb = ch.get("bbox") or [0, 0, 0, 0]
            scaled_box = [
                round(bb[0] * sx, 1),
                round(bb[1] * sy, 1),
                round(bb[2] * sx, 1),
                round(bb[3] * sy, 1),
            ]
            label = ch.get("label") or "Text"
            content_html = sanitize_html(ch.get("content") or "")
            plain_txt = BeautifulSoup(content_html, "html.parser").get_text(" ", strip=True)
            ignored = (
                label in ("Page-Header", "Page-Footer") and not include_headers_footers
            ) or (label in ("Image", "Figure") and not include_images)
            blocks_out.append(
                {
                    "order": idx + 1,
                    "label": label,
                    "box": scaled_box,
                    "orig_bbox": bb,
                    "text": plain_txt,
                    "html": content_html,
                    "ignored": ignored,
                }
            )

        preview_html = _embed_images_in_html(res.html, res.images)
        post_ms = round((time.perf_counter() - t2) * 1000, 1)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        page_json = {
            "page": page_idx + 1,
            "method": method,
            "prompt_type": prompt_type,
            "page_box": res.page_box,
            "token_count": res.token_count,
            "error": res.error,
            "markdown": res.markdown,
            "html": res.html,
            "chunks": res.chunks,
        }

        return {
            "page": page_idx + 1,
            "w": disp_w,
            "h": disp_h,
            "img": data_url,
            "method": method,
            "prompt_type": prompt_type,
            "token_count": res.token_count,
            "error": res.error,
            "blocks": blocks_out,
            "preview_html": preview_html,
            "markdown": res.markdown,
            "html": res.html,
            "json": page_json,
            "timings": {
                "render_ms": render_ms,
                "ocr_ms": ocr_ms,
                "post_ms": post_ms,
                "total_ms": total_ms,
            },
        }


def make_handler(engine: ViewerEngine):
    class ViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, payload: Any, status: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                body = render_viewer_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/info":
                self._send_json(engine.info())
                return
            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0

            if path == "/upload":
                if length > engine.max_upload_bytes:
                    self._send_json(
                        {"error": "Uploaded file exceeds maximum allowed size."},
                        status=413,
                    )
                    return
                data = self.rfile.read(length) if length > 0 else b""
                raw_name = self.headers.get("X-Filename", "document.pdf")
                filename = urllib.parse.unquote(raw_name)
                try:
                    res = engine.upload(data, filename=filename)
                    self._send_json(res)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                except Exception as exc:
                    self._send_json({"error": f"Upload failed: {exc}"}, status=500)
                return

            if path == "/page":
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    params = json.loads(raw_body.decode("utf-8"))
                    doc_id = str(params.get("id") or "")
                    page_n = int(params.get("n", 0))
                    opts = params.get("opts") or {}
                    if not isinstance(opts, dict):
                        opts = {}
                    res = engine.page(doc_id, page_n, opts)
                    self._send_json(res)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                except Exception as exc:
                    self._send_json({"error": f"OCR failed: {exc}"}, status=500)
                return

            self._send_json({"error": "Not found"}, status=404)

    return ViewerHandler


def render_viewer_html() -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Chandra Viewer · Datalab</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
:root{{
 --bg:#FAF9F6;--bg-2:#F3F1EC;--card:#FFFFFF;--ink:#18181B;--ink-2:#3F3F46;--ink-3:#71717A;--ink-4:#A1A1AA;
 --line:#E4E2DC;--line-2:#D4D1C8;--accent:#10B981;--accent-2:#059669;
 --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
 --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}}
*{{box-sizing:border-box}}html,body{{margin:0;height:100%;background:var(--bg);color:var(--ink);font:13px/1.45 var(--sans);-webkit-font-smoothing:antialiased}}
.app{{display:grid;grid-template-columns:288px 1fr;height:100vh;overflow:hidden}}
.side{{background:var(--bg-2);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow-y:auto}}
.brand{{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line)}}
.brand svg{{width:26px;height:26px;flex-shrink:0}}
.brand .t{{font-weight:650;font-size:14px;letter-spacing:-.01em}}
.brand .sub{{font-size:11px;color:var(--ink-3)}}
.sec{{padding:12px 16px;border-bottom:1px solid var(--line)}}
.sec h4{{margin:0 0 8px;font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3)}}
.drop{{border:1.5px dashed var(--line-2);border-radius:8px;padding:14px 12px;text-align:center;background:var(--card);cursor:pointer;transition:.15s}}
.drop:hover,.drop.drag{{border-color:var(--accent);background:#F0FDF4}}
.drop . title{{font-weight:600;color:var(--ink);margin-bottom:2px}}
.drop .hint{{font-size:11px;color:var(--ink-3)}}
.seg{{display:grid;grid-template-columns:1fr 1fr;gap:4px;background:var(--line);padding:3px;border-radius:7px}}
.seg button{{border:0;background:transparent;padding:6px 8px;border-radius:5px;font:500 12px var(--sans);color:var(--ink-2);cursor:pointer}}
.seg button.on{{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.06)}}
.row{{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:7px 0}}
.row label{{color:var(--ink-2);font-size:12px;display:flex;align-items:center;gap:6px;cursor:pointer}}
.inp{{width:100%;padding:6px 8px;border:1px solid var(--line-2);border-radius:6px;background:var(--card);font:12px var(--mono);color:var(--ink)}}
.inp-sm{{width:86px;padding:4px 7px;border:1px solid var(--line-2);border-radius:5px;background:var(--card);font:12px var(--mono);text-align:right}}
.btn-dl{{width:100%;padding:7px 10px;border:1px solid var(--line-2);border-radius:6px;background:var(--card);font:500 12px var(--sans);color:var(--ink);cursor:pointer;margin-top:4px}}
.btn-dl:hover:not(:disabled){{border-color:var(--ink-3);background:var(--bg)}}
.btn-dl:disabled{{opacity:.45;cursor:not-allowed}}
.main{{display:flex;flex-direction:column;height:100vh;overflow:hidden}}
.top{{display:flex;align-items:center;gap:12px;padding:10px 18px;border-bottom:1px solid var(--line);background:var(--card);flex-shrink:0}}
.top .fn{{font-weight:600;font-size:13.5px}}
.top .meta{{color:var(--ink-3);font-size:12px;font-family:var(--mono)}}
.sp{{flex:1}}
.pbanner{{display:flex;align-items:center;gap:12px;padding:9px 18px;background:#ECFDF5;border-bottom:1px solid #A7F3D0;color:#065F46;font-size:12.5px;flex-shrink:0}}
.pbanner[hidden]{{display:none}}
.pbanner .spin{{width:14px;height:14px;border:2px solid #10B981;border-top-color:transparent;border-radius:50%;animation:sp .75s linear infinite;flex-shrink:0}}
.pbanner .pbar-wrap{{width:140px;height:6px;background:#D1FAE5;border-radius:99px;overflow:hidden}}
.pbanner .pbar-fill{{height:100%;background:#10B981;transition:width .2s}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
.scroll{{flex:1;overflow-y:auto;padding:18px}}
.empty{{max-width:460px;margin:90px auto;text-align:center;color:var(--ink-3)}}
.empty h2{{color:var(--ink);font-size:16px;margin:0 0 6px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:18px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.03)}}
.ch{{display:flex;align-items:center;gap:10px;padding:9px 14px;border-bottom:1px solid var(--line);background:var(--bg)}}
.ch h3{{margin:0;font-size:12.5px;font-weight:650}}
.chip{{font:600 10.5px var(--mono);padding:2px 7px;border-radius:99px;background:#D1FAE5;color:#065F46;text-transform:uppercase}}
.chip.hf{{background:#DBEAFE;color:#1E40AF}}
.chip.err{{background:#FEE2E2;color:#991B1B}}
.body{{display:grid;grid-template-columns:minmax(300px,46%) 1fr;min-height:460px}}
.pageview{{padding:12px;background:#F4F3EE;border-right:1px solid var(--line);display:flex;align-items:flex-start;justify-content:center;overflow:auto}}
svg.pg{{max-width:100%;height:auto;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
rect.bk{{fill-opacity:.08;stroke-width:1.4;cursor:pointer;transition:fill-opacity .1s}}
rect.bk:hover,rect.bk.on{{fill-opacity:.28;stroke-width:2.2}}
rect.bk.ignored{{stroke-dasharray:3 2;opacity:.45}}
text.bl{{fill:#fff;font-family:var(--mono);font-weight:600;pointer-events:none}}
path.ord{{fill:none;stroke:#10B981;stroke-width:1.4;stroke-dasharray:4 3;pointer-events:none}}
circle.on-n{{fill:#10B981}}
text.n{{fill:#fff;font-family:var(--mono);font-weight:700;text-anchor:middle;dominant-baseline:central;pointer-events:none}}
.right{{display:flex;flex-direction:column;min-width:0}}
.tabs{{display:flex;align-items:center;gap:2px;padding:6px 10px;border-bottom:1px solid var(--line);background:var(--bg)}}
.tabs button{{border:0;background:transparent;padding:5px 10px;border-radius:5px;font:500 12px var(--sans);color:var(--ink-3);cursor:pointer}}
.tabs button.on{{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.05)}}
.tabs .copy-btn{{border:1px solid var(--line-2);background:var(--card);color:var(--ink-2);font-size:11px;padding:3px 8px}}
.pane{{padding:14px 16px;overflow:auto;max-height:680px}}
.pane.src{{font:12px/1.55 var(--mono);white-space:pre-wrap;word-break:break-word;color:var(--ink-2)}}
.pane.preview table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}}
.pane.preview th,.pane.preview td{{border:1px solid var(--line-2);padding:5px 8px;text-align:left;vertical-align:top}}
.pane.preview th{{background:var(--bg-2);font-weight:600}}
.pane.preview img{{max-width:100%;height:auto}}
.bk-row{{display:flex;align-items:baseline;gap:8px;padding:6px 8px;border-bottom:1px solid var(--line);font-size:12px;cursor:pointer}}
.bk-row:hover,.bk-row.on{{background:#ECFDF5}}
.bk-row.ignored{{opacity:.5}}
.bk-row .k{{font:600 11px var(--mono);color:var(--ink-3);min-width:22px}}
.bk-row .btag{{font:600 11px var(--mono);padding:1px 6px;border-radius:4px;background:var(--bg-2);white-space:nowrap}}
.bk-row .t{{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}}
</style></head>
<body>
<div class="app">
 <aside class="side">
  <div class="brand">{LOGO}<div><div class="t">Chandra Viewer</div><div class="sub">Datalab · v{CHANDRA_VERSION}</div></div></div>
  <div class="sec">
   <h4>Document</h4>
   <div class="drop" id="drop"><input type="file" id="file" hidden accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.tiff,.tif,.bmp,.avif"/><div class="title">Drop PDF or image</div><div class="hint">or click to browse</div></div>
  </div>
  <div class="sec">
   <h4>Inference Backend</h4>
   <div class="seg" id="methodSeg">
    <button data-v="vllm" class="on" title="Connect to an OpenAI-compatible vLLM server">vLLM Server</button>
    <button data-v="hf" title="Run local HuggingFace Transformers model">HuggingFace</button>
   </div>
   <div style="margin-top:8px" id="vllmBox">
    <div style="font-size:11px;color:var(--ink-3);margin-bottom:3px">vLLM Base URL</div>
    <input class="inp" id="vllmUrl" value="{_html.escape(settings.VLLM_API_BASE)}" placeholder="http://localhost:8000/v1"/>
   </div>
  </div>
  <div class="sec">
   <h4>Prompt Mode</h4>
   <div class="seg" id="promptSeg">
    <button data-v="ocr_layout" class="on">OCR + Layout</button>
    <button data-v="ocr">Plain OCR</button>
   </div>
  </div>
  <div class="sec">
   <h4>Pipeline Options</h4>
   <div class="row"><label><input type="checkbox" id="optImages" checked/>Include images & figures</label></div>
   <div class="row"><label><input type="checkbox" id="optHeaders"/>Include headers & footers</label></div>
   <div class="row"><label>Render DPI</label><input type="number" class="inp-sm" id="optDpi" value="{settings.IMAGE_DPI}" min="72" max="400" step="24"/></div>
   <div class="row"><label>Max pages</label><input type="number" class="inp-sm" id="optMaxPages" value="20" min="1" max="500"/></div>
   <div class="row"><label>Page range</label><input type="text" class="inp-sm" id="optPageRange" placeholder="e.g. 0-3,5"/></div>
  </div>
  <div class="sec">
   <h4>Canvas Overlays</h4>
   <div class="row"><label><input type="checkbox" id="ovBlocks" checked/>Layout bounding boxes</label></div>
   <div class="row"><label><input type="checkbox" id="ovLabels" checked/>Block labels</label></div>
   <div class="row"><label><input type="checkbox" id="ovOrder" checked/>Reading order path</label></div>
  </div>
  <div class="sec">
   <h4>Export</h4>
   <button class="btn-dl" id="dlMd" disabled>Download Markdown (.md)</button>
   <button class="btn-dl" id="dlJson" disabled>Download JSON (.json)</button>
  </div>
 </aside>
 <main class="main">
  <header class="top">
   <span class="fn" id="fn">No document loaded</span>
   <span class="meta" id="docMeta"></span>
   <span class="sp"></span>
   <span class="meta" id="status">Idle</span>
  </header>
  <div class="pbanner" id="pbanner" hidden>
   <div class="spin"></div>
   <strong id="pbannerTitle">Running Chandra OCR…</strong>
   <span id="pbannerSub" style="font-family:var(--mono)"></span>
   <span class="sp"></span>
   <div class="pbar-wrap"><div class="pbar-fill" id="pbannerFill" style="width:0%"></div></div>
  </div>
  <div class="scroll" id="out">
   <div class="empty">
    <h2>Chandra OCR 2 Interactive Viewer</h2>
    <p>Upload any PDF or image to inspect layout bounding boxes, reading order, LaTeX math, HTML tables, and Markdown page by page.</p>
   </div>
  </div>
 </main>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const BLOCK={{
 'Section-Header':'#0D9488','Text':'#2563EB','List-Group':'#16A34A','Table':'#D97706',
 'Figure':'#EA580C','Image':'#B45309','Caption':'#9333EA','Equation-Block':'#7C3AED',
 'Page-Header':'#64748B','Page-Footer':'#64748B','Footnote':'#0891B2','Form':'#E11D48',
 'Code-Block':'#4F46E5','Chemical-Block':'#059669','Diagram':'#D97706','Bibliography':'#475569',
 'Table-Of-Contents':'#0284C7','Complex-Block':'#BE185D'
}};
let curDoc=null,pagesData=[],runSeq=0,activeCtrl=null,timerId=null,runStart=0;

function stopTimer(){{if(timerId){{clearInterval(timerId);timerId=null;}}}}
function abortRun(){{
 runSeq++;
 if(activeCtrl){{try{{activeCtrl.abort();}}catch(_){{}}activeCtrl=null;}}
 stopTimer();
 $('#pbanner').hidden=true;
 const inf=$('#inflight');if(inf)inf.remove();
}}

function parsePageSpec(spec,total,maxP){{
 if(!spec||!spec.trim()){{
  const n=Math.min(total,maxP);
  return Array.from({{length:n}},(_,i)=>i);
 }}
 const set=new Set();
 for(const part of spec.split(',')){{
  const t=part.trim();if(!t)continue;
  if(t.includes('-')){{
   const [a,b]=t.split('-').map(x=>parseInt(x.trim(),10));
   if(!Number.isNaN(a)&&!Number.isNaN(b)){{
    for(let i=Math.max(0,a);i<=Math.min(total-1,b);i++)set.add(i);
   }}
  }}else{{
   const v=parseInt(t,10);
   if(!Number.isNaN(v)&&v>=0&&v<total)set.add(v);
  }}
 }}
 return Array.from(set).sort((a,b)=>a-b).slice(0,maxP);
}}

function getOpts(){{
 const method=$('#methodSeg button.on')?.dataset.v||'vllm';
 const prompt_type=$('#promptSeg button.on')?.dataset.v||'ocr_layout';
 return {{
  method,
  prompt_type,
  vllm_api_base:$('#vllmUrl').value.trim(),
  include_images:$('#optImages').checked,
  include_headers_footers:$('#optHeaders').checked,
  dpi:parseInt($('#optDpi').value,10)||192,
  max_pages:parseInt($('#optMaxPages').value,10)||20,
  page_range:$('#optPageRange').value.trim()
 }};
}}

async function handleFile(file){{
 if(!file)return;
 abortRun();
 $('#fn').textContent=file.name;
 $('#status').textContent='Uploading…';
 const buf=await file.arrayBuffer();
 const r=await fetch('/upload',{{method:'POST',headers:{{'X-Filename':encodeURIComponent(file.name)}},body:buf}});
 const d=await r.json();
 if(!r.ok||d.error){{$('#status').textContent='Upload error: '+(d.error||r.statusText);return;}}
 curDoc=d;
 $('#docMeta').textContent=`${{d.pages}} page${{d.pages>1?'s':''}}`;
 runDoc();
}}

async function runDoc(){{
 if(!curDoc)return;
 abortRun();
 const mySeq=++runSeq;
 const opts=getOpts();
 const indices=parsePageSpec(opts.page_range,curDoc.pages,opts.max_pages);
 pagesData=[];
 $('#dlMd').disabled=true;$('#dlJson').disabled=true;
 $('#out').innerHTML='';
 if(!indices.length){{
  $('#out').innerHTML='<div class="empty"><h2>No pages in range</h2><p>Adjust the page range filter in the sidebar.</p></div>';
  $('#status').textContent='0 pages selected';
  return;
 }}
 runStart=performance.now();
 $('#pbanner').hidden=false;
 $('#pbannerFill').style.width='0%';

 for(let idx=0;idx<indices.length;idx++){{
  if(mySeq!==runSeq)return;
  const pNum=indices[idx];
  const updateClock=()=>{{
   const el=((performance.now()-runStart)/1000).toFixed(1);
   $('#pbannerTitle').textContent=`Running Chandra (${{opts.method.toUpperCase()}}) · Page ${{idx+1}} of ${{indices.length}}`;
   $('#pbannerSub').textContent=`page ${{pNum+1}} · ${{el}}s elapsed`;
   $('#status').textContent=`Processing ${{idx+1}}/${{indices.length}} (${{el}}s)`;
  }};
  updateClock();
  stopTimer();
  timerId=setInterval(updateClock,250);

  let inf=$('#inflight');
  if(!inf){{
   inf=document.createElement('div');
   inf.id='inflight';
   inf.className='card';
   $('#out').appendChild(inf);
  }}
  inf.innerHTML=`<div class="ch"><h3>Page ${{pNum+1}}</h3><span class="chip">${{esc(opts.method)}}</span><span class="meta">Running Chandra OCR…</span></div>`;

  const ctrl=new AbortController();
  activeCtrl=ctrl;
  let r,d;
  try{{
   r=await fetch('/page',{{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{id:curDoc.id,n:pNum,opts}}),
    signal:ctrl.signal
   }});
   d=await r.json();
  }}catch(err){{
   if(ctrl.signal.aborted||mySeq!==runSeq)return;
   d={{error:String(err)}};
  }}
  if(mySeq!==runSeq)return;
  if(!r||!r.ok||d.error){{
   const errCard=document.createElement('div');
   errCard.className='card';
   errCard.innerHTML=`<div class="ch"><h3>Page ${{pNum+1}}</h3><span class="chip err">Error</span><span class="meta">${{esc(d.error||'Failed')}}</span></div>`;
   $('#out').insertBefore(errCard,inf);
  }}else{{
   pagesData.push(d);
   renderPage(d,inf);
   $('#dlMd').disabled=false;$('#dlJson').disabled=false;
  }}
  $('#pbannerFill').style.width=`${{Math.round(((idx+1)/indices.length)*100)}}%`;
 }}
 stopTimer();
 const inf=$('#inflight');if(inf)inf.remove();
 $('#pbanner').hidden=true;
 const tot=((performance.now()-runStart)/1000).toFixed(1);
 $('#status').textContent=`Done · ${{pagesData.length}} page(s) in ${{tot}}s`;
}}

function renderMathIn(el){{
 if(!window.katex)return;
 el.querySelectorAll('math').forEach(m=>{{
  const tex=(m.textContent||'').trim();
  const disp=m.getAttribute('display')==='block';
  const span=document.createElement(disp?'div':'span');
  try{{window.katex.render(tex,span,{{displayMode:disp,throwOnError:false}});m.replaceWith(span);}}catch(_){{}}
 }});
}}

function renderPage(p,beforeEl){{
 const c=document.createElement('div');c.className='card';c.dataset.page=p.page;
 const showB=$('#ovBlocks').checked,showL=$('#ovLabels').checked,showO=$('#ovOrder').checked;
 const u=Math.max(p.w,p.h)/900;
 const blocks=showB?p.blocks.map((b,k)=>{{
  const [x0,y0,x1,y1]=b.box;
  const col=BLOCK[b.label]||'#64748B';
  const cls=b.ignored?'bk ignored':'bk';
  const tag=showL?`<g class="bl-g" pointer-events="none"><rect x="${{x0}}" y="${{Math.max(0,y0-13*u)}}" width="${{(b.label.length*5.8+8)*u}}" height="${{13*u}}" rx="${{2*u}}" fill="${{col}}"/><text class="bl" x="${{x0+4*u}}" y="${{Math.max(9*u,y0-3.5*u)}}" font-size="${{9*u}}">${{esc(b.label)}}</text></g>`:'';
  return `<g data-b="${{k}}"><rect class="${{cls}}" data-b="${{k}}" x="${{x0}}" y="${{y0}}" width="${{Math.max(1,x1-x0)}}" height="${{Math.max(1,y1-y0)}}" stroke="${{col}}" fill="${{col}}"><title>#${{k+1}} ${{esc(b.label)}}</title></rect>${{tag}}</g>`;
 }}).join(''):'';
 const activeBlocks=p.blocks.filter(b=>!b.ignored);
 const cen=showO?activeBlocks.map(b=>[(b.box[0]+b.box[2])/2,(b.box[1]+b.box[3])/2]):[];
 const ord=cen.length>1?`<path class="ord" d="M${{cen.map(pt=>pt.map(v=>v.toFixed(1)).join(',')).join(' L')}}"/>`:'';
 const nums=cen.map((pt,k)=>`<circle class="on-n" cx="${{pt[0]}}" cy="${{pt[1]}}" r="${{7.5*u}}"/><text class="n" x="${{pt[0]}}" y="${{pt[1]}}" font-size="${{8.5*u}}">${{k+1}}</text>`).join('');
 const blockRows=p.blocks.map((b,k)=>{{
  const col=BLOCK[b.label]||'#64748B';
  const snippet=(b.text||'').slice(0,220)||'(image / non-text block)';
  return `<div class="bk-row ${{b.ignored?'ignored':''}}" data-b="${{k}}"><span class="k">${{k+1}}</span><span class="btag" style="border-left:3px solid ${{col}}">${{esc(b.label)}}</span><span class="t">${{esc(snippet)}}</span></div>`;
 }}).join('');
 const tabs=['Preview','Markdown','Blocks','HTML','JSON'];
 const chipCls=p.error?'chip err':(p.method==='hf'?'chip hf':'chip');
 c.innerHTML=`<div class="ch"><h3>Page ${{p.page}}</h3><span class="${{chipCls}}">${{esc(p.method)}} · ${{esc(p.prompt_type)}}</span><span class="meta">${{p.blocks.length}} blocks · ${{p.token_count}} tokens</span><span class="sp"></span><span class="meta">${{p.timings.total_ms}} ms</span></div>
 <div class="body">
  <div class="pageview"><svg class="pg" viewBox="0 0 ${{p.w}} ${{p.h}}"><image href="${{p.img}}" width="${{p.w}}" height="${{p.h}}"/>${{blocks}}${{ord}}${{nums}}</svg></div>
  <div class="right">
   <div class="tabs">${{tabs.map((n,i)=>`<button data-tab="${{n}}" class="${{i?'':'on'}}">${{n}}</button>`).join('')}}<span class="sp"></span><button class="copy-btn">Copy Markdown</button></div>
   <div class="pane preview" data-pane="Preview">${{p.preview_html||'<span class="meta">(Empty page)</span>'}}</div>
   <div class="pane src" data-pane="Markdown" hidden>${{esc(p.markdown||'')}}</div>
   <div class="pane" data-pane="Blocks" hidden>${{blockRows||'<span class="meta">No layout blocks.</span>'}}</div>
   <div class="pane src" data-pane="HTML" hidden>${{esc(p.html||'')}}</div>
   <div class="pane src" data-pane="JSON" hidden>${{esc(JSON.stringify(p.json||{{}},null,2))}}</div>
  </div>
 </div>`;
 renderMathIn(c.querySelector('[data-pane="Preview"]'));
 c.querySelectorAll('.tabs button[data-tab]').forEach(b=>b.onclick=()=>{{
  c.querySelectorAll('.tabs button[data-tab]').forEach(x=>x.classList.toggle('on',x===b));
  c.querySelectorAll('.pane').forEach(x=>x.hidden=x.dataset.pane!==b.dataset.tab);
 }});
 const copyBtn=c.querySelector('.copy-btn');
 if(copyBtn)copyBtn.onclick=()=>{{navigator.clipboard.writeText(p.markdown||'');copyBtn.textContent='Copied!';setTimeout(()=>copyBtn.textContent='Copy Markdown',1200);}};
 const hlBlock=b=>c.querySelectorAll('[data-b]').forEach(el=>el.classList.toggle('on',el.dataset.b==b));
 c.querySelectorAll('.bk-row').forEach(el=>{{el.onmouseenter=()=>hlBlock(el.dataset.b);el.onmouseleave=()=>hlBlock(-1)}});
 c.querySelectorAll('rect.bk').forEach(el=>{{
  el.onmouseenter=()=>{{hlBlock(el.dataset.b);const r=c.querySelector(`.bk-row[data-b="${{el.dataset.b}}"]`);r&&r.scrollIntoView({{block:'nearest'}})}};
  el.onmouseleave=()=>hlBlock(-1);
 }});
 if(beforeEl&&beforeEl.parentNode===$('#out'))$('#out').insertBefore(c,beforeEl);
 else $('#out').appendChild(c);
}}

function rerenderSavedPages(){{
 if(!pagesData.length)return;
 const inf=$('#inflight');
 $('#out').querySelectorAll('.card:not(#inflight)').forEach(el=>el.remove());
 for(const p of pagesData)renderPage(p,inf);
}}

$('#drop').onclick=()=>$('#file').click();
$('#file').onchange=e=>handleFile(e.target.files[0]);
$('#drop').ondragover=e=>{{e.preventDefault();$('#drop').classList.add('drag');}};
$('#drop').ondragleave=()=>$('#drop').classList.remove('drag');
$('#drop').ondrop=e=>{{e.preventDefault();$('#drop').classList.remove('drag');handleFile(e.dataTransfer.files[0]);}};

['#methodSeg','#promptSeg'].forEach(sel=>{{
 document.querySelectorAll(sel+' button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll(sel+' button').forEach(x=>x.classList.toggle('on',x===b));
  $('#vllmBox').style.display=($('#methodSeg button.on')?.dataset.v==='vllm')?'block':'none';
  if(curDoc)runDoc();
 }});
}});
['#optImages','#optHeaders','#optDpi','#optMaxPages','#optPageRange','#vllmUrl'].forEach(sel=>{{
 $(sel).onchange=()=>{{if(curDoc)runDoc();}};
}});
['#ovBlocks','#ovLabels','#ovOrder'].forEach(sel=>{{
 $(sel).onchange=()=>rerenderSavedPages();
}});

$('#dlMd').onclick=()=>{{
 if(!pagesData.length)return;
 const md=pagesData.map(p=>p.markdown||'').join('\\n\\n');
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([md],{{type:'text/markdown'}}));
 a.download=((curDoc?.filename||'document').replace(/\\.[^.]+$/,''))+'.md';
 a.click();
}};
$('#dlJson').onclick=()=>{{
 if(!pagesData.length)return;
 const js=JSON.stringify(pagesData.map(p=>p.json),null,2);
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([js],{{type:'application/json'}}));
 a.download=((curDoc?.filename||'document').replace(/\\.[^.]+$/,''))+'.json';
 a.click();
}};
</script></body></html>"""


def viewer_cli(port: int | None = None, host: str | None = None):
    if port is not None or host is not None:
        resolved_port = port if port is not None else 8000
        resolved_host = host if host is not None else "127.0.0.1"
        engine = ViewerEngine()
        srv = ThreadingHTTPServer((resolved_host, resolved_port), make_handler(engine))
        print(
            f"Chandra Viewer running on http://{resolved_host}:{resolved_port}",
            flush=True,
        )
        srv.serve_forever()
        return

    ap = argparse.ArgumentParser(
        description="Launch the Chandra OCR 2 interactive web viewer."
    )
    ap.add_argument("--port", type=int, default=8000, help="Port to run the viewer on")
    ap.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the viewer to"
    )
    ap.add_argument(
        "--method",
        choices=["vllm", "hf"],
        default="vllm",
        help="Default inference backend (default: vllm)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=settings.IMAGE_DPI,
        help="Default render DPI (default: 192)",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Default maximum pages per document (default: 20)",
    )
    ap.add_argument(
        "--vllm-api-base",
        type=str,
        default=settings.VLLM_API_BASE,
        help=f"Default vLLM API base URL (default: {settings.VLLM_API_BASE})",
    )
    ap.add_argument(
        "--version",
        action="version",
        version=f"chandra-ocr, version {CHANDRA_VERSION}",
    )
    args, _ = ap.parse_known_args(sys.argv[1:])

    engine = ViewerEngine(
        defaults={
            "method": args.method,
            "prompt_type": "ocr_layout",
            "include_images": True,
            "include_headers_footers": False,
            "dpi": args.dpi,
            "max_pages": args.max_pages,
            "vllm_api_base": args.vllm_api_base,
            "vllm_model_name": settings.VLLM_MODEL_NAME,
        }
    )
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"Chandra Viewer running on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Chandra Viewer...")


def main():
    viewer_cli()


if __name__ == "__main__":
    viewer_cli()
