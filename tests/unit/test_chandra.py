from __future__ import annotations

import io
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pypdfium2 as pdfium
import pytest
from click.testing import CliRunner
from PIL import Image

import chandra
from chandra.input import (
    SUPPORTED_FILE_EXTENSIONS,
    discover_input_files,
    load_file,
    load_image,
    load_pdf_images,
    parse_range_str,
)
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem, BatchOutputItem, GenerationResult
from chandra.model.util import detect_repeat_token, scale_to_fit
from chandra.output import (
    extract_images,
    get_image_name,
    parse_chunks,
    parse_html,
    parse_layout,
    parse_markdown,
    sanitize_html,
)
from chandra.scripts.cli import main as cli_main, save_merged_output
from chandra.scripts.viewer import ViewerEngine, make_handler
from chandra.scripts.vllm import get_gpu_settings, main as vllm_main
from chandra.skills import get_skill_path, get_skill_text, install_skill
from http.server import ThreadingHTTPServer


def _make_sample_pdf(path: Path, num_pages: int = 3) -> Path:
    pdf = pdfium.PdfDocument.new()
    for _ in range(num_pages):
        pdf.new_page(200, 300)
    pdf.save(str(path))
    pdf.close()
    return path


def _make_sample_image(path: Path, size: tuple[int, int] = (240, 160)) -> Path:
    img = Image.new("RGB", size, color=(240, 245, 250))
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# 1. Input & File Discovery Tests (#30, #66, #98, #111)
# ---------------------------------------------------------------------------


def test_supported_extensions_include_tif_and_avif():
    assert ".tif" in SUPPORTED_FILE_EXTENSIONS
    assert ".tiff" in SUPPORTED_FILE_EXTENSIONS
    assert ".avif" in SUPPORTED_FILE_EXTENSIONS
    assert ".pdf" in SUPPORTED_FILE_EXTENSIONS


def test_parse_range_str():
    assert parse_range_str("1-3,5,8-9") == [0, 1, 2, 4, 7, 8]
    with pytest.raises(ValueError):
        parse_range_str("0")


def test_load_pdf_images_optional_and_empty_page_range(tmp_path: Path):
    pdf_path = _make_sample_pdf(tmp_path / "doc.pdf", num_pages=3)
    # Issue #66: page_range is optional and defaults to None (all pages)
    all_imgs = load_pdf_images(pdf_path)
    assert len(all_imgs) == 3

    # Empty page_range=[] should return [] instead of all pages
    empty_imgs = load_pdf_images(pdf_path, page_range=[])
    assert empty_imgs == []

    # Filtered page_range (including out-of-range page index)
    sub_imgs = load_pdf_images(pdf_path, page_range=[1, 99])
    assert len(sub_imgs) == 1


def test_load_file_pdf_fallback_when_filetype_guess_is_none(tmp_path: Path):
    pdf_path = _make_sample_pdf(tmp_path / "fallback.pdf", num_pages=2)
    with patch("chandra.input.filetype.guess", return_value=None):
        images = load_file(str(pdf_path), {"page_range": "1"})
    assert len(images) == 1


def test_discover_input_files_deduplicates_and_supports_recursive(tmp_path: Path):
    _make_sample_image(tmp_path / "page1.PNG")
    _make_sample_image(tmp_path / "page2.tif")
    (tmp_path / "ignore.txt").write_text("hello")
    subdir = tmp_path / "nested"
    subdir.mkdir()
    _make_sample_image(subdir / "page3.jpg")

    non_rec = discover_input_files(tmp_path, recursive=False)
    assert [p.name for p in non_rec] == ["page1.PNG", "page2.tif"]

    rec = discover_input_files(tmp_path, recursive=True)
    assert [p.name for p in rec] == ["page1.PNG", "page2.tif", "page3.jpg"]


# ---------------------------------------------------------------------------
# 2. Output, Layout, Markdown, and Security Sanitization Tests (#84, #92, #119)
# ---------------------------------------------------------------------------


def test_parse_html_preserves_block_boundaries_issue_119():
    raw = (
        '<div data-bbox="[10, 10, 900, 50]" data-label="Page-header">Header Line</div>'
        '<div data-bbox="[10, 100, 900, 150]" data-label="Section-header"><h2>Title</h2></div>'
        '<div data-bbox="[10, 200, 900, 260]" data-label="Text"><b>Bold start</b> first paragraph</div>'
        '<div data-bbox="[10, 300, 900, 360]" data-label="Text"><i>Italic start</i> second paragraph</div>'
        '<div data-bbox="[10, 920, 900, 970]" data-label="Page-footer">Footer Line</div>'
    )
    # Without headers/footers: two text blocks with inline formatting must stay on separate lines
    md_no_hf = parse_markdown(raw, include_headers_footers=False)
    assert "Header Line" not in md_no_hf
    assert "Footer Line" not in md_no_hf
    assert "**Bold start** first paragraph\n\n*Italic start* second paragraph" in md_no_hf

    # With headers/footers enabled: header and footer must also be separated as distinct blocks
    md_with_hf = parse_markdown(raw, include_headers_footers=True)
    lines = [line.strip() for line in md_with_hf.splitlines() if line.strip()]
    assert lines[0] == "Header Line"
    assert lines[-1] == "Footer Line"


def test_ordered_list_does_not_double_number_issue_84():
    raw = (
        '<div data-bbox="[0, 0, 1000, 500]" data-label="List-Group">'
        "<ol><li>1. First item</li><li>2) Second item</li><li>Third item</li></ol>"
        "</div>"
    )
    md = parse_markdown(raw)
    assert "1. 1." not in md
    assert "2. 2)" not in md
    assert "1. First item" in md
    assert "2. Second item" in md
    assert "3. Third item" in md


def test_sanitize_html_strips_xss_payloads_pr_92():
    dirty = (
        '<div onclick="alert(1)">Hello '
        '<script>evil()</script>'
        '<iframe src="https://evil.example"></iframe>'
        '<a href="javascript:alert(2)" style="color:red">Click</a>'
        '<img src="data:text/html;base64,PHNjcmlwdD4=" onerror=" evil() "/>'
        "</div>"
    )
    clean = sanitize_html(dirty)
    assert "<script" not in clean.lower()
    assert "<iframe" not in clean.lower()
    assert "onclick" not in clean.lower()
    assert "onerror" not in clean.lower()
    assert "javascript:" not in clean.lower()
    assert "data:text/html" not in clean.lower()
    assert "Hello" in clean


def test_parse_layout_and_extract_images():
    img = Image.new("RGB", (400, 200), color=(100, 150, 200))
    raw = (
        '<div data-bbox="[100, 100, 500, 500]" data-label="Image">'
        "<p>Diagram caption</p></div>"
        '<div data-bbox="invalid" data-label="Text"><p>Fallback bbox</p></div>'
    )
    layout = parse_layout(raw, img)
    assert len(layout) == 2
    assert layout[0].label == "Image"
    assert layout[0].bbox == [40, 20, 200, 100]
    assert layout[1].bbox == [0, 0, 400, 200]

    chunks = parse_chunks(raw, img)
    assert len(chunks) == 2
    images = extract_images(raw, chunks, img)
    expected_name = get_image_name(raw, 1)
    assert expected_name in images
    assert images[expected_name].size == (160, 80)


# ---------------------------------------------------------------------------
# 3. Aspect Ratio & Repeat Token Utilities (#95)
# ---------------------------------------------------------------------------


def test_scale_to_fit_preserves_aspect_ratio_issue_95():
    # Extreme aspect ratio image should never collapse to 0 or distort wildly
    img = Image.new("RGB", (1920, 1080))
    scaled = scale_to_fit(img)
    w, h = scaled.size
    assert w % 28 == 0
    assert h % 28 == 0
    orig_ratio = 1920 / 1080
    scaled_ratio = w / h
    assert abs(orig_ratio - scaled_ratio) < 0.08


def test_detect_repeat_token():
    assert detect_repeat_token("") is False
    assert detect_repeat_token("normal output tokens") is False
    assert detect_repeat_token("abc" * 50, cut_from_end=10) is True


# ---------------------------------------------------------------------------
# 4. Model Schema, InferenceManager, vLLM & HF Mocks (#105)
# ---------------------------------------------------------------------------


def test_batch_input_item_default_prompt_type():
    img = Image.new("RGB", (64, 64))
    item = BatchInputItem(image=img)
    assert item.prompt_type == "ocr_layout"


def test_inference_manager_vllm_and_empty_batch():
    from chandra.model.vllm import generate_vllm

    assert generate_vllm([]) == []

    fake_completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='<div data-bbox="[10, 20, 300, 80]" data-label="Text">Hello Chandra</div>'
                )
            )
        ],
        usage=None,  # Verify None usage doesn't crash
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion

    with patch("chandra.model.vllm.OpenAI", return_value=fake_client):
        manager = InferenceManager(method="vllm")
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        results = manager.generate([BatchInputItem(image=img)])

    assert len(results) == 1
    assert "Hello Chandra" in results[0].markdown
    assert results[0].token_count == 0
    assert len(results[0].chunks) == 1


def test_resolve_device_fallback():
    from chandra.model.hf import resolve_device

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    assert resolve_device(fake_torch) == "cpu"


# ---------------------------------------------------------------------------
# 5. CLI, Skill Installation, and vLLM Launcher Tests (#118)
# ---------------------------------------------------------------------------


def test_cli_skill_and_install_skill(tmp_path: Path):
    assert get_skill_path().exists()
    assert "Chandra" in get_skill_text()

    installed = install_skill(tmp_path / ".agents" / "skills")
    assert installed.exists()

    runner = CliRunner()
    res = runner.invoke(cli_main, ["--skill"])
    assert res.exit_code == 0
    assert "chandra" in res.output


def test_cli_stdout_and_save_merged_output(tmp_path: Path):
    img_path = _make_sample_image(tmp_path / "sample.png")
    out_dir = tmp_path / "out"

    fake_result = BatchOutputItem(
        markdown="# Sample Heading\n\nBody text",
        html="<div><h1>Sample Heading</h1><p>Body text</p></div>",
        chunks=[{"bbox": [0, 0, 100, 50], "label": "Section-header", "content": "<h1>Sample Heading</h1>"}],
        raw='<div data-bbox="[0, 0, 500, 500]" data-label="Section-header"><h1>Sample Heading</h1></div>',
        page_box=[0, 0, 240, 160],
        token_count=42,
        images={},
        error=False,
    )

    with patch("chandra.scripts.cli.InferenceManager") as mock_mgr_cls:
        mock_mgr_cls.return_value.generate.return_value = [fake_result]
        runner = CliRunner()

        # 1. --stdout mode without output_dir
        res_stdout = runner.invoke(cli_main, [str(img_path), "--stdout"])
        assert res_stdout.exit_code == 0
        assert "# Sample Heading" in res_stdout.output

        # 2. Directory output mode with chunks sidecar (#118)
        res_dir = runner.invoke(cli_main, [str(img_path), str(out_dir)])
        assert res_dir.exit_code == 0
        assert (out_dir / "sample" / "sample.md").exists()
        assert (out_dir / "sample" / "sample.html").exists()
        assert (out_dir / "sample" / "sample_metadata.json").exists()
        assert (out_dir / "sample" / "sample_chunks.json").exists()


def test_vllm_gpu_presets_and_help():
    tokens_b200, seqs_b200 = get_gpu_settings("b200")
    assert tokens_b200 >= 16384
    assert seqs_b200 >= 128

    tokens_t4, seqs_t4 = get_gpu_settings("t4")
    assert tokens_t4 >= 1024
    assert seqs_t4 >= 8

    with patch.object(sys, "argv", ["chandra_vllm", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            vllm_main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# 6. Interactive Web Viewer (chandra_app) End-to-End HTTP Tests
# ---------------------------------------------------------------------------


def _dispatch_http(
    handler_cls,
    method: str,
    path: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    hdrs = {"Content-Length": str(len(body)), **(headers or {})}
    raw_req = (
        f"{method} {path} HTTP/1.1\r\n"
        + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
        + "\r\n"
    ).encode("iso-8859-1") + body

    wbuf = io.BytesIO()

    class _FakeSocket:
        def makefile(self, mode: str, *args, **kwargs):
            if "r" in mode:
                return io.BytesIO(raw_req)
            return wbuf

        def sendall(self, data: bytes):
            wbuf.write(data)

    handler_cls(_FakeSocket(), ("127.0.0.1", 0), None)
    resp_bytes = wbuf.getvalue()
    head, _, payload = resp_bytes.partition(b"\r\n\r\n")
    status_line = head.splitlines()[0].decode("iso-8859-1")
    status_code = int(status_line.split()[1])
    return status_code, payload


def test_viewer_http_endpoints(tmp_path: Path):
    fake_result = BatchOutputItem(
        markdown="# Viewer Title\n\nExtracted paragraph.",
        html="<div><h1>Viewer Title</h1><p>Extracted paragraph.</p></div>",
        chunks=[
            {
                "bbox": [10, 10, 200, 60],
                "label": "Section-header",
                "content": "<h1>Viewer Title</h1>",
            }
        ],
        raw='<div data-bbox="[40, 60, 800, 350]" data-label="Section-header"><h1>Viewer Title</h1></div>',
        page_box=[0, 0, 240, 160],
        token_count=19,
        images={},
        error=False,
    )
    mock_mgr = MagicMock()
    mock_mgr.generate.return_value = [fake_result]
    engine = ViewerEngine(model_supplier=lambda _method: mock_mgr)
    handler_cls = make_handler(engine)

    # 1. GET /
    status, body = _dispatch_http(handler_cls, "GET", "/")
    assert status == 200
    assert "Chandra Viewer" in body.decode("utf-8")

    # 2. GET /info
    status, body = _dispatch_http(handler_cls, "GET", "/info")
    assert status == 200
    info = json.loads(body.decode("utf-8"))
    assert info["version"] == chandra.__version__
    assert "ocr_layout" in info["prompt_types"]

    # 3. POST /upload with a sample PNG
    buf = io.BytesIO()
    Image.new("RGB", (240, 160), color=(255, 255, 255)).save(buf, format="PNG")
    status, body = _dispatch_http(
        handler_cls,
        "POST",
        "/upload",
        body=buf.getvalue(),
        headers={"X-Filename": "test.png"},
    )
    assert status == 200
    up_data = json.loads(body.decode("utf-8"))
    assert up_data["pages"] == 1
    doc_id = up_data["id"]

    # 4. POST /page to run OCR and inspect layout blocks
    status, body = _dispatch_http(
        handler_cls,
        "POST",
        "/page",
        body=json.dumps(
            {
                "id": doc_id,
                "n": 0,
                "opts": {
                    "method": "vllm",
                    "prompt_type": "ocr_layout",
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert status == 200
    pg_data = json.loads(body.decode("utf-8"))
    assert "# Viewer Title" in pg_data["markdown"]
    assert len(pg_data["blocks"]) == 1
    assert pg_data["blocks"][0]["label"] == "Section-header"

    # 5. Reject unsupported file extension on /upload
    status, body = _dispatch_http(
        handler_cls,
        "POST",
        "/upload",
        body=b"bad",
        headers={"X-Filename": "malware.exe"},
    )
    assert status == 400
    assert "Unsupported file type" in json.loads(body.decode("utf-8"))["error"]


def test_vllm_launcher_cache_dtypes():
    """--state-dtype, --kv-cache-dtype and --attention-backend reach the vLLM command; without them
    nothing is added."""
    import chandra.scripts.vllm as launcher

    def command(argv):
        with patch.object(sys, "argv", ["chandra_vllm", *argv]), patch.object(
            launcher, "detect_container_runtime", return_value="docker"
        ), patch.object(launcher.subprocess, "run") as run:
            launcher.main()
        return run.call_args[0][0]

    cmd = command(["--gpu", "l40s", "--state-dtype", "float16", "--kv-cache-dtype", "int8_per_token_head",
                   "--attention-backend", "TRITON_ATTN"])
    assert cmd[cmd.index("--mamba-ssm-cache-dtype") + 1] == "float16"
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "int8_per_token_head"
    assert cmd[cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    cmd = command(["--gpu", "l40s"])
    assert not {"--mamba-ssm-cache-dtype", "--kv-cache-dtype", "--attention-backend"} & set(cmd)

