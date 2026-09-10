"""Cover the chunk sidecar and pin the existing outputs.

These tests assert that ``save_merged_output`` writes the per-page layout chunks
to ``<stem>_chunks.json`` and that the markdown, HTML and ``<stem>_metadata.json``
outputs are exactly what they were before the sidecar was added.
"""

import json
from pathlib import Path

import pytest

from chandra.model.schema import BatchOutputItem
from chandra.scripts.cli import save_merged_output

PAGE_BOX = [0, 0, 1224, 1584]

# Captured from the pre-change code, byte for byte.
EXPECTED_MARKDOWN = "# Page one\n\nBody text.\n\n| a | b |\n| - | - |\n| 1 | 2 |"
EXPECTED_HTML = (
    "<h1>Page one</h1><p>Body text.</p>\n\n<table><tr><td>1</td><td>2</td></tr></table>"
)
EXPECTED_METADATA = {
    "file_name": "sample.pdf",
    "num_pages": 2,
    "total_token_count": 118,
    "total_chunks": 3,
    "total_images": 0,
    "pages": [
        {
            "page_num": 0,
            "page_box": PAGE_BOX,
            "token_count": 41,
            "num_chunks": 2,
            "num_images": 0,
        },
        {
            "page_num": 1,
            "page_box": PAGE_BOX,
            "token_count": 77,
            "num_chunks": 1,
            "num_images": 0,
        },
    ],
}


@pytest.fixture
def results():
    return [
        BatchOutputItem(
            markdown="# Page one\n\nBody text.",
            html="<h1>Page one</h1><p>Body text.</p>",
            chunks=[
                {
                    "bbox": [10, 20, 300, 60],
                    "label": "Section-header",
                    "content": "<h1>Page one</h1>",
                },
                {
                    "bbox": [10, 70, 300, 140],
                    "label": "Text",
                    "content": "<p>Body text.</p>",
                },
            ],
            raw="<div data-label='Section-header'></div>",
            page_box=list(PAGE_BOX),
            token_count=41,
            images={},
            error=False,
        ),
        BatchOutputItem(
            markdown="| a | b |\n| - | - |\n| 1 | 2 |",
            html="<table><tr><td>1</td><td>2</td></tr></table>",
            chunks=[
                {
                    "bbox": [5, 5, 500, 400],
                    "label": "Table",
                    "content": "<table><tr><td>1</td><td>2</td></tr></table>",
                }
            ],
            raw="<div data-label='Table'></div>",
            page_box=list(PAGE_BOX),
            token_count=77,
            images={},
            error=False,
        ),
    ]


def run(tmp_path: Path, results, **kwargs) -> Path:
    save_merged_output(tmp_path, "sample.pdf", results, **kwargs)
    return tmp_path / "sample"


def test_chunks_sidecar_is_written(tmp_path, results):
    out = run(tmp_path, results)

    chunks_path = out / "sample_chunks.json"
    assert chunks_path.exists()

    with open(chunks_path, encoding="utf-8") as f:
        pages = json.load(f)

    assert pages == [
        {
            "page_num": 0,
            "page_box": PAGE_BOX,
            "chunks": results[0].chunks,
        },
        {
            "page_num": 1,
            "page_box": PAGE_BOX,
            "chunks": results[1].chunks,
        },
    ]


def test_chunks_sidecar_can_be_disabled(tmp_path, results):
    out = run(tmp_path, results, save_chunks=False)

    assert not (out / "sample_chunks.json").exists()
    assert (out / "sample.md").exists()


def test_markdown_and_html_are_unchanged(tmp_path, results):
    out = run(tmp_path, results)

    assert (out / "sample.md").read_bytes() == EXPECTED_MARKDOWN.encode("utf-8")
    assert (out / "sample.html").read_bytes() == EXPECTED_HTML.encode("utf-8")


def test_paginated_markdown_and_html_are_unchanged(tmp_path, results):
    out = run(tmp_path, results, paginate_output=True)

    expected_markdown = (
        "# Page one\n\nBody text.\n\n1"
        + "-" * 48
        + ("\n\n| a | b |\n| - | - |\n| 1 | 2 |")
    )
    expected_html = (
        "<h1>Page one</h1><p>Body text.</p>\n\n<!-- Page 2 -->\n\n"
        "<table><tr><td>1</td><td>2</td></tr></table>"
    )
    assert (out / "sample.md").read_bytes() == expected_markdown.encode("utf-8")
    assert (out / "sample.html").read_bytes() == expected_html.encode("utf-8")


def test_metadata_keys_are_unchanged(tmp_path, results):
    out = run(tmp_path, results)

    with open(out / "sample_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata == EXPECTED_METADATA


def test_no_html_option_still_skips_html(tmp_path, results):
    out = run(tmp_path, results, save_html=False)

    assert not (out / "sample.html").exists()
    assert (out / "sample_chunks.json").exists()
