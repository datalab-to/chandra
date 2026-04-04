"""Tests for image filename consistency between parse_html and extract_images.

When a document contains Blank-Page divs before Image/Figure divs, the div
index used to generate image filenames must be identical in both parse_html
(which creates the <img src="..."> references in the output HTML) and
extract_images (which saves the cropped image files).

Before the fix, extract_images iterated over `chunks` (which has Blank-Page
divs already filtered out by parse_layout), maintaining its own div_idx
counter that started from 1 for the first *non-blank* div.  parse_html, on
the other hand, counted *all* divs including Blank-Page, so its div_idx for
the same Image div was higher.  This caused the HTML to reference an image
filename that didn't match the filename under which extract_images saved it.
"""

import hashlib
import re

import pytest
from PIL import Image

from chandra.output import (
    extract_images,
    get_image_name,
    parse_chunks,
    parse_html,
    parse_layout,
    parse_markdown,
    LayoutBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width=200, height=200):
    """Create a simple test image."""
    return Image.new("RGB", (width, height), "white")


def _build_html(*divs):
    """Build an HTML string from a list of div strings."""
    return "".join(divs)


def _div(label, content, bbox="0 0 500 500"):
    """Convenience to create a single div block."""
    return f'<div data-label="{label}" data-bbox="{bbox}">{content}</div>'


def _image_div(bbox="0 0 500 500"):
    return _div("Image", '<img alt="photo"/>', bbox)


def _figure_div(bbox="0 0 500 500"):
    return _div("Figure", '<img alt="diagram"/>', bbox)


def _text_div(text="Hello", bbox="0 0 500 100"):
    return _div("Text", text, bbox)


def _blank_div():
    return '<div data-label="Blank-Page" data-bbox="0 0 1000 1000"></div>'


def _extract_img_srcs(html_out):
    """Extract all img src values from rendered HTML."""
    return re.findall(r'src=["\']([^"\']+)["\']', html_out)


# ---------------------------------------------------------------------------
# Core regression test: Blank-Page before Image
# ---------------------------------------------------------------------------

class TestBlankPageImageMismatch:
    """Regression tests for the div_idx mismatch when Blank-Page precedes Image."""

    def test_single_blank_before_image(self):
        """Blank-Page (idx 1) + Image (idx 2): filenames must match."""
        html = _build_html(_blank_div(), _image_div())
        image = _make_image()

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 1, f"Expected 1 img src, got {srcs}"
        assert len(images) == 1, f"Expected 1 extracted image, got {len(images)}"
        assert srcs[0] in images, (
            f"HTML references '{srcs[0]}' but extract_images saved {list(images.keys())}"
        )

    def test_multiple_blanks_before_image(self):
        """Three Blank-Page divs then an Image: idx must be 4 in both."""
        html = _build_html(_blank_div(), _blank_div(), _blank_div(), _image_div())
        image = _make_image()

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 1
        assert len(images) == 1
        assert srcs[0] in images

    def test_blank_between_text_and_image(self):
        """Text + Blank-Page + Image: Image div_idx should be 3."""
        html = _build_html(_text_div(), _blank_div(), _image_div())
        image = _make_image()

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 1
        assert len(images) == 1
        assert srcs[0] in images

    def test_multiple_images_with_blanks(self):
        """Two Image divs separated by Blank-Pages: all filenames must match."""
        html = _build_html(
            _image_div("0 0 500 500"),
            _blank_div(),
            _blank_div(),
            _image_div("500 0 1000 500"),
        )
        image = _make_image(1000, 500)

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 2, f"Expected 2 img srcs, got {srcs}"
        assert len(images) == 2, f"Expected 2 extracted images, got {len(images)}"
        for src in srcs:
            assert src in images, (
                f"HTML references '{src}' but extract_images saved {list(images.keys())}"
            )

    def test_figure_with_blanks(self):
        """Figure label should behave identically to Image."""
        html = _build_html(_blank_div(), _figure_div())
        image = _make_image()

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 1
        assert len(images) == 1
        assert srcs[0] in images


# ---------------------------------------------------------------------------
# No-blank baseline: ensure we didn't break the normal case
# ---------------------------------------------------------------------------

class TestNoBlankPages:
    """Verify the fix doesn't regress the happy path (no Blank-Page divs)."""

    def test_single_image_no_blanks(self):
        html = _build_html(_image_div())
        image = _make_image()

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 1
        assert srcs[0] in images

    def test_text_then_image_no_blanks(self):
        html = _build_html(_text_div(), _image_div())
        image = _make_image()

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 1
        assert srcs[0] in images

    def test_multiple_images_no_blanks(self):
        html = _build_html(
            _image_div("0 0 500 500"),
            _text_div("middle"),
            _figure_div("500 0 1000 500"),
        )
        image = _make_image(1000, 500)

        html_out = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        srcs = _extract_img_srcs(html_out)
        assert len(srcs) == 2
        assert len(images) == 2
        for src in srcs:
            assert src in images


# ---------------------------------------------------------------------------
# parse_layout div_idx tracking
# ---------------------------------------------------------------------------

class TestLayoutDivIdx:
    """Verify parse_layout stores the correct original div_idx."""

    def test_div_idx_sequential_no_blanks(self):
        html = _build_html(_text_div(), _image_div())
        image = _make_image()
        blocks = parse_layout(html, image)
        assert len(blocks) == 2
        assert blocks[0].div_idx == 1
        assert blocks[1].div_idx == 2

    def test_div_idx_skips_blank_page(self):
        html = _build_html(_text_div(), _blank_div(), _image_div())
        image = _make_image()
        blocks = parse_layout(html, image)
        # Blank-Page is filtered, but Image gets div_idx=3 (its real position)
        assert len(blocks) == 2
        assert blocks[0].div_idx == 1  # Text
        assert blocks[1].div_idx == 3  # Image (Blank-Page was at 2)

    def test_div_idx_multiple_blanks(self):
        html = _build_html(_blank_div(), _blank_div(), _text_div(), _blank_div(), _image_div())
        image = _make_image()
        blocks = parse_layout(html, image)
        assert len(blocks) == 2
        assert blocks[0].div_idx == 3  # Text (after 2 blanks)
        assert blocks[1].div_idx == 5  # Image (after 3 blanks total)

    def test_div_idx_preserved_in_chunks(self):
        """parse_chunks (which calls parse_layout → asdict) should carry div_idx."""
        html = _build_html(_blank_div(), _image_div())
        image = _make_image()
        chunks = parse_chunks(html, image)
        assert len(chunks) == 1
        assert chunks[0]["div_idx"] == 2


# ---------------------------------------------------------------------------
# Markdown output also references correct image names
# ---------------------------------------------------------------------------

class TestMarkdownImageNames:
    """parse_markdown uses parse_html internally, so image names should match too."""

    def test_markdown_image_name_with_blanks(self):
        html = _build_html(_blank_div(), _image_div())
        image = _make_image()

        md = parse_markdown(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)

        # The markdown should contain the image filename
        img_names = list(images.keys())
        assert len(img_names) == 1
        assert img_names[0] in md, (
            f"Markdown does not reference '{img_names[0]}'. Markdown:\n{md}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for robustness."""

    def test_all_blank_pages(self):
        """All divs are Blank-Page: no images, no chunks."""
        html = _build_html(_blank_div(), _blank_div())
        image = _make_image()

        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        assert len(chunks) == 0
        assert len(images) == 0

    def test_image_without_img_tag(self):
        """Image div without <img> tag: no image extracted, but no crash."""
        html = _build_html(_blank_div(), _div("Image", "just text"))
        image = _make_image()

        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        # parse_html adds an img tag, but extract_images checks chunks
        # which don't have one
        assert len(images) == 0

    def test_empty_html(self):
        html = ""
        image = _make_image()
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        assert len(chunks) == 0
        assert len(images) == 0

    def test_no_image_blocks(self):
        html = _build_html(_text_div(), _text_div())
        image = _make_image()
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        assert len(images) == 0
