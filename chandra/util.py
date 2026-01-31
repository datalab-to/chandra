import hashlib
import json
import re
import logging
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Any, Optional

import six
from PIL import Image
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter, re_whitespace

from chandra.settings import settings

logger = logging.getLogger(__name__)

@lru_cache
def _hash_html(html: str) -> str:
    return hashlib.md5(html.encode("utf-8")).hexdigest()


def get_image_name(html: str, div_idx: int) -> str:
    html_hash = _hash_html(html)
    return f"{html_hash}_{div_idx}_img.webp"


def extract_images(html: str, chunks: List[Dict[str, Any]], image: Image.Image) -> Dict[str, Image.Image]:
    images = {}
    div_idx = 0
    for idx, chunk in enumerate(chunks):
        div_idx += 1
        if chunk["label"] in ["Image", "Figure"]:
            # Check if content is a soup object or string
            # The original code did chunk["content"].find("img") which implies string?
            # parse_layout returns content as string.
            # But string doesn't have .find("img") like soup. 
            # Wait, `str.find` returns index. `soup.find` returns tag.
            # Let's check parse_layout: `content = str(div.decode_contents())`. It is a string.
            # So `chunk["content"].find("img")` returns an integer index. 
            # If it returns -1 (not found), then `if not img:` (int != 0) is true? No.
            # -1 is truthy in Python. 0 is falsy.
            # If "img" is at start (index 0), `if not img` is True (skips it!). BUG?
            # Ah, looking at `extract_images` in original:
            # `img = chunk["content"].find("img")`
            # `if not img:` 
            # If `img` is an integer, this logic is flawed for index 0.
            # BUT, usually `find` on string returns -1 if not found.
            # `if -1:` is True.
            # `if 0:` is False.
            # So if <img... starts at 0, it skips?
            # Let's assume the intention was to parse it as soup. 
            # Or maybe `chunk["content"]` WAS a soup object in some version?
            # Re-reading original `extract_images`:
            # `img = chunk["content"].find("img")`
            # If `chunk["content"]` is a string, `find` is string search.
            # If "img" is not found (-1), it continues? No, -1 is True.
            # This looks like a bug in the original code or my understanding.
            # Let's fix it by parsing properly.
            
            soup_fragment = BeautifulSoup(chunk["content"], "html.parser")
            img_tag = soup_fragment.find("img")
            if not img_tag:
                continue
                
            bbox = chunk["bbox"]
            try:
                block_image = image.crop(bbox)
            except ValueError:
                # Happens when bbox coordinates are invalid
                logger.warning(f"Invalid bbox for cropping: {bbox}")
                continue
            img_name = get_image_name(html, div_idx)
            images[img_name] = block_image
    return images


def parse_html(
    html: str, include_headers_footers: bool = False, include_images: bool = True
) -> str:
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_html = ""
    image_idx = 0
    div_idx = 0
    for div in top_level_divs:
        div_idx += 1
        label = div.get("data-label")

        # Skip headers and footers if not included
        if label and not include_headers_footers:
            if label in ["Page-Header", "Page-Footer"]:
                continue
        if label and not include_images:
            if label in ["Image", "Figure"]:
                continue

        if label in ["Image", "Figure"]:
            img = div.find("img")
            img_src = get_image_name(html, div_idx)

            # If no tag, add one in
            if img:
                img["src"] = img_src
                image_idx += 1
            else:
                img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                div.append(img)

        # Wrap text content in <p> tags if no inner HTML tags exist
        if label in ["Text"] and not re.search(
            "<.+>", str(div.decode_contents()).strip()
        ):
            # Add inner p tags if missing for text blocks
            text_content = str(div.decode_contents()).strip()
            text_content = f"<p>{text_content}</p>"
            div.clear()
            div.append(BeautifulSoup(text_content, "html.parser"))

        content = str(div.decode_contents())
        out_html += content
    return out_html


class Markdownify(MarkdownConverter):
    def __init__(
        self,
        inline_math_delimiters,
        block_math_delimiters,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inline_math_delimiters = inline_math_delimiters
        self.block_math_delimiters = block_math_delimiters

    def convert_math(self, el, text, parent_tags):
        block = el.has_attr("display") and el["display"] == "block"
        if block:
            return (
                "\n"
                + self.block_math_delimiters[0]
                + text.strip()
                + self.block_math_delimiters[1]
                + "\n"
            )
        else:
            return (
                " "
                + self.inline_math_delimiters[0]
                + text.strip()
                + self.inline_math_delimiters[1]
                + " "
            )

    def convert_table(self, el, text, parent_tags):
        return "\n\n" + str(el) + "\n\n"

    def convert_a(self, el, text, parent_tags):
        text = self.escape(text)
        # Escape brackets and parentheses in text
        text = re.sub(r"([\[\]()])", r"\\\1", text)
        return super().convert_a(el, text, parent_tags)

    def escape(self, text, parent_tags=None):
        text = super().escape(text, parent_tags)
        if self.options["escape_dollars"]:
            text = text.replace("$", r"\$")
        return text

    def process_text(self, el, parent_tags=None):
        text = six.text_type(el) or ""

        # normalize whitespace if we're not inside a preformatted element
        if not el.find_parent("pre"):
            text = re_whitespace.sub(" ", text)

        # escape special characters if we're not inside a preformatted or code element
        if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text)

        # remove trailing whitespaces if any of the following condition is true:
        # - current text node is the last node in li
        # - current text node is followed by an embedded list
        if el.parent.name == "li" and (
            not el.next_sibling or el.next_sibling.name in ["ul", "ol"]
        ):
            text = text.rstrip()

        return text


def parse_markdown(
    html: str, include_headers_footers: bool = False, include_images: bool = True
) -> str:
    html = parse_html(html, include_headers_footers, include_images)

    md_cls = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )
    try:
        markdown = md_cls.convert(html)
    except Exception as e:
        logger.error(f"Error converting HTML to Markdown: {e}")
        markdown = ""
    return markdown.strip()


@dataclass
class LayoutBlock:
    bbox: list[int]
    label: str
    content: str


def parse_layout(html: str, image: Image.Image, bbox_scale=settings.BBOX_SCALE) -> List[LayoutBlock]:
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    width, height = image.size
    width_scaler = width / bbox_scale
    height_scaler = height / bbox_scale
    layout_blocks = []
    
    for div in top_level_divs:
        bbox_raw = div.get("data-bbox")
        bbox = [0, 0, 1, 1] # Default fallback

        if bbox_raw:
            try:
                # Try parsing as JSON list
                parsed_bbox = json.loads(bbox_raw)
                if isinstance(parsed_bbox, list) and len(parsed_bbox) == 4:
                    bbox = parsed_bbox
                else:
                    raise ValueError("Invalid bbox format (json)")
            except (json.JSONDecodeError, ValueError):
                # Try parsing as space-separated string
                try:
                    parts = bbox_raw.split(" ")
                    if len(parts) == 4:
                        bbox = [int(p) for p in parts]
                    else:
                        logger.warning(f"Invalid bbox string format: {bbox_raw}")
                except ValueError:
                     logger.warning(f"Could not parse bbox values: {bbox_raw}")

        bbox = list(map(int, bbox))
        # Normalize bbox
        bbox = [
            max(0, int(bbox[0] * width_scaler)),
            max(0, int(bbox[1] * height_scaler)),
            min(int(bbox[2] * width_scaler), width),
            min(int(bbox[3] * height_scaler), height),
        ]
        label = div.get("data-label", "block")
        content = str(div.decode_contents())
        layout_blocks.append(LayoutBlock(bbox=bbox, label=label, content=content))
    return layout_blocks


def parse_chunks(html: str, image: Image.Image, bbox_scale=settings.BBOX_SCALE) -> List[Dict[str, Any]]:
    layout = parse_layout(html, image, bbox_scale=bbox_scale)
    chunks = [asdict(block) for block in layout]
    return chunks
