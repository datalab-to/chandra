from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import re

from PIL import Image
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter, re_whitespace

from chandra.settings import settings

_DANGEROUS_CONTAINER_RE = re.compile(
    r"<\s*(script|style|iframe|object|embed|applet)\b[^>]*>[\s\S]*?<\s*/\s*\1\s*>",
    re.IGNORECASE,
)
_DANGEROUS_TAGS_RE = re.compile(
    r"<\s*/?\s*(?:script|iframe|object|embed|applet|meta|link|base|form|style)\b[^>]*>",
    re.IGNORECASE,
)
_EVENT_ATTR_RE = re.compile(
    r"""\s+on[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""",
    re.IGNORECASE,
)
_JS_URL_RE = re.compile(
    r"""(href|src)\s*=\s*(['"]?)\s*(?:(?:javascript|vbscript)\s*:|data\s*:\s*text/html\b)[^'">\s]*\2""",
    re.IGNORECASE,
)
_LEADING_OL_NUM_RE = re.compile(r"^\s*\d+[\.\)]\s+")


def sanitize_html(text: str) -> str:
    """Strip unsafe tags, event handler attributes, and javascript:/data:text/html URLs."""
    if not text:
        return ""
    cleaned = _DANGEROUS_CONTAINER_RE.sub("", text)
    cleaned = _DANGEROUS_TAGS_RE.sub("", cleaned)
    cleaned = _EVENT_ATTR_RE.sub("", cleaned)
    cleaned = _JS_URL_RE.sub(r'\1="#"', cleaned)
    return cleaned


@lru_cache
def _hash_html(html: str):
    return hashlib.md5(html.encode("utf-8")).hexdigest()


def get_image_name(html: str, div_idx: int):
    html_hash = _hash_html(html)
    return f"{html_hash}_{div_idx}_img.webp"


def extract_images(html: str, chunks: list[dict], image: Image.Image):
    images = {}
    for div_idx, chunk in enumerate(chunks, start=1):
        if (chunk.get("label") or "").lower() in ("image", "figure"):
            bbox = chunk["bbox"]
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                continue
            try:
                block_image = image.crop(bbox)
            except ValueError:
                continue
            img_name = get_image_name(html, div_idx)
            images[img_name] = block_image
    return images


def parse_html(
    html: str, include_headers_footers: bool = False, include_images: bool = True
):
    soup = BeautifulSoup(sanitize_html(html), "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_blocks = []
    image_idx = 0
    div_idx = 0
    for div in top_level_divs:
        div_idx += 1
        label = div.get("data-label")
        label_lower = (label or "").lower()

        if label_lower == "blank-page":
            continue

        # Skip headers and footers if not included
        if label_lower and not include_headers_footers:
            if label_lower in ("page-header", "page-footer"):
                continue
        if label_lower and not include_images:
            if label_lower in ("image", "figure"):
                continue

        if label_lower in ("image", "figure"):
            img = div.find("img")
            img_src = get_image_name(html, div_idx)

            # If no tag, add one in
            if img:
                img["src"] = img_src
                image_idx += 1
            else:
                img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                div.append(img)

        # Strip img tags without src in non-image blocks (model hallucinations)
        if label_lower not in ("image", "figure"):
            for img_tag in div.find_all("img"):
                if not img_tag.get("src"):
                    img_tag.decompose()

        # Wrap text content in <p> tags if no inner HTML tags exist
        if label_lower == "text" and not re.search(
            "<.+>", str(div.decode_contents()).strip()
        ):
            text_content = str(div.decode_contents()).strip()
            text_content = f"<p>{text_content}</p>"
            div.clear()
            div.append(BeautifulSoup(text_content, "html.parser"))

        content = str(div.decode_contents())
        # Preserve layout block boundaries even when the contents are only inline
        # markup so adjacent headers, footers, and headings never glue together.
        out_blocks.append(f"<div>{content}</div>")

    if not top_level_divs and html.strip():
        return sanitize_html(html)

    return "".join(out_blocks)


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

    def convert_div(self, el, text, parent_tags):
        stripped = text.strip()
        return f"\n\n{stripped}\n\n" if stripped else ""

    def convert_table(self, el, text, parent_tags):
        return "\n\n" + str(el) + "\n\n"

    def convert_li(self, el, text, parent_tags):
        parent = el.parent
        if parent and parent.name == "ol" and text:
            text = _LEADING_OL_NUM_RE.sub("", text, count=1)
        return super().convert_li(el, text, parent_tags)

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
        text = str(el) or ""

        # normalize whitespace if we're not inside a preformatted element
        if not el.find_parent("pre"):
            text = re_whitespace.sub(" ", text)

        # escape special characters if we're not inside a preformatted or code element
        if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text)

        # remove trailing whitespaces if any of the following conditions are true:
        # - current text node is the last node in li
        # - current text node is followed by an embedded list
        if el.parent and el.parent.name == "li" and (
            not el.next_sibling or getattr(el.next_sibling, "name", None) in ["ul", "ol"]
        ):
            text = text.rstrip()

        return text


def parse_markdown(
    html: str, include_headers_footers: bool = False, include_images: bool = True
):
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
        print(f"Error converting HTML to Markdown: {e}")
        markdown = ""
    return markdown.strip()


@dataclass
class LayoutBlock:
    bbox: list[int]
    label: str
    content: str


def _parse_raw_bbox(raw_bbox: str | None) -> list[float]:
    if not raw_bbox or not isinstance(raw_bbox, str):
        raise ValueError("Missing data-bbox")
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw_bbox)
    if len(nums) != 4:
        raise ValueError(f"Expected 4 numbers in bbox, got {raw_bbox!r}")
    return [float(n) for n in nums]


def parse_layout(html: str, image: Image.Image, bbox_scale=settings.BBOX_SCALE):
    soup = BeautifulSoup(sanitize_html(html), "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    width, height = image.size
    width_scaler = width / bbox_scale
    height_scaler = height / bbox_scale
    layout_blocks = []
    for div in top_level_divs:
        label = div.get("data-label")
        if label == "Blank-Page":
            continue

        raw_bbox = div.get("data-bbox")
        try:
            coords = _parse_raw_bbox(raw_bbox)
        except Exception:
            print(f"Invalid bbox format: {raw_bbox}, defaulting to full image")
            coords = [0.0, 0.0, float(bbox_scale), float(bbox_scale)]

        x0 = max(0, min(width, int(coords[0] * width_scaler)))
        y0 = max(0, min(height, int(coords[1] * height_scaler)))
        x1 = max(x0, min(width, int(coords[2] * width_scaler)))
        y1 = max(y0, min(height, int(coords[3] * height_scaler)))
        bbox = [x0, y0, x1, y1]

        if not label:
            label = "block"
        content = str(div.decode_contents())

        # Strip nested data-bbox attributes (not needed in open source)
        content_soup = BeautifulSoup(content, "html.parser")
        for tag in content_soup.find_all(attrs={"data-bbox": True}):
            del tag["data-bbox"]
        content = str(content_soup)

        layout_blocks.append(LayoutBlock(bbox=bbox, label=label, content=content))
    return layout_blocks


def parse_chunks(html: str, image: Image.Image, bbox_scale=settings.BBOX_SCALE):
    layout = parse_layout(html, image, bbox_scale=bbox_scale)
    chunks = [asdict(block) for block in layout]
    return chunks
