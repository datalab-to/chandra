from dataclasses import dataclass
from typing import Dict, List

from PIL import Image


@dataclass
class GenerationResult:
    raw: str
    token_count: int
    error: bool = False


@dataclass
class BatchInputItem:
    image: Image.Image
    prompt: str | None = None
    prompt_type: str | None = "ocr_layout"


@dataclass
class BatchOutputItem:
    markdown: str
    html: str
    chunks: List[Dict]
    raw: str
    page_box: List[int]
    token_count: int
    images: Dict[str, Image.Image]
    error: bool
