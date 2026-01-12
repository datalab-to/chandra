from dataclasses import dataclass
from typing import List

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
    prompt_type: str | None = None


@dataclass
class BatchOutputItem:
    markdown: str
    html: str
    chunks: dict
    raw: str
    page_box: List[int]
    token_count: int
    images: dict
    error: bool
