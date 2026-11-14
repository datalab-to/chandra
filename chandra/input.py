import os
from pathlib import Path
from typing import List

import filetype
from PIL import Image
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from chandra.settings import settings

SUPPORTED_FILE_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
    ".avif",
}


def discover_input_files(
    input_path: str | Path, recursive: bool = False
) -> List[Path]:
    """Discover supported PDF and image files from a file or directory path.

    Iterates directory entries once and checks lowercase extensions so files are
    never duplicated on case-insensitive filesystems (macOS APFS, Windows NTFS).
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Invalid input path: {input_path}")

    discovered: List[Path] = []
    if recursive:
        for root, dirs, filenames in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_FILE_EXTENSIONS:
                    full = Path(root) / fname
                    if full.is_file():
                        discovered.append(full)
    else:
        for entry in sorted(path.iterdir(), key=lambda p: p.name):
            if entry.name.startswith(".") or not entry.is_file():
                continue
            if entry.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                discovered.append(entry)

    return discovered


def flatten(page, flag=pdfium_c.FLAT_NORMALDISPLAY):
    rc = pdfium_c.FPDFPage_Flatten(page, flag)
    if rc == pdfium_c.FLATTEN_FAIL:
        print(f"Failed to flatten annotations / form fields on page {page}.")


def load_image(
    filepath: str | Path, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    with Image.open(filepath) as im:
        image = im.convert("RGB")
    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(
    filepath: str | Path,
    page_range: List[int] | None = None,
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> List[Image.Image]:
    if page_range is not None and len(page_range) == 0:
        return []

    page_set = set(page_range) if page_range is not None else None
    doc = pdfium.PdfDocument(filepath)
    try:
        doc.init_forms()
        images = []
        for page in range(len(doc)):
            if page_set is None or page in page_set:
                page_obj = doc[page]
                min_page_dim = max(1.0, min(page_obj.get_width(), page_obj.get_height()))
                scale_dpi = (min_pdf_image_dim / min_page_dim) * 72
                scale_dpi = max(scale_dpi, image_dpi)
                flatten(page_obj)
                page_obj = doc[page]
                pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
                images.append(pil_image)
        return images
    finally:
        doc.close()


def parse_range_str(range_str: str) -> List[int]:
    if not range_str or not range_str.strip():
        return []
    range_lst = range_str.split(",")
    page_lst = []
    for item in range_lst:
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            start, end = int(parts[0].strip()), int(parts[1].strip())
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range: {token!r}")
            page_lst.extend(range(start - 1, end))
        else:
            val = int(token)
            if val < 1:
                raise ValueError(f"Invalid page number: {token!r}")
            page_lst.append(val - 1)
    return sorted(set(page_lst))


def load_file(filepath: str | Path, config: dict | None = None) -> List[Image.Image]:
    config = config or {}
    page_range = config.get("page_range")
    if isinstance(page_range, str):
        page_range = parse_range_str(page_range)

    image_dpi = config.get("image_dpi", settings.IMAGE_DPI)
    min_pdf_image_dim = config.get("min_pdf_image_dim", settings.MIN_PDF_IMAGE_DIM)
    min_image_dim = config.get("min_image_dim", settings.MIN_IMAGE_DIM)

    input_type = filetype.guess(str(filepath))
    is_pdf = (input_type and input_type.extension == "pdf") or str(
        filepath
    ).lower().endswith(".pdf")

    if is_pdf:
        return load_pdf_images(
            filepath,
            page_range=page_range,
            image_dpi=image_dpi,
            min_pdf_image_dim=min_pdf_image_dim,
        )
    return [load_image(filepath, min_image_dim=min_image_dim)]
