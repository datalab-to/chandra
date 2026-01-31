import os
import logging
from typing import List, Optional

import filetype
from PIL import Image, UnidentifiedImageError
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from chandra.settings import settings

logger = logging.getLogger(__name__)


def flatten(page, flag=pdfium_c.FLAT_NORMALDISPLAY):
    rc = pdfium_c.FPDFPage_Flatten(page, flag)
    if rc == pdfium_c.FLATTEN_FAIL:
        logger.warning(f"Failed to flatten annotations / form fields on page {page}.")


def load_image(
    filepath: str, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    try:
        image = Image.open(filepath).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Could not identify image file: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to load image {filepath}: {e}")

    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(
    filepath: str,
    page_range: List[int],
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> List[Image.Image]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    try:
        doc = pdfium.PdfDocument(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF document {filepath}: {e}")

    try:
        doc.init_forms()
        images = []
        for page_idx in range(len(doc)):
            if not page_range or page_idx in page_range:
                try:
                    page_obj = doc[page_idx]
                    min_page_dim = min(page_obj.get_width(), page_obj.get_height())
                    
                    # Calculate scale to match DPI requirement
                    scale_dpi = (min_pdf_image_dim / min_page_dim) * 72
                    scale_dpi = max(scale_dpi, image_dpi)
                    
                    flatten(page_obj)
                    
                    # Re-access page object after flattening if necessary or just use the same handle
                    # pypdfium2 page objects might be invalidated by some operations, but flatten usually works in place.
                    # The original code re-accessed `doc[page]`. Let's stick to that pattern to be safe.
                    page_obj = doc[page_idx]
                    
                    pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
                    images.append(pil_image)
                except Exception as e:
                    logger.error(f"Error processing page {page_idx} of {filepath}: {e}")
                    # Continue to next page instead of failing everything?
                    # Depending on strictness, maybe we want to raise. 
                    # For now, logging and skipping is robust.
                    continue
        return images
    finally:
        doc.close()


def parse_range_str(range_str: str) -> List[int]:
    try:
        range_lst = range_str.split(",")
        page_lst = []
        for i in range_lst:
            i = i.strip()
            if not i:
                continue
            if "-" in i:
                start, end = i.split("-")
                page_lst += list(range(int(start), int(end) + 1))
            else:
                page_lst.append(int(i))
        page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
        return page_lst
    except ValueError as e:
        raise ValueError(f"Invalid page range format '{range_str}': {e}")


def load_file(filepath: str, config: dict) -> List[Image.Image]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    page_range = config.get("page_range")
    parsed_page_range = []
    if page_range:
        parsed_page_range = parse_range_str(page_range)

    input_type = filetype.guess(filepath)
    # Default to image if type cannot be guessed or is not PDF, 
    # but we should probably check if it IS an image or PDF.
    # filetype returns None if unknown.
    
    is_pdf = False
    if input_type and input_type.extension == "pdf":
        is_pdf = True
    elif filepath.lower().endswith(".pdf"): # Fallback if filetype fails but extension is pdf
        is_pdf = True

    if is_pdf:
        images = load_pdf_images(filepath, parsed_page_range)
    else:
        # Assume it's an image if not PDF
        images = [load_image(filepath)]
    return images
