from typing import List, Union, Optional
import filetype
from PIL import Image
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from chandra.settings import settings


def flatten(page, flag=pdfium_c.FLAT_NORMALDISPLAY):
    rc = pdfium_c.FPDFPage_Flatten(page, flag)
    if rc == pdfium_c.FLATTEN_FAIL:
        print(f"Failed to flatten annotations / form fields on page {page}.")


def load_image(
    filepath: str, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    image = Image.open(filepath).convert("RGB")
    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(
    filepath: str,
    page_range: List[int],
    image_dpi: Optional[Union[int, List[int]]] = None,
    min_pdf_image_dim: Optional[Union[int, List[int]]] = None,
) -> List[Image.Image]:
    """
    Load PDF pages as images with configurable DPI.

    Args:
        filepath: Path to PDF file
        page_range: List of page indices to render
        image_dpi: Target DPI for rendering. Can be:
            - None: use settings.IMAGE_DPI for all pages (default)
            - int: use same DPI for all pages
            - List[int]: per-page DPI (must match length of page_range)
        min_pdf_image_dim: Minimum image dimension. Can be:
            - None: use settings.MIN_PDF_IMAGE_DIM for all pages (default)
            - int: use same value for all pages
            - List[int]: per-page value (must match length of page_range)

    Returns:
        List of PIL Images, one per page in page_range
    """
    doc = pdfium.PdfDocument(filepath)
    doc.init_forms()

    # Determine default values
    default_image_dpi = image_dpi if isinstance(image_dpi, int) else settings.IMAGE_DPI
    default_min_pdf_image_dim = min_pdf_image_dim if isinstance(min_pdf_image_dim, int) else settings.MIN_PDF_IMAGE_DIM

    # Handle per-page DPI lists
    is_per_page_dpi = isinstance(image_dpi, list)
    if not is_per_page_dpi and image_dpi is not None:
        # Convert single DPI value to list for all pages
        image_dpi = [image_dpi] * len(page_range)
        is_per_page_dpi = True

    is_per_page_min_dim = isinstance(min_pdf_image_dim, list)
    if not is_per_page_min_dim and min_pdf_image_dim is not None:
        # Convert single min_dim value to list for all pages
        min_pdf_image_dim = [min_pdf_image_dim] * len(page_range)
        is_per_page_min_dim = True

    if is_per_page_dpi and len(image_dpi) != len(page_range):
        raise ValueError(f"image_dpi list length ({len(image_dpi)}) must match page_range length ({len(page_range)})")
    if is_per_page_min_dim and len(min_pdf_image_dim) != len(page_range):
        raise ValueError(f"min_pdf_image_dim list length ({len(min_pdf_image_dim)}) must match page_range length ({len(page_range)})")

    images = []
    page_idx_in_range = 0

    for page in range(len(doc)):
        if not page_range or page in page_range:
            # Get DPI for this specific page
            if is_per_page_dpi:
                current_dpi = image_dpi[page_idx_in_range]
            elif image_dpi is None:
                current_dpi = settings.IMAGE_DPI
            else:
                current_dpi = default_image_dpi

            # Get min_dim for this specific page
            if is_per_page_min_dim:
                current_min_dim = min_pdf_image_dim[page_idx_in_range]
            elif min_pdf_image_dim is None:
                current_min_dim = settings.MIN_PDF_IMAGE_DIM
            else:
                current_min_dim = default_min_pdf_image_dim

            page_idx_in_range += 1

            page_obj = doc[page]
            min_page_dim = min(page_obj.get_width(), page_obj.get_height())

            scale_dpi = (current_min_dim / min_page_dim) * 72
            scale_dpi = max(scale_dpi, current_dpi)
            page_obj = doc[page]
            flatten(page_obj)
            page_obj = doc[page]
            pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
            images.append(pil_image)

    doc.close()
    return images


def parse_range_str(range_str: str) -> List[int]:
    range_lst = range_str.split(",")
    page_lst = []
    for i in range_lst:
        if "-" in i:
            start, end = i.split("-")
            page_lst += list(range(int(start), int(end) + 1))
        else:
            page_lst.append(int(i))
    page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
    return page_lst


def load_file(filepath: str, config: dict):
    page_range = config.get("page_range")
    if page_range:
        page_range = parse_range_str(page_range)

    input_type = filetype.guess(filepath)
    if input_type and input_type.extension == "pdf":
        images = load_pdf_images(filepath, page_range)
    else:
        images = [load_image(filepath)]
    return images
