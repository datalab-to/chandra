from PIL import Image
from PIL.ImageDraw import ImageDraw

from chandra.output import LayoutBlock


def draw_layout(image: Image.Image, layout_blocks: list[LayoutBlock]):
    draw_image = image.copy()
    draw = ImageDraw(draw_image)
    for block in layout_blocks:
        if block.bbox[2] <= block.bbox[0] or block.bbox[3] <= block.bbox[1]:
            continue

        draw.rectangle(block.bbox, outline="red", width=2)
        draw.text((block.bbox[0], block.bbox[1]), block.label, fill="blue")

        for table in block.table_row_bboxes:
            for row_idx, row_bbox in enumerate(table):
                draw.rectangle(row_bbox, outline="green", width=2)
                draw.text(
                    (row_bbox[0], row_bbox[1]),
                    f"Row {row_idx}",
                    fill="green",
                )

    return draw_image
