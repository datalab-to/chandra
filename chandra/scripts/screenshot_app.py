"""
Simple Flask app for generating screenshot-ready OCR visualizations.
Displays original image with layout overlays on the left and extracted markdown on the right.
"""

from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

from PIL import Image
from chandra.model import InferenceManager
from chandra.input import load_file
from chandra.model.schema import BatchInputItem
from chandra.output import parse_layout

app = Flask(__name__)

# Load model once at startup
model = None


def get_model():
    global model
    if model is None:
        model = InferenceManager(method="vllm")
    return model


def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def get_color_palette():
    """Return a color palette for different block types."""
    return {
        "Section-Header": "#4ECDC4",
        "Text": "#45B7D1",
        "List-Group": "#96CEB4",
        "Table": "#FFEAA7",
        "Figure": "#DDA15E",
        "Image": "#BC6C25",
        "Caption": "#C77DFF",
        "Equation": "#9D4EDD",
        "Page-Header": "#E0AFA0",
        "Page-Footer": "#D4A5A5",
        "Footnote": "#A8DADC",
        "Form": "#F4A261",
        "default": "#FF00FF",
    }


@app.route("/")
def index():
    return render_template("screenshot.html")


@app.route("/process", methods=["POST"])
def process():
    data = request.json
    file_path = data.get("file_path")
    page_number = data.get("page_number", 0)

    if not file_path:
        return jsonify({"error": "file_path is required"}), 400

    try:
        # Load image
        images = load_file(file_path, {"page_range": str(page_number)})
        if not images:
            return jsonify({"error": "No images found"}), 400

        img = images[0]

        # Run OCR
        model = get_model()
        batch = BatchInputItem(image=img, prompt_type="ocr_layout")
        result = model.generate([batch])[0]

        # Parse layout
        layout_blocks = parse_layout(result.raw, img)

        # Get markdown and HTML
        html = result.html

        # Convert extracted images to base64 and embed in HTML
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for img_name, pil_img in result.images.items():
            img_base64 = pil_image_to_base64(pil_img, format="PNG")
            # Find all img tags with this src
            img_tags = soup.find_all("img", src=img_name)
            if len(img_tags) == 0:
                print(f"Warning: No img tags found for {img_name}")
            for img_tag in img_tags:
                # Replace src with base64
                img_tag["src"] = img_base64

                # Wrap image with alt text display
                alt_text = img_tag.get("alt", "")
                if alt_text:
                    wrapper = soup.new_tag("div", **{"class": "image-wrapper"})
                    alt_div = soup.new_tag("div", **{"class": "image-alt-text"})
                    alt_div.string = alt_text
                    img_container = soup.new_tag(
                        "div", **{"class": "image-container-wrapper"}
                    )

                    # Move img into container
                    img_tag_copy = img_tag
                    img_tag.replace_with(wrapper)
                    img_container.append(img_tag_copy)

                    wrapper.append(alt_div)
                    wrapper.append(img_container)

        # Convert back to HTML string
        html_with_images = str(soup)

        # Prepare response
        img_base64 = pil_image_to_base64(img, format="PNG")
        img_width, img_height = img.size

        color_palette = get_color_palette()

        # Prepare layout blocks data
        blocks_data = []
        for block in layout_blocks:
            color = color_palette.get(block.label, color_palette["default"])
            blocks_data.append(
                {"bbox": block.bbox, "label": block.label, "color": color}
            )

        return jsonify(
            {
                "image_base64": img_base64,
                "image_width": img_width,
                "image_height": img_height,
                "blocks": blocks_data,
                "html": html_with_images,
                "markdown": result.markdown,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    app.run(host="0.0.0.0", port=8503)


if __name__ == "__main__":
    main()
