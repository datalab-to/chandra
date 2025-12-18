import pypdfium2 as pdfium
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import re

from chandra.model import InferenceManager
from chandra.util import draw_layout
from chandra.input import load_pdf_images
from chandra.model.schema import BatchInputItem
from chandra.output import parse_layout


@st.cache_resource()
def load_model(method: str):
    return InferenceManager(method=method)


@st.cache_data()
def get_page_image(pdf_file, page_num):
    return load_pdf_images(pdf_file, [page_num])[0]


@st.cache_data()
def page_counter(pdf_file):
    doc = pdfium.PdfDocument(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def embed_images_in_markdown(markdown: str, images: dict) -> str:
    """Replace image filenames in markdown with base64 data URLs."""
    for img_name, pil_image in images.items():
        # Convert PIL image to base64 data URL
        data_url = pil_image_to_base64(pil_image, format="PNG")
        # Replace the image reference in markdown
        # Pattern matches: ![...](img_name) or ![...](img_name "title")
        pattern = rf'(!\[.*?\])\({re.escape(img_name)}(?:\s+"[^"]*")?\)'
        markdown = re.sub(pattern, rf"\1({data_url})", markdown)
    return markdown


def ocr_layout(
    img: Image.Image,
    model=None,
    prompt_type: str = "ocr_layout",
) -> (Image.Image, str):
    batch = BatchInputItem(
        image=img,
        prompt_type=prompt_type,
    )
    result = model.generate([batch])[0]
    layout = parse_layout(result.raw, img)
    layout_image = draw_layout(img, layout)
    return result, layout_image


st.set_page_config(layout="wide", page_title="Chandra OCR Demo")
col1, col2 = st.columns([0.5, 0.5])

st.markdown("""
# Chandra OCR Demo

This app will let you try chandra, a layout-aware vision language model.
""")

# Get model mode selection
model_mode = st.sidebar.selectbox(
    "Model Mode",
    ["None", "hf", "vllm"],
    index=0,
    help="Select how to run inference: hf loads the model in memory using huggingface transformers, vllm connects to a running vLLM server.",
)

# Only load model if a mode is selected
model = None
if model_mode == "None":
    st.warning("Please select a model mode (Local Model or vLLM Server) to run OCR.")
else:
    model = load_model(model_mode)

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.stop()

filetype = in_file.type
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_number = st.sidebar.number_input(
        f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )

    pil_image = get_page_image(in_file, page_number)
else:
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

prompt_type = st.sidebar.selectbox(
    "Prompt type",
    [
        "ocr_layout",
        "ocr",
        "ocr_layout_table_row",
    ],
)
run_ocr = st.sidebar.button("Run OCR")

if pil_image is None:
    st.stop()

if run_ocr:
    if model_mode == "None":
        st.error("Please select a model mode (hf or vllm) to run OCR.")
    else:
        result, layout_image = ocr_layout(
            pil_image,
            model,
            prompt_type,
        )

        # Embed images as base64 data URLs in the markdown
        markdown_with_images = embed_images_in_markdown(result.markdown, result.images)

        with col1:
            html_tab, text_tab, layout_tab = st.tabs(
                ["HTML", "HTML as text", "Layout Image"]
            )
            with html_tab:
                st.markdown(markdown_with_images, unsafe_allow_html=True)
                st.download_button(
                    label="Download Markdown",
                    data=result.markdown,
                    file_name=f"{in_file.name.rsplit('.', 1)[0]}_page{page_number if page_number is not None else 0}.md",
                    mime="text/markdown",
                )
            with text_tab:
                st.text(result.html)

            if layout_image:
                with layout_tab:
                    st.image(
                        layout_image,
                        caption="Detected Layout",
                        use_container_width=True,
                    )
                    st.text_area(result.raw)

with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
