---
name: chandra
description: Convert PDFs and images with complex tables, LaTeX math, forms, checkboxes, handwriting, and 90+ languages into structured Markdown, HTML, and JSON layout chunks using Chandra OCR 2 (chandra-ocr).
---

# Chandra OCR 2 Skill (`chandra-ocr`)

Use **Chandra OCR 2** (`datalab-to/chandra-ocr-2`) when you need high-accuracy OCR on documents containing **handwriting**, **complex tables (`colspan`/`rowspan`)**, **forms & checkboxes**, **equations/math**, **diagrams/charts**, or **multilingual text (90+ languages)**.

## 1. CLI Commands

### Convert Single File or Directory (`chandra`)
```bash
# Stream Markdown directly to stdout (ideal for AI agents and pipelines)
chandra document.pdf --stdout

# Save Markdown, HTML, layout chunks (<stem>_chunks.json), metadata, and cropped images to ./output
chandra document.pdf ./output --method vllm

# Recursively process a directory of PDFs/images
chandra ./documents ./output --recursive --method vllm

# Use local HuggingFace Transformers backend (requires `pip install "chandra-ocr[hf]"`)
chandra document.pdf ./output --method hf

# Select specific 0-indexed pages and include headers/footers
chandra document.pdf ./output --page-range "0-4,7" --include-headers-footers
```

### Launch vLLM Server (`chandra_vllm`)
```bash
# Launch optimized vLLM Docker/Podman container on 127.0.0.1:8000
chandra_vllm --gpu h100
```

### Launch Interactive Web Viewer (`chandra_app`)
```bash
# Start self-contained browser UI to inspect layout boxes, Markdown, HTML, and JSON page by page
chandra_app --port 8000
```

## 2. Python API

```python
from chandra import BatchInputItem, InferenceManager, load_file

manager = InferenceManager(method="vllm")  # or method="hf"
images = load_file("document.pdf", {"page_range": "0-2"})

batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
results = manager.generate(batch, include_images=True, include_headers_footers=False)

for page_idx, res in enumerate(results):
    print(f"--- Page {page_idx} ({len(res.chunks)} blocks) ---")
    print(res.markdown)
```
