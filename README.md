<p align="center">
  <img src="assets/datalab-logo.png" alt="Datalab Logo" width="150"/>
</p>
<h1 align="center">Datalab</h1>
<p align="center">
  <strong>State of the Art models for Document Intelligence</strong>
</p>
<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License"></a>
  <a href="https://www.datalab.to/pricing"><img src="https://img.shields.io/badge/Model%20License-OpenRAIL--M-blue.svg" alt="Model License"></a>
  <a href="https://discord.gg/KuZwXNGnfH"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

# Chandra

An OCR model for complex documents — handwriting, tables, math equations, and messy forms.

<img src="assets/examples/forms/handwritten_form.png" width="700px"/>

## Benchmarks

Overall scores on the [olmocr bench](https://github.com/allenai/olmocr):

<img src="assets/benchmarks/bench.png" width="600px"/>

## Hosted API

A hosted API with additional accuracy improvements is available at [datalab.to](https://www.datalab.to/). Try the [free playground](https://www.datalab.to/playground) without installing.

## Community

Join [Discord](https://discord.gg//KuZwXNGnfH) to discuss development and get help.

## Quick Start

```shell
pip install chandra-ocr

# Start vLLM server, then run OCR
chandra_vllm
chandra input.pdf ./output

# Or use HuggingFace locally
chandra input.pdf ./output --method hf

# Interactive web app
chandra_app
```

**Python:**

```python
from chandra.model import InferenceManager
from chandra.input import load_pdf_images

manager = InferenceManager(method="hf")
images = load_pdf_images("document.pdf")
results = manager.generate(images)
print(results[0].markdown)
```

## How it Works.

- **Two inference modes**: Run locally via HuggingFace Transformers, or deploy a vLLM server for production throughput
- **Layout-aware output**: Every text block, table, and image comes with bounding box coordinates
- **Structured formats**: Output as Markdown, HTML, or JSON with full layout metadata
- **40+ languages** supported

## What It Handles

**Handwriting** — Doctor notes, filled forms, homework. Chandra reads cursive and messy print that trips up traditional OCR.

**Tables** — Preserves structure including merged cells (colspan/rowspan). Works on financial filings, invoices, and data tables.

**Math** — Inline and block equations rendered as LaTeX. Handles textbooks, worksheets, and research papers.

**Forms** — Reconstructs checkboxes, radio buttons, and form fields with their values.

**Complex Layouts** — Multi-column documents, newspapers, textbooks with figures and captions.

## Examples

| | |
|---|---|
| <img src="assets/examples/handwriting/doctor_note.png" width="350px"/><br/>**Handwriting** | <img src="assets/examples/tables/water_damage.png" width="350px"/><br/>**Tables** |
| <img src="assets/examples/math/ega.png" width="350px"/><br/>**Math** | <img src="assets/examples/newspapers/nyt.png" width="350px"/><br/>**Newspapers** |

<details>
<summary>More examples</summary>

| Type | Name | Link |
|------|------|------|
| Tables | 10K Filing | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/tables/10k.png) |
| Forms | Lease Agreement | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/forms/lease.png) |
| Handwriting | Math Homework | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/handwriting/math_hw.png) |
| Books | Geography Textbook | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/books/geo_textbook_page.png) |
| Books | Exercise Problems | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/books/exercises.png) |
| Math | Attention Diagram | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/math/attn_all.png) |
| Math | Worksheet | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/math/worksheet.png) |
| Newspapers | LA Times | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/newspapers/la_times.png) |
| Other | Transcript | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/other/transcript.png) |
| Other | Flowchart | [View](https://github.com/datalab-to/chandra/blob/master/assets/examples/other/flowchart.png) |

</details>

## Installation

```bash
pip install chandra-ocr
```

For HuggingFace inference, we recommend installing [flash attention](https://github.com/Dao-AILab/flash-attention) for better performance.

**From source:**

```bash
git clone https://github.com/datalab-to/chandra.git
cd chandra
uv sync
source .venv/bin/activate
```

## Usage

### CLI

```bash
# Single file with vLLM server
chandra input.pdf ./output --method vllm

# Directory with local model
chandra ./documents ./output --method hf
```

**Options:**
- `--method [hf|vllm]`: Inference method (default: vllm)
- `--page-range TEXT`: Page range for PDFs (e.g., "1-5,7,9-12")
- `--max-output-tokens INTEGER`: Max tokens per page
- `--max-workers INTEGER`: Parallel workers for vLLM
- `--include-images/--no-images`: Extract and save images (default: include)
- `--include-headers-footers/--no-headers-footers`: Include page headers/footers (default: exclude)
- `--batch-size INTEGER`: Pages per batch (default: 1)

**Output structure:**

```
output/
└── filename/
    ├── filename.md           # Markdown
    ├── filename.html         # HTML with bounding boxes
    ├── filename_metadata.json
    └── images/               # Extracted images
```

### vLLM Server

For production or batch processing:

```bash
chandra_vllm
```

Launches a Docker container with optimized inference. Configure via environment:

- `VLLM_API_BASE`: Server URL (default: `http://localhost:8000/v1`)
- `VLLM_MODEL_NAME`: Model name (default: `chandra`)
- `VLLM_GPUS`: GPU device IDs (default: `0`)

### Configuration

Settings via environment variables or `local.env`:

```bash
MODEL_CHECKPOINT=datalab-to/chandra
MAX_OUTPUT_TOKENS=8192
VLLM_API_BASE=http://localhost:8000/v1
VLLM_GPUS=0
```

## Commercial Usage

Code is Apache 2.0. Model weights use a modified OpenRAIL-M license: free for research, personal use, and startups under $2M funding/revenue. Cannot be used competitively with our API. For broader commercial licensing, see [pricing](https://www.datalab.to/pricing?utm_source=gh-chandra).

## Credits

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [olmocr](https://github.com/allenai/olmocr)
- [Qwen3 VL](https://github.com/QwenLM/Qwen3)

## Support Datalab
If you find this repository helpful, please consider giving it a star ⭐
