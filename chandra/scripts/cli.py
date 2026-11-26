"""Chandra CLI."""

import json
import os
from pathlib import Path
import sys
from typing import List

import click

from chandra import __version__
from chandra.input import (
    SUPPORTED_FILE_EXTENSIONS,
    discover_input_files,
    load_file,
)
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.settings import settings
from chandra.skills import get_skill_markdown, install_skill


def get_supported_files(input_path: Path, recursive: bool = False) -> List[Path]:
    """Get list of supported image/PDF files from path."""
    if not input_path.exists():
        raise click.BadParameter(f"Path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
            raise click.BadParameter(f"Unsupported file type: {input_path.suffix}")
        return [input_path]
    try:
        return discover_input_files(input_path, recursive=recursive)
    except Exception as exc:
        raise click.BadParameter(str(exc)) from exc


def save_merged_output(
    output_dir: Path,
    file_name: str,
    results: List,
    save_images: bool = True,
    save_html: bool = True,
    save_chunks: bool = True,
    paginate_output: bool = False,
):
    """Save merged OCR results for all pages to output directory."""
    safe_name = os.path.basename(Path(file_name).stem) or "document"
    file_output_dir = output_dir / safe_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    all_markdown = []
    all_html = []
    all_metadata = []
    all_chunks = []
    total_tokens = 0
    total_chunks = 0
    total_images = 0

    for page_num, result in enumerate(results):
        if page_num > 0 and paginate_output:
            all_markdown.append(f"\n\n{page_num}" + "-" * 48 + "\n\n")
            all_html.append(f"\n\n<!-- Page {page_num + 1} -->\n\n")

        all_markdown.append(result.markdown)
        all_html.append(result.html)

        if not paginate_output and page_num < len(results) - 1:
            all_markdown.append("\n\n")
            all_html.append("\n\n")

        total_tokens += result.token_count
        total_chunks += len(result.chunks)
        total_images += len(result.images)

        page_metadata = {
            "page_num": page_num,
            "page_box": result.page_box,
            "token_count": result.token_count,
            "num_chunks": len(result.chunks),
            "num_images": len(result.images),
        }
        all_metadata.append(page_metadata)

        all_chunks.append(
            {
                "page_num": page_num,
                "page_box": result.page_box,
                "chunks": result.chunks,
            }
        )

        if save_images and result.images:
            for img_name, pil_image in result.images.items():
                safe_img_name = os.path.basename(img_name)
                img_path = file_output_dir / safe_img_name
                pil_image.save(img_path)

    markdown_path = file_output_dir / f"{safe_name}.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("".join(all_markdown))

    if save_html:
        html_path = file_output_dir / f"{safe_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(all_html))

    metadata = {
        "file_name": file_name,
        "num_pages": len(results),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "total_images": total_images,
        "pages": all_metadata,
    }
    metadata_path = file_output_dir / f"{safe_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if save_chunks:
        chunks_path = file_output_dir / f"{safe_name}_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

    click.echo(f"  Saved: {markdown_path} ({len(results)} page(s))")


@click.command(
    help="Convert PDFs and images to Markdown, HTML, and JSON layout chunks using Chandra OCR 2."
)
@click.version_option(version=__version__, prog_name="chandra-ocr")
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
@click.argument(
    "output_path",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--method",
    type=click.Choice(["hf", "vllm"], case_sensitive=False),
    default="vllm",
    help="Inference method: 'hf' for local model, 'vllm' for vLLM server.",
)
@click.option(
    "--page-range",
    type=str,
    default=None,
    help="0-indexed page range for PDFs (e.g., '0-4,6,8-11'). Only applicable to PDF files.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Maximum number of output tokens per page.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of parallel workers for vLLM inference.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Maximum number of retries for vLLM inference.",
)
@click.option(
    "--vllm-api-base",
    type=str,
    default=None,
    help=f"vLLM OpenAI API base URL (default: {settings.VLLM_API_BASE}).",
)
@click.option(
    "--vllm-api-key",
    type=str,
    default=None,
    help="vLLM OpenAI API key.",
)
@click.option(
    "--vllm-model-name",
    type=str,
    default=None,
    help=f"Served model name on the vLLM endpoint (default: {settings.VLLM_MODEL_NAME}).",
)
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Include images in output.",
)
@click.option(
    "--include-headers-footers/--no-headers-footers",
    default=False,
    help="Include page headers and footers in output.",
)
@click.option(
    "--save-html/--no-html",
    default=True,
    help="Save HTML output files.",
)
@click.option(
    "--save-chunks/--no-chunks",
    default=True,
    help="Save per-page layout chunks (bbox, label, content) to <name>_chunks.json.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process in a batch (default: all pages of a file for vllm, 1 for hf).",
)
@click.option(
    "--paginate_output",
    "--paginate-output",
    is_flag=True,
    default=False,
    help="Insert page separators between pages in merged output.",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    default=False,
    help="Recursively search subdirectories when INPUT_PATH is a directory.",
)
@click.option(
    "--stdout",
    is_flag=True,
    default=False,
    help="Write converted Markdown directly to stdout instead of saving to disk.",
)
@click.option(
    "--skill",
    is_flag=True,
    default=False,
    help="Print the Chandra AI agent skill (SKILL.md) to stdout and exit.",
)
@click.option(
    "--install-skill",
    "install_skill_dir",
    type=str,
    default=None,
    help="Install the Chandra SKILL.md into the given directory (e.g. .agents/skills) and exit.",
)
def main(
    input_path: Path | None,
    output_path: Path | None,
    method: str,
    page_range: str | None,
    max_output_tokens: int | None,
    max_workers: int | None,
    max_retries: int | None,
    vllm_api_base: str | None = None,
    vllm_api_key: str | None = None,
    vllm_model_name: str | None = None,
    include_images: bool = True,
    include_headers_footers: bool = False,
    save_html: bool = True,
    save_chunks: bool = True,
    batch_size: int | None = None,
    paginate_output: bool = False,
    recursive: bool = False,
    stdout: bool = False,
    skill: bool = False,
    install_skill_dir: str | None = None,
):
    if skill:
        sys.stdout.write(get_skill_markdown())
        return
    if install_skill_dir:
        dest = install_skill(install_skill_dir)
        click.echo(f"Installed Chandra skill to {dest}")
        return

    if input_path is None:
        raise click.UsageError("Missing argument 'INPUT_PATH'.")
    if output_path is None and not stdout:
        raise click.UsageError(
            "Missing argument 'OUTPUT_PATH' (or pass --stdout to write to stdout)."
        )

    method = method.lower()
    if method == "hf":
        if not stdout:
            click.echo(
                "When using '--method hf', ensure that the batch size is set correctly. We will default to batch size of 1."
            )
        if batch_size is None:
            batch_size = 1
    # For vllm, batch_size None sends all of a file's pages at once: the client
    # keeps --max-workers of them in flight and sends the next page as soon as
    # one finishes, instead of waiting for each batch's slowest page.

    if not stdout:
        click.echo("Chandra CLI - Starting OCR processing")
        click.echo(f"Input: {input_path}")
        click.echo(f"Output: {output_path}")
        click.echo(f"Method: {method}")
        output_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"\nLoading model with method '{method}'...")

    model = InferenceManager(method=method)
    if not stdout:
        click.echo("Model loaded successfully.")

    files_to_process = get_supported_files(input_path, recursive=recursive)
    if not stdout:
        click.echo(f"\nFound {len(files_to_process)} file(s) to process.")

    if not files_to_process:
        if not stdout:
            click.echo("No supported files found. Exiting.")
        return

    stdout_chunks: List[str] = []
    for file_idx, file_path in enumerate(files_to_process, 1):
        if not stdout:
            click.echo(
                f"\n[{file_idx}/{len(files_to_process)}] Processing: {file_path.name}"
            )

        try:
            config = {"page_range": page_range} if page_range else {}
            images = load_file(str(file_path), config)
            if not stdout:
                click.echo(f"  Loaded {len(images)} page(s)")

            all_results = []
            step = batch_size or max(1, len(images))
            for batch_start in range(0, len(images), step):
                batch_end = min(batch_start + step, len(images))
                batch_images = images[batch_start:batch_end]
                batch = [
                    BatchInputItem(image=img, prompt_type="ocr_layout")
                    for img in batch_images
                ]

                if not stdout:
                    click.echo(f"  Processing pages {batch_start + 1}-{batch_end}...")

                generate_kwargs = {
                    "include_images": include_images,
                    "include_headers_footers": include_headers_footers,
                }
                if max_output_tokens is not None:
                    generate_kwargs["max_output_tokens"] = max_output_tokens

                if method == "vllm":
                    if max_workers is not None:
                        generate_kwargs["max_workers"] = max_workers
                    if max_retries is not None:
                        generate_kwargs["max_retries"] = max_retries
                    if vllm_api_base is not None:
                        generate_kwargs["vllm_api_base"] = vllm_api_base
                    if vllm_api_key is not None:
                        generate_kwargs["vllm_api_key"] = vllm_api_key
                    if vllm_model_name is not None:
                        generate_kwargs["vllm_model_name"] = vllm_model_name

                results = model.generate(batch, **generate_kwargs)
                all_results.extend(results)

            if stdout:
                parts = []
                for page_num, res in enumerate(all_results):
                    if page_num > 0 and paginate_output:
                        parts.append(f"\n\n{page_num}" + "-" * 48 + "\n\n")
                    elif page_num > 0:
                        parts.append("\n\n")
                    parts.append(res.markdown)
                stdout_chunks.append("".join(parts))
            else:
                save_merged_output(
                    output_path,
                    file_path.name,
                    all_results,
                    save_images=include_images,
                    save_html=save_html,
                    save_chunks=save_chunks,
                    paginate_output=paginate_output,
                )
                click.echo(f"  Completed: {file_path.name}")

        except Exception as e:
            click.echo(f"  Error processing {file_path.name}: {e}", err=True)
            continue

    if stdout:
        out_text = "\n\n".join(stdout_chunks)
        try:
            sys.stdout.write(out_text)
            if out_text and not out_text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    else:
        click.echo(f"\nProcessing complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
