import json
from pathlib import Path
from typing import List

import click

from chandra.input import load_file
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.settings import settings


def get_supported_files(input_path: Path) -> List[Path]:
    """Get list of supported image/PDF files from path."""
    supported_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".tiff",
        ".bmp",
    }

    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        else:
            raise click.BadParameter(f"Unsupported file type: {input_path.suffix}")

    elif input_path.is_dir():
        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(files)

    else:
        raise click.BadParameter(f"Path does not exist: {input_path}")


def save_merged_output(
    output_dir: Path,
    file_name: str,
    results: List,
    save_images: bool = True,
    save_html: bool = True,
    paginate_output: bool = False,
):
    """Save merged OCR results for all pages to output directory."""
    # Create subfolder for this file
    safe_name = Path(file_name).stem
    file_output_dir = output_dir / safe_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Merge all pages
    all_markdown = []
    all_html = []
    all_metadata = []
    total_tokens = 0
    total_chunks = 0
    total_images = 0

    # Process each page result
    for page_num, result in enumerate(results):
        # Add page separator for multi-page documents
        if page_num > 0 and paginate_output:
            all_markdown.append(f"\n\n{page_num}" + "-" * 48 + "\n\n")
            all_html.append(f"\n\n<!-- Page {page_num + 1} -->\n\n")

        all_markdown.append(result.markdown)
        all_html.append(result.html)

        # Ensure page separation in merged output
        if not paginate_output and page_num < len(results) - 1:
            all_markdown.append("\n\n")
            all_html.append("\n\n")

        # Accumulate metadata
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

        # Save extracted images if requested
        if save_images and result.images:
            images_dir = file_output_dir
            images_dir.mkdir(exist_ok=True)

            for img_name, pil_image in result.images.items():
                img_path = images_dir / img_name
                pil_image.save(img_path)

    # Save merged markdown
    markdown_path = file_output_dir / f"{safe_name}.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("".join(all_markdown))

    # Save merged HTML if requested
    if save_html:
        html_path = file_output_dir / f"{safe_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(all_html))

    # Save combined metadata
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

    click.echo(f"  Saved: {markdown_path} ({len(results)} page(s))")


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
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
    help="Page range for PDFs (e.g., '1-5,7,9-12'). Only applicable to PDF files.",
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
    default=settings.VLLM_API_BASE,
    help=f"default: {settings.VLLM_API_BASE!r}",
)
@click.option(
    "--vllm-api-key",
    type=str,
    default=settings.VLLM_API_KEY,
    help=f"default: {settings.VLLM_API_KEY!r}",
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
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process in a batch.",
)
@click.option(
    "--paginate_output",
    is_flag=True,
    default=False,
)
def main(
    input_path: Path,
    output_path: Path,
    method: str,
    page_range: str,
    max_output_tokens: int,
    max_workers: int,
    max_retries: int,
    vllm_api_base: str,
    vllm_api_key: str,
    include_images: bool,
    include_headers_footers: bool,
    save_html: bool,
    batch_size: int,
    paginate_output: bool,
):
    if method == "hf":
        click.echo(
            "When using '--method hf', ensure that the batch size is set correctly.  We will default to batch size of 1."
        )
        if batch_size is None:
            batch_size = 1
    elif method == "vllm":
        if batch_size is None:
            batch_size = 28

    click.echo("Chandra CLI - Starting OCR processing")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Method: {method}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    click.echo(f"\nLoading model with method '{method}'...")
    model = InferenceManager(method=method)
    click.echo("Model loaded successfully.")

    # Get files to process
    files_to_process = get_supported_files(input_path)
    click.echo(f"\nFound {len(files_to_process)} file(s) to process.")

    if not files_to_process:
        click.echo("No supported files found. Exiting.")
        return

    # Process each file
    for file_idx, file_path in enumerate(files_to_process, 1):
        click.echo(
            f"\n[{file_idx}/{len(files_to_process)}] Processing: {file_path.name}"
        )

        try:
            # Load images from file
            config = {"page_range": page_range} if page_range else {}
            images = load_file(str(file_path), config)
            click.echo(f"  Loaded {len(images)} page(s)")

            # Accumulate all results for this document
            all_results = []

            # Process pages in batches
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]

                # Create batch input items
                batch = [
                    BatchInputItem(image=img, prompt_type="ocr_layout")
                    for img in batch_images
                ]

                # Run inference
                click.echo(f"  Processing pages {batch_start + 1}-{batch_end}...")

                # Build kwargs for generate
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

                results = model.generate(batch, **generate_kwargs)
                all_results.extend(results)

            # Save merged output for all pages
            save_merged_output(
                output_path,
                file_path.name,
                all_results,
                save_images=include_images,
                save_html=save_html,
                paginate_output=paginate_output,
            )

            click.echo(f"  Completed: {file_path.name}")

        except Exception as e:
            click.echo(f"  Error processing {file_path.name}: {e}", err=True)
            continue

    click.echo(f"\nProcessing complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
