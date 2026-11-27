from typing import List

from chandra.model.hf import generate_hf, load_model
from chandra.model.schema import BatchInputItem, BatchOutputItem, GenerationResult
from chandra.model.vllm import generate_vllm
from chandra.output import extract_images, parse_chunks, parse_html, parse_markdown
from chandra.settings import settings


class InferenceManager:
    def __init__(self, method: str = "vllm"):
        if method not in ("vllm", "hf", "cuda"):
            raise ValueError(f"method must be 'vllm', 'hf' or 'cuda', got {method!r}")
        self.method = method

        if method == "hf":
            self.model = load_model()
        elif method == "cuda":  # the optional chandra-cuda engine (cuda/)
            from chandra_cuda.ocr import CudaGenerator

            self.model = CudaGenerator()
        else:
            self.model = None

    def generate(
        self, batch: List[BatchInputItem], max_output_tokens=None, **kwargs
    ) -> List[BatchOutputItem]:
        if not batch:
            return []

        output_kwargs = {}
        if "include_images" in kwargs:
            output_kwargs["include_images"] = kwargs.pop("include_images")
        if "include_headers_footers" in kwargs:
            output_kwargs["include_headers_footers"] = kwargs.pop(
                "include_headers_footers"
            )
        bbox_scale = kwargs.pop("bbox_scale", settings.BBOX_SCALE)
        vllm_api_base = kwargs.pop("vllm_api_base", settings.VLLM_API_BASE)
        vllm_api_key = kwargs.pop("vllm_api_key", settings.VLLM_API_KEY)
        vllm_model_name = kwargs.pop("vllm_model_name", settings.VLLM_MODEL_NAME)

        if self.method == "vllm":
            results = generate_vllm(
                batch,
                max_output_tokens=max_output_tokens,
                bbox_scale=bbox_scale,
                vllm_api_base=vllm_api_base,
                vllm_api_key=vllm_api_key,
                vllm_model_name=vllm_model_name,
                **kwargs,
            )
        elif self.method == "cuda":
            results = self.model.generate(batch, max_output_tokens=max_output_tokens)
        else:
            results = generate_hf(
                batch,
                self.model,
                max_output_tokens=max_output_tokens,
                bbox_scale=bbox_scale,
                **kwargs,
            )

        output = []
        for result, input_item in zip(results, batch):
            chunks = parse_chunks(result.raw, input_item.image, bbox_scale=bbox_scale)
            output.append(
                BatchOutputItem(
                    markdown=parse_markdown(result.raw, **output_kwargs),
                    html=parse_html(result.raw, **output_kwargs),
                    chunks=chunks,
                    raw=result.raw,
                    page_box=[0, 0, input_item.image.width, input_item.image.height],
                    token_count=result.token_count,
                    images=extract_images(result.raw, chunks, input_item.image),
                    error=result.error,
                )
            )
        return output


__all__ = [
    "BatchInputItem",
    "BatchOutputItem",
    "GenerationResult",
    "InferenceManager",
]
