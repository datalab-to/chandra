import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import List

from PIL import Image
from openai import OpenAI

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import scale_to_fit, detect_repeat_token
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_vllm(
    batch: List[BatchInputItem],
    max_output_tokens: int = None,
    max_retries: int = None,
    max_workers: int | None = None,
    custom_headers: dict | None = None,
    max_failure_retries: int | None = None,
    bbox_scale: int = settings.BBOX_SCALE,
    vllm_api_base: str = settings.VLLM_API_BASE,
) -> List[GenerationResult]:
    client = OpenAI(
        api_key=settings.VLLM_API_KEY,
        base_url=vllm_api_base,
        default_headers=custom_headers,
    )
    model_name = settings.VLLM_MODEL_NAME

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    if max_workers is None:
        max_workers = min(64, len(batch))

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if model_name is None:
        models = client.models.list()
        model_name = models.data[0].id

    def _generate(
        item: BatchInputItem, temperature: float = 0, top_p: float = 0.1
    ) -> GenerationResult:
        prompt = item.prompt
        if not prompt:
            prompt = PROMPT_MAPPING[item.prompt_type].replace(
                "{bbox_scale}", str(bbox_scale)
            )

        content = []
        image = scale_to_fit(item.image)
        image_b64 = image_to_base64(image)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

        content.append({"type": "text", "text": prompt})

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw = completion.choices[0].message.content
            result = GenerationResult(
                raw=raw,
                token_count=completion.usage.completion_tokens,
                error=False,
            )
        except Exception as e:
            print(f"Error during VLLM generation: {e}")
            return GenerationResult(raw="", token_count=0, error=True)

        return result

    def process_item(item, max_retries, max_failure_retries=None):
        result = _generate(item)
        retries = 0

        while _should_retry(result, retries, max_retries, max_failure_retries):
            result = _generate(item, temperature=0.3, top_p=0.95)
            retries += 1

        return result

    def _should_retry(result, retries, max_retries, max_failure_retries):
        has_repeat = detect_repeat_token(result.raw) or (
            len(result.raw) > 50 and detect_repeat_token(result.raw, cut_from_end=50)
        )

        if retries < max_retries and has_repeat:
            print(
                f"Detected repeat token, retrying generation (attempt {retries + 1})..."
            )
            return True

        if retries < max_retries and result.error:
            print(
                f"Detected vllm error, retrying generation (attempt {retries + 1})..."
            )
            time.sleep(2 * (retries + 1))  # Sleeping can help under load
            return True

        if (
            result.error
            and max_failure_retries is not None
            and retries < max_failure_retries
        ):
            print(
                f"Detected vllm error, retrying generation (attempt {retries + 1})..."
            )
            time.sleep(2 * (retries + 1))  # Sleeping can help under load
            return True

        return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                process_item, batch, repeat(max_retries), repeat(max_failure_retries)
            )
        )

    return results
