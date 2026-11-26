import base64
from concurrent.futures import ThreadPoolExecutor
import io
from itertools import repeat
import time
from typing import List

from PIL import Image
from openai import OpenAI

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import detect_repeat_token, scale_to_fit
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


# A generation that falls into a loop otherwise runs to max_tokens (12K+
# tokens) before the repeat detector sees the finished text and retries.
# While streaming, the detector runs every REPEAT_CHECK_EVERY chunks (about one
# token each); once it has fired on REPEAT_STOP_AFTER checks in a row (~32
# tokens) the request is stopped and retried at once. Legitimate repeats, such
# as a row of empty table cells, end long before that.
REPEAT_CHECK_EVERY = 8
REPEAT_STOP_AFTER = 4
# detect_repeat_token never looks back further than ~1.3K characters.
REPEAT_TAIL_CHARS = 2048


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def _normalize_api_base(api_base: str) -> str:
    base = (api_base or settings.VLLM_API_BASE).strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def generate_vllm(
    batch: List[BatchInputItem],
    max_output_tokens: int | None = None,
    max_retries: int | None = None,
    max_workers: int | None = None,
    custom_headers: dict | None = None,
    max_failure_retries: int | None = None,
    bbox_scale: int = settings.BBOX_SCALE,
    vllm_api_base: str = settings.VLLM_API_BASE,
    vllm_api_key: str = settings.VLLM_API_KEY,
    vllm_model_name: str | None = settings.VLLM_MODEL_NAME,
    temperature: float = 0.0,
    top_p: float = 0.1,
    stop_on_repeat: bool = True,
) -> List[GenerationResult]:
    if not batch:
        return []

    normalized_base = _normalize_api_base(vllm_api_base)
    client = OpenAI(
        api_key=vllm_api_key or settings.VLLM_API_KEY,
        base_url=normalized_base,
        default_headers=custom_headers,
    )
    model_name = vllm_model_name if vllm_model_name is not None else settings.VLLM_MODEL_NAME

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    if max_workers is None:
        max_workers = max(1, min(64, len(batch)))
    else:
        max_workers = max(1, int(max_workers))

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if not model_name:
        models = client.models.list()
        model_name = models.data[0].id

    def _generate(item: BatchInputItem, temperature, top_p) -> GenerationResult:
        prompt = item.prompt
        prompt_type = item.prompt_type or "ocr_layout"
        if not prompt:
            if prompt_type not in PROMPT_MAPPING:
                raise ValueError(
                    f"Unknown prompt_type {prompt_type!r}. Available: {sorted(PROMPT_MAPPING)}"
                )
            prompt = PROMPT_MAPPING[prompt_type]

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
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stream_options={"include_usage": True},
            )
            raw, usage, n_chunks, hits, stopped = "", None, 0, 0, False
            for chunk in stream:
                if getattr(chunk, "error", None):
                    raise RuntimeError(str(chunk.error))
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices or chunk.choices[0].delta is None:
                    continue
                piece = chunk.choices[0].delta.content
                if not piece:
                    continue
                raw += piece
                n_chunks += 1
                if stop_on_repeat and n_chunks % REPEAT_CHECK_EVERY == 0:
                    if detect_repeat_token(raw[-REPEAT_TAIL_CHARS:]):
                        hits += 1
                        if hits >= REPEAT_STOP_AFTER:
                            stopped = True
                            break
                    else:
                        hits = 0
            if stopped:
                # Closing the connection makes the server abort the request.
                stream.close()
                print(f"Stopped a repeating generation after {n_chunks} tokens")
            if usage is not None and getattr(usage, "completion_tokens", None) is not None:
                token_count = int(usage.completion_tokens)
            else:
                # The server sends usage only with the last chunk; a stopped
                # stream has none, so count chunks (about one token each).
                token_count = n_chunks if stopped else 0
            result = GenerationResult(
                raw=raw,
                token_count=token_count,
                error=False,
            )
        except Exception as e:
            print(f"Error during VLLM generation: {e}")
            return GenerationResult(raw="", token_count=0, error=True)

        return result

    def process_item(item, max_retries, max_failure_retries=None):
        result = _generate(item, temperature=temperature, top_p=top_p)
        retries = 0

        while _should_retry(result, retries, max_retries, max_failure_retries):
            retry_temperature = min(temperature + 0.2 * (retries + 1), 0.8)
            result = _generate(item, temperature=retry_temperature, top_p=0.95)
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
