from typing import List

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import scale_to_fit
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def resolve_device(torch_mod, model=None) -> str:
    if settings.TORCH_DEVICE:
        return settings.TORCH_DEVICE
    if model is not None and hasattr(model, "device") and model.device is not None:
        dev_str = str(model.device)
        if dev_str and dev_str != "meta":
            return dev_str
    if torch_mod.cuda.is_available():
        return "cuda"
    if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    bbox_scale: int = settings.BBOX_SCALE,
    **kwargs,
) -> List[GenerationResult]:
    if not batch:
        return []

    import torch

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    conversations = [[process_batch_element(item)] for item in batch]

    inputs = model.processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    target_device = resolve_device(torch, model=model)
    inputs = inputs.to(target_device)

    # Include both <|endoftext|> and <|im_end|> as stop tokens.
    # generation_config only has <|endoftext|>, but the model emits <|im_end|> at turn boundaries.
    eos_token_id = model.generation_config.eos_token_id
    im_end_id = model.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    elif eos_token_id is not None:
        eos_token_id = list(eos_token_id)
    else:
        eos_token_id = []
    if im_end_id is not None and im_end_id not in eos_token_id:
        eos_token_id.append(im_end_id)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_output_tokens, eos_token_id=eos_token_id
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = [
        GenerationResult(raw=out, token_count=len(ids), error=False)
        for out, ids in zip(output_text, generated_ids_trimmed)
    ]
    return results


def process_batch_element(item: BatchInputItem):
    prompt = item.prompt
    prompt_type = item.prompt_type or "ocr_layout"

    if not prompt:
        if prompt_type not in PROMPT_MAPPING:
            raise ValueError(
                f"Unknown prompt_type {prompt_type!r}. Available: {sorted(PROMPT_MAPPING)}"
            )
        prompt = PROMPT_MAPPING[prompt_type]

    content = []
    image = scale_to_fit(item.image)  # Guarantee max size
    content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def load_model():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        raise ImportError(
            "HuggingFace backend requires additional dependencies. "
            'Install with: pip install "chandra-ocr[hf]"'
        )

    device = resolve_device(torch)
    if settings.TORCH_DEVICE:
        device_map = {"": settings.TORCH_DEVICE}
    elif device == "cuda":
        device_map = "auto"
    else:
        # Avoid meta-tensor disk offloading on CPU/MPS
        device_map = {"": device}

    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
    }
    if settings.TORCH_ATTN:
        kwargs["attn_implementation"] = settings.TORCH_ATTN

    model = AutoModelForImageTextToText.from_pretrained(
        settings.MODEL_CHECKPOINT, **kwargs
    )
    model = model.eval()
    processor = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    processor.tokenizer.padding_side = "left"
    model.processor = processor
    return model
