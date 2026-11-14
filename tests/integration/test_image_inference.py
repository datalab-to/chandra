import pytest

from chandra.model import InferenceManager, BatchInputItem


def test_inference_image(simple_text_image):
    pytest.importorskip("torch")
    manager = InferenceManager(method="hf")
    batch = [
        BatchInputItem(
            image=simple_text_image,
            prompt_type="ocr_layout",
        )
    ]
    outputs = manager.generate(batch, max_output_tokens=128)
    assert len(outputs) == 1
    output = outputs[0]
    assert "Hello, World!" in output.markdown

    chunks = output.chunks
    assert len(chunks) == 1
