import os

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Paths & rendering
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536
    # Page images sent to the model: at most this many pixels, sides on this
    # grid. The ViT's cost grows with the square of the pixel count, and the
    # processor snaps images to a 32-px grid (16-px patches, 2x2 merge).
    IMAGE_MAX_PIXELS: int = 3072 * 2048
    IMAGE_GRID_SIZE: int = 28
    MODEL_CHECKPOINT: str = "datalab-to/chandra-ocr-2"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 12384
    TORCH_ATTN: str | None = None
    BBOX_SCALE: int = 1000

    # vLLM server settings
    VLLM_API_KEY: str = "EMPTY"
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "chandra"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6

    model_config = SettingsConfigDict(
        env_file=find_dotenv("local.env"),
        extra="ignore",
    )


settings = Settings()
