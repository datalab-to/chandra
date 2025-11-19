from dotenv import find_dotenv
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536
    MODEL_CHECKPOINT: str = "datalab-to/chandra"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 12384
    TORCH_ATTN: str | None = None
    BBOX_SCALE: int = 1024

    # vLLM server settings
    VLLM_API_KEY: str = "EMPTY"
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "chandra"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
