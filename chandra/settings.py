from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 192
    MIN_IMAGE_DIM: int = 1024
    MODEL_CHECKPOINT: str = "datalab-to/chandra"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 8192
    TORCH_ATTN: str | None = None

    # vLLM server settings
    VLLM_API_KEY: str = "EMPTY"
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "chandra"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6

    # Transformers settings
    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @computed_field
    @property
    def TORCH_DTYPE(self) -> torch.dtype:
        return torch.bfloat16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
