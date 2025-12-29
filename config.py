"""
Configuration for KENR GPU Service.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Security
    api_key: str = "dev-key-change-in-production"

    # Hugging Face (for gated models like Llama)
    hf_token: Optional[str] = None

    # Model Configuration
    model_cache_dir: str = "/models"
    default_device: str = "cuda"

    # SAE Configuration
    sae_cache_dir: str = "/models/saes"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Performance
    max_batch_size: int = 8
    max_sequence_length: int = 512

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# Model configurations with their SAE sources
MODEL_CONFIGS = {
    "gpt2-small": {
        "hf_model_id": "gpt2",
        "transformer_lens_name": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.{layer}.hook_resid_pre",
        "n_layers": 12,
        "d_model": 768,
        "requires_auth": False,
    },
    "gemma-2-2b": {
        "hf_model_id": "google/gemma-2-2b",
        "transformer_lens_name": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_{layer}/width_16k/average_l0_71",
        "n_layers": 26,
        "d_model": 2304,
        "requires_auth": True,
    },
    "gemma-2-2b-it": {
        "hf_model_id": "google/gemma-2-2b-it",
        "transformer_lens_name": "gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",  # Use base model SAEs
        "sae_id": "layer_{layer}/width_16k/average_l0_71",
        "n_layers": 26,
        "d_model": 2304,
        "requires_auth": True,
    },
    "llama-3.1-8b": {
        "hf_model_id": "meta-llama/Llama-3.1-8B",
        "transformer_lens_name": "meta-llama/Llama-3.1-8B",
        "sae_release": "fnlp/Llama-Scope",
        "sae_id": "Llama-3.1-8B-Base-Residual-Stream-32k/layer_{layer}",
        "n_layers": 32,
        "d_model": 4096,
        "requires_auth": True,
    },
}
