from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # LLM
    llm_provider: Literal["groq", "openai", "anthropic", "ollama"] = "groq"
    llm_model: str = "compound"
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Local HF CLIP: text + image same space for all modalities (env: HF_CLIP_MODEL_ID)
    hf_clip_model_id: str = "openai/clip-vit-base-patch32"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # Storage
    upload_dir: str = "/app/uploads"
    max_upload_mb: int = 500

    class Config:
        env_file = ".env"


settings = Settings()