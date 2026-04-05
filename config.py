"""
Application settings loaded from environment variables / .env file.
All configuration is centralised here — no magic strings elsewhere.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_llm_model: str = Field("gpt-4o", env="OPENAI_LLM_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(1536, env="EMBEDDING_DIM")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    data_dir: str = Field("data", env="DATA_DIR")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(5, env="TOP_K")
    faiss_index_path: str = Field("faiss_index", env="FAISS_INDEX_PATH")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
