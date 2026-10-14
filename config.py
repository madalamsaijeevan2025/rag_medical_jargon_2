from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_host: str = "http://localhost:11434"

    # Cloud fallback when Ollama is not available. OpenAI-compatible endpoint;
    # provider/api_key/api_base_url/api_model/api_embed_model can also be set
    # at runtime via POST /llm/config (Streamlit sidebar).
    llm_provider: str = "local"  # "local" (Ollama) or "api" (cloud, needs key)
    api_key: str = ""
    api_base_url: str = ""       # e.g. https://api.openai.com/v1 (default), OpenRouter, Groq
    api_model: str = ""          # vision-capable chat model, e.g. gpt-4o-mini
    api_embed_model: str = ""    # e.g. text-embedding-3-small
    ollama_timeout: int = 120
    ollama_max_retries: int = 3
    ollama_cb_threshold: int = 5
    ollama_cb_reset_seconds: int = 60

    # Models
    embed_model: str = "nomic-embed-text"
    vlm_model: str = "qwen3-vl:4b"

    # Limits
    max_pages: int = 50
    max_dpi: int = 600
    min_dpi: int = 72

    # Paths
    output_dir: str = "./extracted_pages"
    db_path: str = "./db"
    cache_dir: str = "./cache"
    doc_index_path: str = "./db/doc_index.json"
    dict_db_path: str = "./medical_terms.db"

    # Cleanup
    cleanup_ttl_days: int = 30
    cleanup_schedule_hour: int = 3

    # Logging
    log_level: str = "INFO"

    # Cache
    cache_size_limit_bytes: int = 1_000_000_000  # 1GB


@lru_cache()
def get_settings() -> Settings:
    return Settings()