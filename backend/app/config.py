from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag_tutor:rag_tutor@localhost:5432/rag_tutor"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 256
    embedding_workers: int = 3
    chat_model: str = "gpt-4o-mini"
    enable_query_rewrite: bool = True
    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k: int = 5
    max_upload_mb: int = 25
    media_dir: str = "media"
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    log_level: str = "INFO"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
