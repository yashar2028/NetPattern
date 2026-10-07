"""Application settings loaded from environment variables and an optional .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration shared by the API and the worker.

    Secrets have no defaults: a missing value fails at startup instead of
    silently running with a known credential.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "NetPattern"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "info"

    DATABASE_URL: str

    # Comma-separated list of browser origins allowed to call the API.
    CORS_ORIGINS: str = "http://localhost:5173"

    # Root of the storage volume shared by the API and the worker
    # (datasets, run directories, checkpoints, models).
    STORAGE_ROOT: str = "/data"

    WORKER_HEARTBEAT_SECONDS: int = 30

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
