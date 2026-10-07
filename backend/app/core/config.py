"""Application settings loaded from environment variables and an optional .env file."""

import socket
from typing import Literal

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
    # (datasets, run directories, checkpoints, environments, caches).
    STORAGE_ROOT: str = "/data"

    # Worker (runtime host)
    WORKER_ID: str = socket.gethostname()
    WORKER_HEARTBEAT_SECONDS: int = 30
    # Hardware tiers this worker serves, comma-separated (cpu, gpu).
    WORKER_TIERS: str = "cpu"
    # Fallback poll interval when no NOTIFY arrives.
    WORKER_POLL_SECONDS: int = 5
    # A running job whose heartbeat is older than this is treated as orphaned.
    JOB_STALE_SECONDS: int = 180
    JOB_MAX_ATTEMPTS: int = 2

    # Sandboxes
    # The engine flake (sources + flake.nix + flake.lock) that environments are built from.
    ENGINE_DIR: str = "/app/engine"
    DEFAULT_ENVIRONMENT_TEMPLATE: str = "pytorch-cpu"
    # none: plain subprocess (trusted development only); bwrap: Linux namespaces.
    SANDBOX_ISOLATION: Literal["none", "bwrap"] = "none"
    SANDBOX_JOB_TIMEOUT_SECONDS: int = 6 * 60 * 60
    SANDBOX_CANCEL_GRACE_SECONDS: int = 30
    SANDBOX_SESSION_IDLE_SECONDS: int = 15 * 60

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def worker_tiers(self) -> list[str]:
        return [tier.strip() for tier in self.WORKER_TIERS.split(",") if tier.strip()]


settings = Settings()
