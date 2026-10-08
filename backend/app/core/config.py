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

    # Auth: signs access tokens. Required, no default.
    AUTH_JWT_SECRET: str
    AUTH_ACCESS_TOKEN_EXPIRE_MINUTES: int = 7 * 24 * 60

    # Dataset uploads (zip archives), sized for real test datasets (PLAN D20).
    MAX_UPLOAD_BYTES: int = 2 * 1024**3
    MAX_EXTRACTED_BYTES: int = 8 * 1024**3
    MAX_UPLOAD_FILES: int = 200_000

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
    # Internal API for editor sessions; reachable only on the compose network.
    WORKER_INTERNAL_PORT: int = 9000
    WORKER_INTERNAL_URL: str = "http://worker:9000"
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
