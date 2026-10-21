"""Application settings loaded from environment variables and an optional .env file."""

import socket
from typing import Literal

from pydantic import field_validator
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
    # Hardware tiers this worker serves, comma-separated (see app/core/tiers.py).
    WORKER_TIERS: str = "cpu"
    # Machine capacity shared by the jobs running at once (PLAN §11.6). Unset = detected.
    WORKER_CPUS: int | None = None
    WORKER_MEMORY_GB: float | None = None
    WORKER_GPUS: int = 0
    # Job slots: how many jobs may run at once (a bigger machine can run more).
    WORKER_MAX_JOBS: int = 1
    # Dedicated machine: only this account's jobs (user email). Unset = shared worker.
    WORKER_ACCOUNT: str | None = None
    # Bubble sizes and rates as JSON; empty = the defaults in app/core/tiers.py.
    HARDWARE_TIERS: str | None = None
    # Fallback poll interval when no NOTIFY arrives.
    WORKER_POLL_SECONDS: int = 5
    # Internal API for editor sessions; reachable only on the compose network.
    WORKER_INTERNAL_PORT: int = 9000
    WORKER_INTERNAL_URL: str = "http://worker:9000"
    # How the API reaches this worker; registered in the workers table (default: the above).
    WORKER_ADVERTISE_URL: str | None = None
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
    # Code block layers run user Python inside the sandbox. Keep off for external users until the
    # hardened isolation of PLAN §4.5 is in place.
    CODE_BLOCKS_ENABLED: bool = True

    # Deployments and the inference API (PLAN §8.5)
    DEPLOYMENT_IDLE_SECONDS: int = 10 * 60  # serverless model processes stop after this
    INFERENCE_MAX_FILES: int = 16  # images per predict call
    INFERENCE_MAX_BYTES: int = 25 * 1024**2  # per predict call
    INFERENCE_RATE_LIMIT_PER_MINUTE: int = 120  # per API key (or user), per backend process

    @field_validator(
        "WORKER_CPUS",
        "WORKER_MEMORY_GB",
        "WORKER_ACCOUNT",
        "WORKER_ADVERTISE_URL",
        "HARDWARE_TIERS",
        mode="before",
    )
    @classmethod
    def _empty_is_unset(cls, value):
        # docker compose passes unset optional variables as empty strings.
        return None if value == "" else value

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def worker_tiers(self) -> list[str]:
        return [tier.strip() for tier in self.WORKER_TIERS.split(",") if tier.strip()]


settings = Settings()
