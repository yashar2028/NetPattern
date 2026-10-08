"""Request and response models for the v1 Platform API."""

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

MAX_SPEC_BYTES = 1_000_000


class Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- auth


def _email(value: str) -> str:
    value = value.strip().lower()
    if "@" not in value or value.startswith("@") or value.endswith("@"):
        raise ValueError("email must be a valid address")
    return value


Email = Annotated[str, Field(max_length=320), AfterValidator(_email)]


class RegisterRequest(BaseModel):
    email: Email
    password: str = Field(min_length=8, max_length=200)
    full_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: Email
    password: str = Field(min_length=1, max_length=200)


class UserOut(Out):
    id: str
    email: str
    full_name: str | None
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ApiKeyOut(Out):
    id: str
    name: str
    lookup: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str  # shown once


# --------------------------------------------------------------------------- sandboxes


class SandboxCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    template: Literal["pytorch-cpu", "pytorch-cuda12"] = "pytorch-cpu"


class SandboxUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SandboxOut(Out):
    id: str
    name: str
    template: str
    engine_version: str | None
    engine_snapshot: str | None
    created_at: datetime


class PipelineCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    spec: dict[str, Any] | None = None


class PipelineVersionCreate(BaseModel):
    spec: dict[str, Any]

    @field_validator("spec")
    @classmethod
    def _shape(cls, spec: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(spec)) > MAX_SPEC_BYTES:
            raise ValueError("the spec is larger than 1 MB")
        if not isinstance(spec.get("task"), dict) or not isinstance(spec.get("nodes"), list):
            raise ValueError("the spec needs a 'task' object and a 'nodes' list")
        return spec


class PipelineOut(Out):
    id: str
    sandbox_id: str
    name: str
    created_at: datetime
    latest_version: int | None = None


class PipelineVersionOut(Out):
    id: str
    pipeline_id: str
    number: int
    created_at: datetime
    spec: dict[str, Any] | None = None


# --------------------------------------------------------------------------- datasets


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DatasetVersionOut(Out):
    id: str
    dataset_id: str
    number: int
    files: int
    size_bytes: int
    created_at: datetime
    top_level: list[str] = []


class DatasetOut(Out):
    id: str
    name: str
    created_at: datetime
    versions: list[DatasetVersionOut] = []


class ProfileRequest(BaseModel):
    task: dict[str, Any]
    # The dataset spec without `root` (taken from the version); `subpath` selects a folder.
    dataset: dict[str, Any]
    subpath: str | None = None


# --------------------------------------------------------------------------- runs


class RunCreate(BaseModel):
    pipeline_version_id: str
    kind: Literal["train", "sanity"] = "train"
    hardware_tier: Literal["cpu", "gpu"] = "cpu"


class RunOut(Out):
    id: str
    kind: str
    status: str
    hardware_tier: str
    sandbox_id: str | None
    pipeline_version_id: str | None
    environment_id: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, Any] | None = None


class RunEventOut(Out):
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: datetime


class UsageOut(BaseModel):
    total_seconds: float
    jobs: int
    by_tier: dict[str, float]
    by_sandbox: dict[str, float]
