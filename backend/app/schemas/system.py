"""Response schemas for system endpoints (health, service info)."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    version: str


class InfoResponse(BaseModel):
    name: str
    version: str
    api_version: str
    environment: str
