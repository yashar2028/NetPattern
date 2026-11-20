"""Hardware tiers = bubble sizes (PLAN §11.6, Q8).

A tier says what one job gets from a machine (CPU cores, memory, GPUs) and what it costs per
second, in compute units. A bigger bubble runs faster and costs more: units = rate × seconds.
A worker runs as many jobs at once as its free capacity and job slots allow.

Override the defaults with HARDWARE_TIERS (JSON), e.g.
    {"cpu": {"cpus": 2, "memory_gb": 4, "gpus": 0, "rate": 1.0, "description": "…"}, …}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache

from app.core.config import settings


@dataclass(frozen=True)
class Tier:
    name: str
    cpus: int
    memory_gb: float
    gpus: int
    rate: float  # compute units per second
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_TIERS = {
    "cpu": Tier("cpu", 2, 4, 0, 1.0, "2 CPU cores, 4 GB: small CNNs, profiling, inference"),
    "cpu-large": Tier("cpu-large", 8, 16, 0, 4.0, "8 CPU cores, 16 GB: CPU training"),
    "gpu": Tier("gpu", 4, 16, 1, 20.0, "1 GPU, 4 CPU cores, 16 GB: fine-tuning, large models"),
}


@lru_cache
def tiers() -> dict[str, Tier]:
    if not settings.HARDWARE_TIERS:
        return dict(DEFAULT_TIERS)
    raw = json.loads(settings.HARDWARE_TIERS)
    return {name: Tier(name=name, **values) for name, values in raw.items()}


def rate(tier: str) -> float:
    """Unknown (old) tiers cost one unit per second."""
    found = tiers().get(tier)
    return found.rate if found else 1.0
