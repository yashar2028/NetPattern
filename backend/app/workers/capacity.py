"""A worker's machine capacity, split into bubbles for the jobs running at once (PLAN §11.6).

Each running job gets specific CPU cores (pinned with taskset), specific GPUs
(CUDA_VISIBLE_DEVICES) and a memory budget. Memory is counted for scheduling but not hard-limited
yet (that needs cgroups; see PLAN Phase 7 notes).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.tiers import Tier


@dataclass
class Allocation:
    tier: str
    cpus: list[int]
    gpus: list[int]
    memory_gb: float


def detect_memory_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return 4.0


@dataclass
class Capacity:
    cpus: list[int]
    memory_gb: float
    gpus: list[int]
    max_jobs: int
    running: list[Allocation] = field(default_factory=list)

    @classmethod
    def detect(
        cls, cpus: int | None, memory_gb: float | None, gpus: int, max_jobs: int
    ) -> Capacity:
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        if not available:
            available = list(range(os.cpu_count() or 1))
        return cls(
            cpus=available[: cpus or len(available)],
            memory_gb=memory_gb or detect_memory_gb(),
            gpus=list(range(gpus)),
            max_jobs=max(1, max_jobs),
        )

    def _free(self) -> tuple[list[int], list[int], float]:
        used_cpus = {c for a in self.running for c in a.cpus}
        used_gpus = {g for a in self.running for g in a.gpus}
        memory = self.memory_gb - sum(a.memory_gb for a in self.running)
        return (
            [c for c in self.cpus if c not in used_cpus],
            [g for g in self.gpus if g not in used_gpus],
            memory,
        )

    def can_ever_run(self, tier: Tier) -> bool:
        """Whether an empty machine fits this bubble (otherwise the worker does not serve it)."""
        return tier.cpus <= len(self.cpus) and tier.gpus <= len(self.gpus)

    def fits(self, tier: Tier) -> bool:
        if len(self.running) >= self.max_jobs:
            return False
        cpus, gpus, memory = self._free()
        # Memory is a soft budget: a lone job may use the whole machine.
        memory_ok = not self.running or tier.memory_gb <= memory
        return tier.cpus <= len(cpus) and tier.gpus <= len(gpus) and memory_ok

    def allocate(self, tier: Tier) -> Allocation:
        cpus, gpus, _ = self._free()
        allocation = Allocation(tier.name, cpus[: tier.cpus], gpus[: tier.gpus], tier.memory_gb)
        self.running.append(allocation)
        return allocation

    def release(self, allocation: Allocation) -> None:
        if allocation in self.running:
            self.running.remove(allocation)

    def describe(self) -> dict:
        return {
            "cpus": len(self.cpus),
            "memory_gb": self.memory_gb,
            "gpus": len(self.gpus),
            "max_jobs": self.max_jobs,
            "running": len(self.running),
        }
