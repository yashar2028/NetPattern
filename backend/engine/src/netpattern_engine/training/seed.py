"""Seeds and determinism (PLAN §8.2 / §12.2)."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if deterministic:
        # Must be set before CUDA initializes; the worker also sets it for every sandbox.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding derived from the torch seed of the worker."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def rng_state(generator: torch.Generator | None) -> dict[str, Any]:
    """Every random number generator training uses, in a form checkpoints can hold
    (tensors and plain Python values only, so they load with weights_only=True)."""
    name, keys, position, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": [str(name), keys.tolist(), int(position), int(has_gauss), float(cached)],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["loader"] = generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator | None) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and "loader" in state:
        generator.set_state(state["loader"])


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            from netpattern_engine.errors import EngineError

            raise EngineError("device 'cuda' was requested but no CUDA GPU is available")
        return torch.device("cuda")
    return torch.device("cpu")
