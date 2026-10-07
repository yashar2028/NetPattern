"""Seeds and determinism (PLAN §8.2 / §12.2)."""

from __future__ import annotations

import os
import random

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


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            from netpattern_engine.errors import EngineError

            raise EngineError("device 'cuda' was requested but no CUDA GPU is available")
        return torch.device("cuda")
    return torch.device("cpu")
