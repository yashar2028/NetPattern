"""Process isolation around sandbox processes (PLAN §4.5).

Nix pins the software; this layer contains the process. With bubblewrap the engine
gets its own user/PID/IPC/network namespaces, the Nix store and datasets read-only,
and write access only to its run directory. No network unless explicitly allowed
(pretrained weights are prefetched in a separate, network-enabled step).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings

# Read-only system files the sandbox needs when it is allowed to download weights.
NETWORK_FILES = ("/etc/resolv.conf", "/etc/hosts", "/etc/ssl/certs", "/etc/nsswitch.conf")
NOBODY = "65534"


@dataclass
class SandboxPolicy:
    read_only: list[Path] = field(default_factory=list)
    read_write: list[Path] = field(default_factory=list)
    network: bool = False
    working_dir: Path | None = None
    # The job's bubble on the machine (app.workers.capacity.Allocation); None = no limits.
    resources: Any = None


def _pin(command: list[str], policy: SandboxPolicy) -> list[str]:
    """Pin the process (and everything it starts) to the job's CPU cores."""
    cpus = getattr(policy.resources, "cpus", None)
    taskset = shutil.which("taskset")
    if not cpus or taskset is None:
        return command
    return [taskset, "--cpu-list", ",".join(str(c) for c in cpus), *command]


def wrap(command: list[str], policy: SandboxPolicy, mode: str | None = None) -> list[str]:
    mode = mode or settings.SANDBOX_ISOLATION
    if mode == "none":
        return _pin(command, policy)
    if mode != "bwrap":
        raise ValueError(f"unknown isolation mode '{mode}'")

    # Absolute path: the sandbox's own PATH only contains the environment's bin folder.
    args = [
        shutil.which("bwrap") or "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--uid",
        NOBODY,
        "--gid",
        NOBODY,
        "--ro-bind",
        "/nix/store",
        "/nix/store",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        # PyTorch data loaders share batches through /dev/shm.
        "--tmpfs",
        "/dev/shm",
    ]
    if policy.network:
        args.append("--share-net")
        for path in NETWORK_FILES:
            if Path(path).exists():
                args += ["--ro-bind", path, path]
    for path in policy.read_only:
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    for path in policy.read_write:
        path.mkdir(parents=True, exist_ok=True)
        args += ["--bind", str(path), str(path)]
    if policy.working_dir is not None:
        args += ["--chdir", str(policy.working_dir)]
    return _pin([*args, "--", *command], policy)


def sandbox_env(env_path: Path, home: Path, network: bool, resources: Any = None) -> dict[str, str]:
    """Allow-listed variables: nothing is inherited from the worker (no secrets)."""
    cache = Path(settings.STORAGE_ROOT) / "cache"
    variables = {
        "PATH": f"{env_path}/bin",
        "HOME": str(home),
        # There is no /etc/passwd inside the sandbox; libraries ask for a user name.
        "USER": "sandbox",
        "LOGNAME": "sandbox",
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_HOME": str(cache / "torch"),
        "HF_HOME": str(cache / "huggingface"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "MPLCONFIGDIR": str(home / ".matplotlib"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    if not network:
        variables["HF_HUB_OFFLINE"] = "1"
    if settings.CODE_BLOCKS_ENABLED:
        variables["NETPATTERN_ALLOW_CODE"] = "1"
    if resources is not None:
        threads = str(max(1, len(resources.cpus)))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            variables[name] = threads
        # Only the job's own GPUs are visible ("" hides all GPUs from CPU bubbles).
        variables["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in resources.gpus)
    return variables
