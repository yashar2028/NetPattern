"""Pinned sandbox environments (PLAN §4.2-4.3).

An environment is a template plus capability packs, built from an immutable snapshot
of the engine flake. Each one gets its own generated flake.nix + flake.lock under
/data/envs/<id>; the lock pins nixpkgs and the engine snapshot by content hash, and
`nix build --out-link` registers a GC root so the environment stays available.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import shutil
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.job import Environment, EnvironmentStatus
from app.sandbox.commands import CommandError, run_command, tail
from app.services import storage

logger = logging.getLogger(__name__)

SNAPSHOT_FILES = ("flake.nix", "flake.lock", "pyproject.toml")
KNOWN_TEMPLATES = ("pytorch-cpu", "pytorch-cuda12")
KNOWN_PACKS = ("detection", "medical")
BUILD_TIMEOUT_SECONDS = 2 * 60 * 60


class EnvironmentError_(RuntimeError):
    """Raised when an environment cannot be prepared."""


@dataclass(frozen=True)
class EngineSnapshot:
    id: str
    version: str
    path: Path


def nix_system() -> str:
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return f"{arch}-linux"


def environment_id(template: str, packs: list[str], snapshot_id: str, system: str) -> str:
    key = json.dumps([template, sorted(packs), snapshot_id, system])
    return "env_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def render_flake(
    template: str, packs: list[str], snapshot: EngineSnapshot, system: str, env_id: str
) -> str:
    pack_list = " ".join(json.dumps(pack) for pack in sorted(packs))
    return (
        "{\n"
        f'  description = "NetPattern sandbox environment {env_id}";\n'
        f'  inputs.engine.url = "path:{snapshot.path}";\n'
        "  outputs = { self, engine }: {\n"
        f"    packages.{system}.default = engine.lib.mkEnvironment {{\n"
        f'      system = "{system}";\n'
        f'      template = "{template}";\n'
        f"      packs = [ {pack_list} ];\n"
        "    };\n"
        "  };\n"
        "}\n"
    )


class EnvironmentManager:
    def __init__(self, engine_dir: Path | None = None) -> None:
        self.engine_dir = Path(engine_dir or settings.ENGINE_DIR)
        self.system = nix_system()
        self._locks: dict[str, asyncio.Lock] = {}

    def engine_version(self) -> str:
        with (self.engine_dir / "pyproject.toml").open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]

    def _engine_files(self) -> list[Path]:
        files = [self.engine_dir / name for name in SNAPSHOT_FILES]
        files += sorted(
            path
            for path in (self.engine_dir / "src").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
        return files

    async def snapshot(self) -> EngineSnapshot:
        """Copy the engine flake into an immutable, content-addressed folder."""
        if not (self.engine_dir / "flake.lock").exists():
            result = await run_command("nix", "flake", "lock", str(self.engine_dir))
            if result.returncode:
                raise EnvironmentError_(f"cannot lock the engine flake:\n{tail(result.stderr)}")

        digest = hashlib.sha256()
        files = self._engine_files()
        for path in files:
            digest.update(str(path.relative_to(self.engine_dir)).encode())
            digest.update(path.read_bytes())
        version = self.engine_version()
        snapshot_id = digest.hexdigest()
        target = storage.snapshots_root() / f"{version}-{snapshot_id[:12]}"
        if not target.exists():
            staging = target.with_name(target.name + ".tmp")
            shutil.rmtree(staging, ignore_errors=True)
            for path in files:
                destination = staging / path.relative_to(self.engine_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            staging.rename(target)
        return EngineSnapshot(snapshot_id, version, target)

    async def ensure(self, db: AsyncSession, template: str, packs: list[str]) -> Environment:
        """Return a ready environment, building it the first time it is needed."""
        if template not in KNOWN_TEMPLATES:
            raise EnvironmentError_(f"unknown environment template '{template}'")
        unknown = sorted(set(packs) - set(KNOWN_PACKS))
        if unknown:
            raise EnvironmentError_(f"unknown capability pack(s): {', '.join(unknown)}")
        packs = sorted(set(packs))
        snapshot = await self.snapshot()
        env_id = environment_id(template, packs, snapshot.id, self.system)

        lock = self._locks.setdefault(env_id, asyncio.Lock())
        async with lock:
            row = await db.get(Environment, env_id)
            if row is not None and row.status == EnvironmentStatus.READY.value and row.store_path:
                if Path(row.store_path).exists():
                    return row
            if row is None:
                row = Environment(
                    id=env_id,
                    template=template,
                    packs=packs,
                    system=self.system,
                    engine_version=snapshot.version,
                    engine_snapshot=snapshot.id,
                    flake_nix=render_flake(template, packs, snapshot, self.system, env_id),
                    status=EnvironmentStatus.BUILDING.value,
                )
                db.add(row)
            row.status, row.error = EnvironmentStatus.BUILDING.value, None
            await db.commit()
            try:
                await self._build(row)
            except (EnvironmentError_, CommandError) as error:
                row.status, row.error = EnvironmentStatus.FAILED.value, str(error)[:4000]
                await db.commit()
                raise EnvironmentError_(str(error)) from error
            row.status, row.built_at = EnvironmentStatus.READY.value, datetime.now(UTC)
            await db.commit()
            return row

    async def _build(self, row: Environment) -> None:
        folder = storage.envs_root() / row.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "flake.nix").write_text(row.flake_nix, encoding="utf-8")
        log = folder / "build.log"

        locked = await run_command("nix", "flake", "lock", str(folder), log_file=log)
        if locked.returncode:
            raise EnvironmentError_(f"nix flake lock failed:\n{tail(locked.stderr)}")
        built = await run_command(
            "nix",
            "build",
            f"{folder}#default",
            "--out-link",
            str(folder / "result"),
            "--print-out-paths",
            timeout=BUILD_TIMEOUT_SECONDS,
            log_file=log,
        )
        if built.returncode:
            raise EnvironmentError_(f"nix build failed:\n{tail(built.stderr)}")

        store_path = built.stdout.strip().splitlines()[-1]
        row.flake_lock = (folder / "flake.lock").read_text(encoding="utf-8")
        row.store_path = store_path
        row.env_hash = Path(store_path).name.split("-", 1)[0]

        schema = await run_command(f"{store_path}/bin/netpattern-runtime", "schema", timeout=300)
        if schema.returncode == 0:
            row.engine_schema = json.loads(schema.stdout)
        else:
            logger.warning(
                "Could not extract the engine schema for %s: %s", row.id, tail(schema.stderr)
            )
