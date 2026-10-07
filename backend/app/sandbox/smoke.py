"""Smoke check: build (or reuse) the pinned CPU environment and run one engine job in it.

    docker compose exec worker python -m app.sandbox.smoke

Exercises the whole sandbox boundary without the queue: environment flake + lock,
GC root, isolation (SANDBOX_ISOLATION), streaming events and the result file.
"""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.sandbox.environments import EnvironmentManager
from app.sandbox.isolation import SandboxPolicy
from app.sandbox.runner import run_engine_job
from app.services import storage


async def run() -> int:
    template = settings.DEFAULT_ENVIRONMENT_TEMPLATE
    print(f"Preparing environment '{template}' (first build downloads PyTorch) …", flush=True)
    async with AsyncSessionLocal() as db:
        environment = await EnvironmentManager().ensure(db, template, [])
    print(f"Environment: {environment.id}")
    print(f"Environment hash: {environment.env_hash}")
    print(f"Store path: {environment.store_path}")
    print(f"Isolation: {settings.SANDBOX_ISOLATION}")

    run_id = f"smoke_{uuid4().hex[:12]}"
    run_dir = storage.runs_root() / run_id

    async def on_event(event: dict) -> None:
        print(
            f"  event: {event['type']:<14} {json.dumps(event, sort_keys=True, default=str)[:300]}"
        )

    outcome = await run_engine_job(
        Path(environment.store_path),
        run_dir,
        {"run_id": run_id, "kind": "smoke"},
        on_event,
        lambda: False,
        SandboxPolicy(),
    )
    print(f"Exit code: {outcome.exit_code}")
    print(json.dumps(outcome.result, indent=2))
    ok = outcome.exit_code == 0 and (outcome.result or {}).get("status") == "completed"
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
