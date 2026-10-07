"""Phase 0 smoke check: build the pinned CPU environment and run one engine job in it.

Run inside the worker container:

    docker compose exec worker python -m app.sandbox.smoke

It exercises the sandbox contract end to end: `request.json` in, JSONL events
streamed line by line from stdout, `result.json` out. The engine process gets an
allow-listed environment only; it does not inherit the worker's variables.
"""

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from app.core.config import settings

# flake.nix lives at the backend root (/app in the container).
FLAKE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT = "env-cpu"


def build_environment(attribute: str = DEFAULT_ENVIRONMENT) -> Path:
    """Build (or fetch from cache) a pinned environment and return its store path.

    Nix progress goes straight to the terminal (stderr); only the store path is captured.
    """
    completed = subprocess.run(
        ["nix", "build", f"{FLAKE_DIR}#{attribute}", "--no-link", "--print-out-paths"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(completed.stdout.strip().splitlines()[-1])


def sandbox_environment(env_path: Path, run_dir: Path) -> dict[str, str]:
    """Allow-listed variables for the sandbox process (nothing inherited from the worker)."""
    return {
        "PATH": str(env_path / "bin"),
        "HOME": str(run_dir),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
    }


def run_job(env_path: Path, run_dir: Path, request: dict) -> tuple[int, list[dict], dict]:
    request_file = run_dir / "request.json"
    result_file = run_dir / "result.json"
    request_file.write_text(json.dumps(request, indent=2), encoding="utf-8")

    events: list[dict] = []
    with subprocess.Popen(
        [
            str(env_path / "bin" / "netpattern-runtime"),
            "job",
            "--request",
            str(request_file),
            "--output",
            str(result_file),
        ],
        cwd=run_dir,
        env=sandbox_environment(env_path, run_dir),
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            event = json.loads(line)
            events.append(event)
            print(f"  event: {event['type']:<14} {json.dumps(event, sort_keys=True)}")
        return_code = process.wait()

    result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
    return return_code, events, result


def main() -> int:
    print(f"Building pinned environment '{DEFAULT_ENVIRONMENT}' from {FLAKE_DIR}/flake.nix …")
    env_path = build_environment()
    # Store paths look like /nix/store/<hash>-<name>; the hash identifies the environment.
    environment_hash = env_path.name.split("-", 1)[0]
    print(f"Environment: {env_path}")
    print(f"Environment hash: {environment_hash}")

    run_id = f"smoke_{uuid4().hex[:12]}"
    run_dir = Path(settings.STORAGE_ROOT) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running job {run_id} in {run_dir}")

    return_code, events, result = run_job(env_path, run_dir, {"run_id": run_id, "kind": "smoke"})

    print(f"Exit code: {return_code}")
    print("Result:")
    print(json.dumps(result, indent=2))
    ok = return_code == 0 and result.get("status") == "completed"
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
