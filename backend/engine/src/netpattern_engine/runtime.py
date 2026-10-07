"""`netpattern-runtime`: the single entry point executed inside a sandbox.

Job contract (shared with the worker):

    request.json in  ->  JSONL events on stdout  ->  result.json out

Phase 0 supports one job kind, `smoke`, which reports the pinned environment.
Training, evaluation, `session` and `serve` modes arrive in Phase 1+.
"""

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from netpattern_engine import __version__
from netpattern_engine.events import emit

LIBRARIES = ("torchvision", "timm", "torchmetrics", "numpy", "PIL", "pydicom", "nibabel")


def environment_info() -> dict[str, Any]:
    """Versions of everything that determines model behavior in this environment."""
    info: dict[str, Any] = {"engine": __version__, "python": platform.python_version()}
    try:
        import torch
    except ImportError:
        info.update(torch=None, cuda_available=False, cuda=None)
    else:
        info.update(
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda=torch.version.cuda,
        )

    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            info[name] = None
        else:
            info[name] = getattr(module, "__version__", "unknown")
    return info


def _write_result(output_path: Path, result: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_job(request_path: Path, output_path: Path) -> int:
    run_id = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request.json must contain a JSON object")
        run_id = request.get("run_id")
        kind = request.get("kind", "smoke")

        emit("run_started", run_id=run_id, kind=kind, engine=__version__)
        if kind != "smoke":
            raise ValueError(f"unsupported job kind: {kind!r}")

        environment = environment_info()
        emit("env_info", run_id=run_id, environment=environment)

        _write_result(
            output_path,
            {"run_id": run_id, "status": "completed", "kind": kind, "environment": environment},
        )
        emit("run_completed", run_id=run_id)
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        emit("run_failed", run_id=run_id, error=message)
        _write_result(output_path, {"run_id": run_id, "status": "failed", "error": message})
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netpattern-runtime",
        description="NetPattern engine runtime (executed inside a sandbox).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    job = commands.add_parser(
        "job", help="run one job: request.json in, JSONL events on stdout, result.json out"
    )
    job.add_argument("--request", type=Path, required=True, help="path to request.json")
    job.add_argument("--output", type=Path, required=True, help="path to write result.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "job":
        return run_job(args.request, args.output)
    return 2


if __name__ == "__main__":
    sys.exit(main())
