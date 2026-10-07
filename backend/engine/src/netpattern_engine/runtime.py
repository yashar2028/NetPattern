"""`netpattern-runtime`: the single entry point executed inside a sandbox.

    job      request.json in -> JSONL events on stdout -> result.json out
    session  JSON-RPC 2.0 over stdin/stdout (editor features)
    schema   print the engine's JSON Schema and capabilities
    requirements --request FILE   print the capability packs a pipeline needs

Job kinds: smoke, train, sanity, prefetch, profile, make_sample_dataset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from netpattern_engine import __version__, events
from netpattern_engine.errors import EngineError, MissingPackError, SpecError
from netpattern_engine.events import emit

LIBRARIES = (
    "torchvision",
    "timm",
    "torchmetrics",
    "numpy",
    "PIL",
    "pydicom",
    "nibabel",
    "pycocotools",
)
# The worker creates this file next to result.json to ask the job to stop.
CANCEL_FILE = "cancel.request"


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
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def _resolve(request: dict[str, Any]):
    from netpattern_engine.spec import resolve_pipeline

    if "pipeline" not in request:
        raise EngineError("this job kind needs a 'pipeline'")
    return resolve_pipeline(request["pipeline"])


def _run_kind(kind: str, request: dict[str, Any], run_dir: Path, cancel: Any) -> dict[str, Any]:
    if kind == "smoke":
        environment = environment_info()
        emit("env_info", environment=environment)
        return {"status": "completed", "environment": environment}

    if kind == "train":
        from netpattern_engine.pipeline import train

        return train(_resolve(request), run_dir, emit, cancel)
    if kind == "sanity":
        from netpattern_engine.pipeline import sanity

        steps = int(request.get("options", {}).get("steps", 60))
        return sanity(_resolve(request), emit, steps)
    if kind == "prefetch":
        from netpattern_engine.pipeline import prefetch

        return prefetch(_resolve(request), emit)
    if kind == "profile":
        from pydantic import TypeAdapter

        from netpattern_engine.data.profile import profile_dataset
        from netpattern_engine.spec import DatasetSpec, TaskSpec

        data = TypeAdapter(DatasetSpec).validate_python(request.get("dataset"))
        task = TaskSpec.model_validate(request.get("task"))
        return {"status": "completed", "profile": profile_dataset(data, task, emit)}
    if kind == "make_sample_dataset":
        from netpattern_engine.data import sample_files

        options = request.get("options", {})
        out = options.get("out")
        if not out:
            raise EngineError("make_sample_dataset needs options.out")
        which = options.get("type", "nifti")
        if which == "nifti":
            paths = sample_files.write_nifti_dataset(out, volumes=int(options.get("volumes", 12)))
        elif which == "dicom":
            paths = {"classification": sample_files.write_dicom_series(out)}
        else:
            raise EngineError(f"unknown sample dataset type '{which}'")
        emit("dataset_written", paths=paths)
        return {"status": "completed", "paths": paths}
    raise EngineError(f"unsupported job kind: {kind!r}")


def run_job(request_path: Path, output_path: Path) -> int:
    from netpattern_engine.training.trainer import CancelToken

    cancel = CancelToken(output_path.parent / CANCEL_FILE)
    signal.signal(signal.SIGTERM, lambda *_: cancel.cancel())
    run_id = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise EngineError("request.json must contain a JSON object")
        run_id = request.get("run_id")
        kind = request.get("kind", "smoke")
        emit("run_started", run_id=run_id, kind=kind, engine=__version__)
        result = _run_kind(kind, request, output_path.parent, cancel)
        result = {"run_id": run_id, "kind": kind, **result}
        if kind != "smoke":
            result.setdefault("environment", environment_info())
        _write_result(output_path, result)
        emit(
            "run_cancelled" if result.get("status") == "cancelled" else "run_completed",
            run_id=run_id,
        )
        return 0
    except Exception as error:  # noqa: BLE001 - every failure must produce a result file
        failure: dict[str, Any] = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        if isinstance(error, SpecError):
            failure["issues"] = [issue.to_dict() for issue in error.issues]
        if isinstance(error, MissingPackError):
            failure["missing_pack"] = error.pack
        if not isinstance(error, EngineError):
            traceback.print_exc(file=sys.stderr)
        emit(
            "run_failed",
            run_id=run_id,
            error=failure["error"],
            **{k: v for k, v in failure.items() if k in ("issues", "missing_pack")},
        )
        _write_result(output_path, failure)
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

    commands.add_parser("session", help="JSON-RPC 2.0 over stdin/stdout for the editor")
    commands.add_parser("schema", help="print the engine's JSON Schema and capabilities")

    requirements = commands.add_parser(
        "requirements", help="print the capability packs a pipeline needs"
    )
    requirements.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Keep stdout for machine-readable output only; library prints go to stderr.
    stdout = sys.stdout
    events.bind(stdout)
    sys.stdout = sys.stderr
    try:
        if args.command == "job":
            return run_job(args.request, args.output)
        if args.command == "session":
            from netpattern_engine.session import serve

            return serve(sys.stdin, stdout)
        if args.command == "schema":
            from netpattern_engine.schema import schema_document

            stdout.write(json.dumps(schema_document()) + "\n")
            return 0
        if args.command == "requirements":
            from netpattern_engine.pipeline import required_packs
            from netpattern_engine.spec import resolve_pipeline

            request = json.loads(args.request.read_text(encoding="utf-8"))
            try:
                packs = (
                    required_packs(resolve_pipeline(request["pipeline"]))
                    if "pipeline" in request
                    else []
                )
            except SpecError as error:
                stdout.write(
                    json.dumps({"packs": [], "error": str(error), **error.to_dict()}) + "\n"
                )
                return 1
            stdout.write(json.dumps({"packs": packs}) + "\n")
            return 0
        return 2
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    sys.exit(main())
