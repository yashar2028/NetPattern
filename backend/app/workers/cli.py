"""Worker CLI: submit and follow sandbox jobs until the Phase 2 API exists.

python -m app.workers.cli submit examples/train_custom_cnn_synthetic.json --wait
python -m app.workers.cli follow job_…
python -m app.workers.cli status job_…
python -m app.workers.cli cancel job_…
python -m app.workers.cli list
python -m app.workers.cli env --packs medical,detection
python -m app.workers.cli verify-env
python -m app.workers.cli session validate_architecture \
    --params-file examples/validate_graph_session.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.job import FINISHED_STATUSES, Job
from app.sandbox.environments import EnvironmentManager
from app.sandbox.isolation import SandboxPolicy
from app.sandbox.session import SessionClient, SessionError
from app.services import storage
from app.workers import queue

FOLLOW_POLL_SECONDS = 1.0


def _format_metrics(metrics: dict[str, Any]) -> str:
    return "  ".join(
        f"{key} {value:.4f}" if isinstance(value, float) else f"{key} {value}"
        for key, value in metrics.items()
        if value is not None
    )


def format_event(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "batch":
        return (
            f"  epoch {payload['epoch']} batch {payload['batch']}/{payload['batches']}  "
            f"loss {payload['loss']:.4f}  lr {payload['lr']:.2e}  "
            f"({payload.get('samples_per_s', 0)} img/s)"
        )
    if event_type == "epoch_end":
        best = "  *best*" if payload.get("best") else ""
        return (
            f"epoch {payload['epoch']}/{payload['epochs']}  {_format_metrics(payload['metrics'])}  "
            f"({payload['duration_s']}s){best}"
        )
    if event_type == "evaluation":
        return f"evaluation [{payload['split']}]  {_format_metrics(payload['metrics'])}"
    if event_type == "phase":
        return f"-- {payload.get('name')}"
    if event_type == "sanity_step":
        return f"  sanity step {payload['step']}/{payload['steps']}  loss {payload['loss']:.4f}"
    details = {k: v for k, v in payload.items() if k not in ("ts", "source")}
    return f"[{event_type}] {json.dumps(details, default=str)[:400]}"


async def follow(job_id: str) -> Job | None:
    seq = 0
    while True:
        async with AsyncSessionLocal() as db:
            for event in await queue.events_after(db, job_id, seq):
                seq = event.seq
                print(format_event(event.type, event.payload), flush=True)
            job = await db.get(Job, job_id, populate_existing=True)
        if job is None:
            print(f"no job {job_id}")
            return None
        if job.status in FINISHED_STATUSES:
            print(f"== {job.id}: {job.status.value}" + (f" — {job.error}" if job.error else ""))
            return job
        await asyncio.sleep(FOLLOW_POLL_SECONDS)


def _summary(job: Job) -> dict[str, Any]:
    result = job.result or {}
    summary: dict[str, Any] = {
        "id": job.id,
        "kind": job.kind,
        "status": job.status.value,
        "environment_id": job.environment_id,
        "env_hash": result.get("env_hash"),
        "run_dir": job.run_dir,
        "error": job.error,
    }
    for key in (
        "evaluation",
        "training",
        "passed",
        "initial_loss",
        "final_loss",
        "profile",
        "paths",
        "checkpoints",
    ):
        if key in result:
            value = result[key]
            if key == "training":
                value = {
                    k: value.get(k)
                    for k in (
                        "epochs_run",
                        "stopped_reason",
                        "best_epoch",
                        "monitor",
                        "best_value",
                        "duration_s",
                    )
                }
            if key == "evaluation":
                value = {split: data.get("metrics") for split, data in value.items()}
            summary[key] = value
    return summary


async def cmd_submit(args: argparse.Namespace) -> int:
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    kind = args.kind or request.pop("kind", None)
    if not kind:
        print("the request file has no 'kind'; pass --kind", file=sys.stderr)
        return 2
    request.pop("kind", None)
    environment = {
        "template": args.template or request.pop("template", settings.DEFAULT_ENVIRONMENT_TEMPLATE)
    }
    packs = [p for p in (args.packs or "").split(",") if p] or request.pop("packs", [])
    if packs:
        environment["packs"] = packs
    tier = args.tier or request.pop("hardware_tier", "cpu")
    async with AsyncSessionLocal() as db:
        job = await queue.enqueue(db, kind, request, environment, tier, args.priority)
    print(f"submitted {job.id} ({kind}, tier {tier})", flush=True)
    if not args.wait:
        return 0
    finished = await follow(job.id)
    if finished is not None:
        print(json.dumps(_summary(finished), indent=2, default=str))
    return 0 if finished is not None and finished.status.value == "completed" else 1


async def cmd_follow(args: argparse.Namespace) -> int:
    job = await follow(args.job_id)
    if job is not None:
        print(json.dumps(_summary(job), indent=2, default=str))
    return 0 if job is not None and job.status.value == "completed" else 1


async def cmd_status(args: argparse.Namespace) -> int:
    async with AsyncSessionLocal() as db:
        job = await db.get(Job, args.job_id)
    if job is None:
        print(f"no job {args.job_id}")
        return 1
    print(
        json.dumps(
            _summary(job) if not args.full else {"summary": _summary(job), "result": job.result},
            indent=2,
            default=str,
        )
    )
    return 0


async def cmd_cancel(args: argparse.Namespace) -> int:
    async with AsyncSessionLocal() as db:
        status = await queue.request_cancel(db, args.job_id)
    print(f"{args.job_id}: {status.value if status else 'not found'}")
    return 0 if status else 1


async def cmd_list(args: argparse.Namespace) -> int:
    async with AsyncSessionLocal() as db:
        jobs = (
            (await db.execute(select(Job).order_by(Job.created_at.desc()).limit(args.limit)))
            .scalars()
            .all()
        )
    for job in jobs:
        print(
            f"{job.id}  {job.kind:<20} {job.status.value:<10} {job.hardware_tier:<5} "
            f"{job.created_at:%Y-%m-%d %H:%M:%S}"
        )
    return 0


async def cmd_env(args: argparse.Namespace) -> int:
    packs = [p for p in (args.packs or "").split(",") if p]
    async with AsyncSessionLocal() as db:
        environment = await EnvironmentManager().ensure(db, args.template, packs)
    print(
        json.dumps(
            {
                "id": environment.id,
                "env_hash": environment.env_hash,
                "store_path": environment.store_path,
                "template": environment.template,
                "packs": environment.packs,
                "engine": environment.engine_version,
                "flake": str(storage.envs_root() / environment.id),
            },
            indent=2,
        )
    )
    return 0


async def cmd_verify_env(args: argparse.Namespace) -> int:
    """Re-evaluate an environment from its flake.lock: the store path must not change."""
    from app.models.job import Environment
    from app.sandbox.commands import run_command

    async with AsyncSessionLocal() as db:
        rows = (
            [await db.get(Environment, args.env_id)]
            if args.env_id
            else (await db.execute(select(Environment).where(Environment.status == "ready")))
            .scalars()
            .all()
        )
    failures = 0
    for row in rows:
        if row is None:
            print("no such environment")
            return 1
        folder = storage.envs_root() / row.id
        evaluated = await run_command(
            "nix", "eval", "--raw", f"{folder}#packages.{row.system}.default.outPath", timeout=600
        )
        same = evaluated.returncode == 0 and evaluated.stdout.strip() == row.store_path
        failures += not same
        print(f"{row.id}  {'same store path' if same else 'MISMATCH'}  {row.store_path}")
        if not same:
            print(f"  evaluated: {evaluated.stdout.strip() or evaluated.stderr.strip()[-300:]}")
    return 1 if failures else 0


async def cmd_session(args: argparse.Namespace) -> int:
    params = (
        json.loads(Path(args.params_file).read_text(encoding="utf-8"))
        if args.params_file
        else json.loads(args.params or "{}")
    )
    packs = [p for p in (args.packs or "").split(",") if p]
    async with AsyncSessionLocal() as db:
        environment = await EnvironmentManager().ensure(db, args.template, packs)
    client = SessionClient(
        Path(environment.store_path),
        storage.root() / "sessions" / environment.id,
        SandboxPolicy(read_only=[storage.datasets_root(), storage.cache_root()]),
    )
    await client.start()
    try:
        result = await client.call(args.method, params)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except SessionError as error:
        print(
            json.dumps(
                {"error": str(error), "code": error.code, "data": error.data}, indent=2, default=str
            )
        )
        return 1
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.workers.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="enqueue a job from a request file")
    submit.add_argument(
        "request", help="JSON file with pipeline/dataset/options (and optionally kind)"
    )
    submit.add_argument(
        "--kind", help="train, sanity, profile, prefetch, smoke, make_sample_dataset"
    )
    submit.add_argument("--tier", help="hardware tier (cpu, gpu)")
    submit.add_argument("--template", help="environment template")
    submit.add_argument("--packs", help="extra capability packs, comma-separated")
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--wait", action="store_true", help="follow the job until it finishes")
    submit.set_defaults(handler=cmd_submit)

    for name, handler, helptext in (
        ("follow", cmd_follow, "stream a job's events until it finishes"),
        ("status", cmd_status, "show a job's state and results"),
        ("cancel", cmd_cancel, "cancel a queued or running job"),
    ):
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("job_id")
        if name == "status":
            sub.add_argument("--full", action="store_true", help="include the full result document")
        sub.set_defaults(handler=handler)

    listing = commands.add_parser("list", help="recent jobs")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(handler=cmd_list)

    env = commands.add_parser("env", help="build (or show) a pinned environment")
    env.add_argument("--template", default=settings.DEFAULT_ENVIRONMENT_TEMPLATE)
    env.add_argument("--packs", default="")
    env.set_defaults(handler=cmd_env)

    verify = commands.add_parser(
        "verify-env", help="check that environments re-evaluate to the same store path"
    )
    verify.add_argument("env_id", nargs="?", help="one environment (default: all ready ones)")
    verify.set_defaults(handler=cmd_verify_env)

    session = commands.add_parser("session", help="call one session method in a pinned environment")
    session.add_argument("method")
    session.add_argument("--params", help="JSON params")
    session.add_argument("--params-file", help="file with JSON params")
    session.add_argument("--template", default=settings.DEFAULT_ENVIRONMENT_TEMPLATE)
    session.add_argument("--packs", default="")
    session.set_defaults(handler=cmd_session)
    return parser


def main(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return asyncio.run(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
