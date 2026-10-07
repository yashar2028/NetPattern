"""The streaming job runner, exercised with a fake engine runtime (no Nix needed)."""

import json
import sys
from pathlib import Path

import pytest

from app.sandbox.isolation import SandboxPolicy
from app.sandbox.runner import run_engine_job

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX process groups")

FAKE_RUNTIME = """#!{python}
import json, os, signal, sys, time
args = sys.argv[1:]
request = json.load(open(args[args.index("--request") + 1]))
output = args[args.index("--output") + 1]
cancel_file = os.path.join(os.path.dirname(output), "cancel.request")
stopped = []
signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))

def emit(kind, **data):
    print(json.dumps(dict(type=kind, **data)), flush=True)

emit("run_started", run_id=request["run_id"])
print("library noise that is not JSON", flush=True)
for epoch in range(1, request.get("epochs", 3) + 1):
    if stopped or (not request.get("ignore_cancel_file") and os.path.exists(cancel_file)):
        json.dump(dict(run_id=request["run_id"], status="cancelled"), open(output, "w"))
        emit("run_cancelled")
        sys.exit(0)
    emit("epoch_end", epoch=epoch)
    time.sleep(request.get("sleep", 0))
json.dump(dict(run_id=request["run_id"], status="completed", epochs=epoch), open(output, "w"))
emit("run_completed")
"""


@pytest.fixture
def fake_env(tmp_path) -> Path:
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    runtime = env / "bin" / "netpattern-runtime"
    runtime.write_text(FAKE_RUNTIME.format(python=sys.executable))
    runtime.chmod(0o755)
    return env


async def _run(fake_env, run_dir, request, should_cancel=lambda: False, timeout=None):
    events = []

    async def on_event(event):
        events.append(event)

    outcome = await run_engine_job(
        fake_env, run_dir, request, on_event, should_cancel, SandboxPolicy(), timeout=timeout
    )
    return outcome, events


async def test_events_stream_in_order_and_result_is_read(fake_env, tmp_path):
    outcome, events = await _run(fake_env, tmp_path / "run", {"run_id": "r1", "epochs": 3})

    assert outcome.exit_code == 0 and not outcome.cancelled
    assert [e["type"] for e in events] == [
        "run_started",
        "epoch_end",
        "epoch_end",
        "epoch_end",
        "run_completed",
    ]
    assert outcome.result == {"run_id": "r1", "status": "completed", "epochs": 3}
    assert json.loads((tmp_path / "run" / "request.json").read_text())["run_id"] == "r1"
    assert "library noise" in (tmp_path / "run" / "stderr.log").read_text()


async def test_cancellation_asks_the_engine_to_stop_cleanly(fake_env, tmp_path):
    outcome, events = await _run(
        fake_env,
        tmp_path / "run",
        {"run_id": "r2", "epochs": 100, "sleep": 0.2},
        should_cancel=lambda: True,
    )

    assert outcome.cancelled
    assert outcome.result["status"] == "cancelled"
    assert events[-1]["type"] == "run_cancelled"
    assert len([e for e in events if e["type"] == "epoch_end"]) < 100


async def test_engines_ignoring_the_request_get_sigterm(fake_env, tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "SANDBOX_CANCEL_GRACE_SECONDS", 3)
    outcome, events = await _run(
        fake_env,
        tmp_path / "run",
        {"run_id": "r4", "epochs": 100, "sleep": 0.2, "ignore_cancel_file": True},
        should_cancel=lambda: True,
    )

    assert outcome.cancelled and outcome.result["status"] == "cancelled"
    assert events[-1]["type"] == "run_cancelled"


async def test_timeout_stops_the_job(fake_env, tmp_path):
    outcome, _ = await _run(
        fake_env, tmp_path / "run", {"run_id": "r3", "epochs": 100, "sleep": 0.2}, timeout=1
    )

    assert outcome.timed_out
