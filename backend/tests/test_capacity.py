"""Worker capacity (bubbles on a machine), sandbox pinning, dedicated workers, compute units."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, update

from app.core.tiers import Tier
from app.db.session import AsyncSessionLocal
from app.models.job import Job, JobStatus
from app.models.usage import UsageRecord
from app.models.user import User
from app.sandbox.isolation import SandboxPolicy, sandbox_env, wrap
from app.workers import queue
from app.workers.capacity import Capacity

SMALL = Tier("small", 2, 4, 0, 1.0)
GPU = Tier("big-gpu", 4, 8, 1, 20.0)


def test_jobs_fill_the_machine_until_no_bubble_fits():
    machine = Capacity(cpus=list(range(8)), memory_gb=16, gpus=[0], max_jobs=3)

    first = machine.allocate(SMALL)
    second = machine.allocate(GPU)

    assert first.cpus == [0, 1] and second.cpus == [2, 3, 4, 5] and second.gpus == [0]
    assert machine.fits(SMALL)  # 2 cores and 4 GB left, one slot left
    assert not machine.fits(GPU)  # the only GPU is taken
    machine.allocate(SMALL)
    assert not machine.fits(SMALL)  # all three slots are used
    machine.release(second)
    assert machine.fits(GPU)


def test_a_machine_does_not_serve_bubbles_bigger_than_itself():
    laptop = Capacity(cpus=[0, 1], memory_gb=8, gpus=[], max_jobs=1)
    assert laptop.can_ever_run(SMALL) and not laptop.can_ever_run(GPU)


def test_sandboxes_are_pinned_to_their_cores_and_gpus(monkeypatch):
    allocation = SimpleNamespace(cpus=[2, 3], gpus=[])
    policy = SandboxPolicy(resources=allocation)
    monkeypatch.setattr("app.sandbox.isolation.shutil.which", lambda name: f"/usr/bin/{name}")

    command = wrap(["python"], policy, mode="none")
    env = sandbox_env(Path("/nix/store/env"), Path("/tmp/run"), False, allocation)

    assert command == ["/usr/bin/taskset", "--cpu-list", "2,3", "python"]
    assert env["OMP_NUM_THREADS"] == "2" and env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.integration
async def test_dedicated_accounts_run_only_on_their_own_worker(client, register):
    _, shared_user = await register()
    _, dedicated_user = await register()
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User).where(User.id == dedicated_user["id"]).values(dedicated_worker=True)
        )
        await db.commit()
        tier = "test-tier-dedicated"
        shared = await queue.enqueue(
            db, "smoke", {}, hardware_tier=tier, owner_id=shared_user["id"]
        )
        mine = await queue.enqueue(
            db, "smoke", {}, hardware_tier=tier, owner_id=dedicated_user["id"]
        )

    async with AsyncSessionLocal() as db:
        first_shared = await queue.claim_next(db, "shared-worker", [tier])
        second_shared = await queue.claim_next(db, "shared-worker", [tier])
        dedicated = await queue.claim_next(db, "own-worker", [tier], dedicated_user["id"])
        await queue.finish(db, dedicated.id, JobStatus.COMPLETED, {"status": "completed"})
        record = (
            await db.execute(UsageRecord.__table__.select().where(UsageRecord.job_id == mine.id))
        ).one()

    assert first_shared.id == shared.id and second_shared is None
    assert dedicated.id == mine.id
    assert record.units == pytest.approx(record.seconds)  # unknown tiers cost 1 unit per second
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Job).where(Job.hardware_tier == tier))
        await db.commit()
