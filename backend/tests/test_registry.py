"""Worker registry routing: stable choice per key, dedicated machines, stale workers ignored."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from app.db.session import AsyncSessionLocal
from app.models.worker import WorkerNode
from app.services import worker

pytestmark = pytest.mark.integration

PREFIX = "test-registry-"


@pytest.fixture
async def workers():
    now = datetime.now(UTC)

    def node(name, tiers, account=None, seen=now):
        return WorkerNode(
            id=PREFIX + name,
            url=f"http://{name}:9000",
            tiers=tiers,
            capacity={},
            account_id=account,
            version="test",
            running=0,
            last_seen_at=seen,
        )

    async with AsyncSessionLocal() as db:
        await db.execute(delete(WorkerNode))  # the dev worker re-registers within 30 s
        db.add_all(
            [
                node("a", ["cpu"]),
                node("b", ["cpu", "gpu"]),
                node("own", ["cpu"], account="usr_dedicated"),
                node("stale", ["cpu", "gpu"], seen=now - timedelta(minutes=10)),
            ]
        )
        await db.commit()
    yield
    async with AsyncSessionLocal() as db:
        await db.execute(delete(WorkerNode).where(WorkerNode.id.like(PREFIX + "%")))
        await db.commit()


async def test_routing_is_stable_respects_tiers_and_dedicated_machines(workers):
    async with AsyncSessionLocal() as db:
        first = await worker.pick(db, "session:sbx_1")
        again = await worker.pick(db, "session:sbx_1")
        gpu = {(await worker.pick(db, f"deployment:{n}", "gpu"))[0] for n in range(20)}
        spread = {(await worker.pick(db, f"session:sbx_{n}"))[0] for n in range(40)}
        dedicated = await worker.pick(db, "session:sbx_1", account_id="usr_dedicated")
        stale_url = await worker.url_of(db, PREFIX + "stale")

    assert first == again
    assert gpu == {PREFIX + "b"}  # the stale GPU worker is ignored
    assert spread == {PREFIX + "a", PREFIX + "b"}  # shared work avoids the dedicated machine
    assert dedicated == (PREFIX + "own", "http://own:9000")
    assert stale_url is None
