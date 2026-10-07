from app.workers.main import asyncpg_dsn


def test_asyncpg_dsn_drops_sqlalchemy_driver_suffix():
    dsn = asyncpg_dsn("postgresql+asyncpg://user:s3cret@db:5432/netpattern_dev")

    assert dsn == "postgresql://user:s3cret@db:5432/netpattern_dev"
