"""users, api keys, sandboxes, pipelines, datasets, usage; job ownership

Revision ID: 20261008_0003
Revises: 20261007_0002
Create Date: 2026-10-08 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20261008_0003"
down_revision = "20261007_0002"
branch_labels = None
depends_on = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("user_id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("lookup", sa.String(length=16), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        _created_at(),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_api_keys_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_api_keys"),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_lookup", "api_keys", ["lookup"])

    op.create_table(
        "sandboxes",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("owner_id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("template", sa.String(length=64), nullable=False),
        sa.Column("engine_snapshot", sa.String(length=64), nullable=True),
        sa.Column("engine_version", sa.String(length=32), nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_sandboxes_owner_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sandboxes"),
    )
    op.create_index("ix_sandboxes_owner_id", "sandboxes", ["owner_id"])

    op.create_table(
        "pipelines",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("sandbox_id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["sandbox_id"],
            ["sandboxes.id"],
            name="fk_pipelines_sandbox_id_sandboxes",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_pipelines"),
    )
    op.create_index("ix_pipelines_sandbox_id", "pipelines", ["sandbox_id"])

    op.create_table(
        "pipeline_versions",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("pipeline_id", sa.String(length=40), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["pipeline_id"],
            ["pipelines.id"],
            name="fk_pipeline_versions_pipeline_id_pipelines",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_pipeline_versions"),
        sa.UniqueConstraint("pipeline_id", "number", name="uq_pipeline_versions_pipeline_id"),
    )
    op.create_index("ix_pipeline_versions_pipeline_id", "pipeline_versions", ["pipeline_id"])

    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("owner_id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_datasets_owner_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
    )
    op.create_index("ix_datasets_owner_id", "datasets", ["owner_id"])

    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("dataset_id", sa.String(length=40), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("files", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_dataset_versions_dataset_id_datasets",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_dataset_versions"),
        sa.UniqueConstraint("dataset_id", "number", name="uq_dataset_versions_dataset_id"),
    )
    op.create_index("ix_dataset_versions_dataset_id", "dataset_versions", ["dataset_id"])

    op.create_table(
        "usage_records",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("owner_id", sa.String(length=40), nullable=False),
        sa.Column("sandbox_id", sa.String(length=40), nullable=True),
        sa.Column("job_id", sa.String(length=40), nullable=False),
        sa.Column("hardware_tier", sa.String(length=32), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_usage_records_owner_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["sandbox_id"],
            ["sandboxes.id"],
            name="fk_usage_records_sandbox_id_sandboxes",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_usage_records"),
        sa.UniqueConstraint("job_id", name="uq_usage_records_job_id"),
    )
    op.create_index("ix_usage_records_owner_id", "usage_records", ["owner_id"])

    op.add_column("jobs", sa.Column("owner_id", sa.String(length=40), nullable=True))
    op.add_column("jobs", sa.Column("sandbox_id", sa.String(length=40), nullable=True))
    op.add_column("jobs", sa.Column("pipeline_version_id", sa.String(length=40), nullable=True))
    op.create_foreign_key(
        "fk_jobs_owner_id_users", "jobs", "users", ["owner_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_jobs_sandbox_id_sandboxes",
        "jobs",
        "sandboxes",
        ["sandbox_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_jobs_pipeline_version_id_pipeline_versions",
        "jobs",
        "pipeline_versions",
        ["pipeline_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_jobs_owner_id", "jobs", ["owner_id"])
    op.create_index("ix_jobs_sandbox_id", "jobs", ["sandbox_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_sandbox_id", table_name="jobs")
    op.drop_index("ix_jobs_owner_id", table_name="jobs")
    op.drop_constraint("fk_jobs_pipeline_version_id_pipeline_versions", "jobs", type_="foreignkey")
    op.drop_constraint("fk_jobs_sandbox_id_sandboxes", "jobs", type_="foreignkey")
    op.drop_constraint("fk_jobs_owner_id_users", "jobs", type_="foreignkey")
    op.drop_column("jobs", "pipeline_version_id")
    op.drop_column("jobs", "sandbox_id")
    op.drop_column("jobs", "owner_id")
    for table in (
        "usage_records",
        "dataset_versions",
        "datasets",
        "pipeline_versions",
        "pipelines",
        "sandboxes",
        "api_keys",
        "users",
    ):
        op.drop_table(table)
