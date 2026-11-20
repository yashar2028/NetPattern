"""dataset version upload status; dataset versions read by each job

Revision ID: 20261014_0004
Revises: 20261008_0003
Create Date: 2026-10-14 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20261014_0004"
down_revision = "20261008_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "dataset_versions",
        sa.Column("status", sa.String(length=16), server_default="ready", nullable=False),
    )
    op.add_column(
        "jobs",
        sa.Column(
            "dataset_version_ids",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_jobs_dataset_version_ids",
        "jobs",
        ["dataset_version_ids"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_dataset_version_ids", table_name="jobs")
    op.drop_column("jobs", "dataset_version_ids")
    op.drop_column("dataset_versions", "status")
