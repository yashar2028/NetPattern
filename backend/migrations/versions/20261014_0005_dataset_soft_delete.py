"""soft delete for datasets and dataset versions

Revision ID: 20261014_0005
Revises: 20261014_0004
Create Date: 2026-10-14 00:00:01.000000
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20261014_0005"
down_revision = "20261014_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "dataset_versions", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("dataset_versions", "deleted_at")
    op.drop_column("datasets", "deleted_at")
