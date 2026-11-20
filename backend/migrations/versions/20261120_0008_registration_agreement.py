"""Record purpose and agreement for new accounts.

Revision ID: 20261120_0008
Revises: 20261020_0007
"""

import sqlalchemy as sa
from alembic import op

revision = "20261120_0008"
down_revision = "20261020_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("purpose", sa.String(250), nullable=True))
    op.add_column("users", sa.Column("agreement_version", sa.String(120), nullable=True))
    op.add_column(
        "users", sa.Column("agreement_accepted_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("users", "agreement_accepted_at")
    op.drop_column("users", "agreement_version")
    op.drop_column("users", "purpose")
