"""baseline: empty schema

Revision ID: 20261002_0001
Revises:
Create Date: 2026-10-02 00:00:00.000000
"""

# revision identifiers, used by Alembic.
revision = "20261002_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Phase 0 baseline. The first tables (users, api keys, sandboxes, …) arrive in Phase 2.
    pass


def downgrade() -> None:
    pass
