"""Add priority to snapshot_runs

Revision ID: a1b2c3d4e5f6
Revises: 9675cec2baa1
Create Date: 2025-10-19 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "9675cec2baa1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add priority column to snapshot_runs table."""
    op.add_column(
        "snapshot_runs", sa.Column("priority", sa.Integer(), nullable=False, server_default="50")
    )


def downgrade() -> None:
    """Remove priority column from snapshot_runs table."""
    op.drop_column("snapshot_runs", "priority")
