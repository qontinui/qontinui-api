"""Add is_start_screenshot to snapshot_actions

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2025-10-19 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6g7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add is_start_screenshot column to snapshot_actions table."""
    op.add_column(
        "snapshot_actions",
        sa.Column("is_start_screenshot", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    """Remove is_start_screenshot column from snapshot_actions table."""
    op.drop_column("snapshot_actions", "is_start_screenshot")
