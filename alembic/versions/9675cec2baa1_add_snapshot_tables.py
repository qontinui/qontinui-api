"""Add snapshot tables

Revision ID: 9675cec2baa1
Revises:
Create Date: 2025-10-19 13:40:52.162740

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9675cec2baa1"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create snapshot_runs table
    op.create_table(
        "snapshot_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("run_directory", sa.Text(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column("execution_mode", sa.String(length=50), nullable=False),
        sa.Column("total_actions", sa.Integer(), nullable=True),
        sa.Column("successful_actions", sa.Integer(), nullable=True),
        sa.Column("failed_actions", sa.Integer(), nullable=True),
        sa.Column("total_screenshots", sa.Integer(), nullable=True),
        sa.Column("patterns_count", sa.Integer(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_snapshot_runs_id"), "snapshot_runs", ["id"], unique=False)
    op.create_index(op.f("ix_snapshot_runs_run_id"), "snapshot_runs", ["run_id"], unique=True)
    op.create_index("idx_snapshot_runs_start_time", "snapshot_runs", ["start_time"], unique=False)
    op.create_index(
        "idx_snapshot_runs_execution_mode", "snapshot_runs", ["execution_mode"], unique=False
    )
    op.create_index("idx_snapshot_runs_workflow_id", "snapshot_runs", ["workflow_id"], unique=False)
    op.create_index("idx_snapshot_runs_created_by", "snapshot_runs", ["created_by"], unique=False)

    # Create snapshot_actions table
    op.create_table(
        "snapshot_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("snapshot_run_id", sa.Integer(), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("action_type", sa.String(length=50), nullable=False),
        sa.Column("pattern_id", sa.String(length=255), nullable=True),
        sa.Column("pattern_name", sa.String(length=255), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("match_count", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column("active_states", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("screenshot_path", sa.Text(), nullable=True),
        sa.Column("action_data_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_run_id"], ["snapshot_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_snapshot_actions_id"), "snapshot_actions", ["id"], unique=False)
    op.create_index(
        op.f("ix_snapshot_actions_snapshot_run_id"),
        "snapshot_actions",
        ["snapshot_run_id"],
        unique=False,
    )
    op.create_index(
        "idx_snapshot_actions_run_sequence",
        "snapshot_actions",
        ["snapshot_run_id", "sequence_number"],
        unique=False,
    )
    op.create_index(
        "idx_snapshot_actions_timestamp", "snapshot_actions", ["timestamp"], unique=False
    )
    op.create_index(
        "idx_snapshot_actions_pattern_id", "snapshot_actions", ["pattern_id"], unique=False
    )
    op.create_index(
        "idx_snapshot_actions_action_type", "snapshot_actions", ["action_type"], unique=False
    )
    op.create_index("idx_snapshot_actions_success", "snapshot_actions", ["success"], unique=False)

    # Create snapshot_patterns table
    op.create_table(
        "snapshot_patterns",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("snapshot_run_id", sa.Integer(), nullable=False),
        sa.Column("pattern_id", sa.String(length=255), nullable=False),
        sa.Column("pattern_name", sa.String(length=255), nullable=False),
        sa.Column("total_finds", sa.Integer(), nullable=True),
        sa.Column("successful_finds", sa.Integer(), nullable=True),
        sa.Column("failed_finds", sa.Integer(), nullable=True),
        sa.Column("total_matches", sa.Integer(), nullable=True),
        sa.Column("avg_duration_ms", sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column("pattern_data_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_run_id"], ["snapshot_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_snapshot_patterns_id"), "snapshot_patterns", ["id"], unique=False)
    op.create_index(
        op.f("ix_snapshot_patterns_snapshot_run_id"),
        "snapshot_patterns",
        ["snapshot_run_id"],
        unique=False,
    )
    op.create_index(
        "idx_snapshot_patterns_run_pattern",
        "snapshot_patterns",
        ["snapshot_run_id", "pattern_id"],
        unique=False,
    )
    op.create_index(
        "idx_snapshot_patterns_pattern_id", "snapshot_patterns", ["pattern_id"], unique=False
    )

    # Create snapshot_matches table
    op.create_table(
        "snapshot_matches",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pattern_id", sa.Integer(), nullable=False),
        sa.Column("action_id", sa.Integer(), nullable=False),
        sa.Column("match_index", sa.Integer(), nullable=False),
        sa.Column("x", sa.Integer(), nullable=False),
        sa.Column("y", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("score", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("match_data_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["action_id"], ["snapshot_actions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pattern_id"], ["snapshot_patterns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_snapshot_matches_id"), "snapshot_matches", ["id"], unique=False)
    op.create_index(
        "idx_snapshot_matches_pattern", "snapshot_matches", ["pattern_id"], unique=False
    )
    op.create_index("idx_snapshot_matches_action", "snapshot_matches", ["action_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_snapshot_matches_action", table_name="snapshot_matches")
    op.drop_index("idx_snapshot_matches_pattern", table_name="snapshot_matches")
    op.drop_index(op.f("ix_snapshot_matches_id"), table_name="snapshot_matches")
    op.drop_table("snapshot_matches")

    op.drop_index("idx_snapshot_patterns_pattern_id", table_name="snapshot_patterns")
    op.drop_index("idx_snapshot_patterns_run_pattern", table_name="snapshot_patterns")
    op.drop_index(op.f("ix_snapshot_patterns_snapshot_run_id"), table_name="snapshot_patterns")
    op.drop_index(op.f("ix_snapshot_patterns_id"), table_name="snapshot_patterns")
    op.drop_table("snapshot_patterns")

    op.drop_index("idx_snapshot_actions_success", table_name="snapshot_actions")
    op.drop_index("idx_snapshot_actions_action_type", table_name="snapshot_actions")
    op.drop_index("idx_snapshot_actions_pattern_id", table_name="snapshot_actions")
    op.drop_index("idx_snapshot_actions_timestamp", table_name="snapshot_actions")
    op.drop_index("idx_snapshot_actions_run_sequence", table_name="snapshot_actions")
    op.drop_index(op.f("ix_snapshot_actions_snapshot_run_id"), table_name="snapshot_actions")
    op.drop_index(op.f("ix_snapshot_actions_id"), table_name="snapshot_actions")
    op.drop_table("snapshot_actions")

    op.drop_index("idx_snapshot_runs_created_by", table_name="snapshot_runs")
    op.drop_index("idx_snapshot_runs_workflow_id", table_name="snapshot_runs")
    op.drop_index("idx_snapshot_runs_execution_mode", table_name="snapshot_runs")
    op.drop_index("idx_snapshot_runs_start_time", table_name="snapshot_runs")
    op.drop_index(op.f("ix_snapshot_runs_run_id"), table_name="snapshot_runs")
    op.drop_index(op.f("ix_snapshot_runs_id"), table_name="snapshot_runs")
    op.drop_table("snapshot_runs")
