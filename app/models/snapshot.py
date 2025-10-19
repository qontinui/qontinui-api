"""SQLAlchemy models for snapshot data."""

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


class SnapshotRun(Base):
    """
    Snapshot run metadata table.

    Stores high-level information about a recording session.
    """

    __tablename__ = "snapshot_runs"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Identifiers
    run_id = Column(String(255), unique=True, nullable=False, index=True)
    run_directory = Column(Text, nullable=False)

    # Timestamps
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Numeric(10, 3), nullable=True)

    # Execution info
    execution_mode = Column(String(50), nullable=False, index=True)

    # Statistics
    total_actions = Column(Integer, default=0)
    successful_actions = Column(Integer, default=0)
    failed_actions = Column(Integer, default=0)
    total_screenshots = Column(Integer, default=0)
    patterns_count = Column(Integer, default=0)

    # Full metadata from metadata.json
    metadata_json = Column(JSONB, nullable=False)

    # Optional associations
    workflow_id = Column(Integer, ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Tags and notes
    tags = Column(ARRAY(Text), nullable=True)
    notes = Column(Text, nullable=True)

    # Priority for smart selection (higher = preferred in recommendations)
    priority = Column(Integer, default=50, nullable=False)

    # Relationships
    actions = relationship(
        "SnapshotAction",
        back_populates="snapshot_run",
        cascade="all, delete-orphan",
    )
    patterns = relationship(
        "SnapshotPattern",
        back_populates="snapshot_run",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_snapshot_runs_start_time", "start_time"),
        Index("idx_snapshot_runs_execution_mode", "execution_mode"),
        Index("idx_snapshot_runs_workflow_id", "workflow_id"),
        Index("idx_snapshot_runs_created_by", "created_by"),
    )

    def __repr__(self) -> str:
        return f"<SnapshotRun(run_id='{self.run_id}', start_time='{self.start_time}')>"


class SnapshotAction(Base):
    """
    Snapshot action records table.

    Stores individual actions from the action log.
    """

    __tablename__ = "snapshot_actions"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    snapshot_run_id = Column(
        Integer,
        ForeignKey("snapshot_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Action identifiers
    sequence_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    action_type = Column(String(50), nullable=False, index=True)

    # Pattern info
    pattern_id = Column(String(255), nullable=True, index=True)
    pattern_name = Column(String(255), nullable=True)

    # Result info
    success = Column(Boolean, nullable=False, default=True, index=True)
    match_count = Column(Integer, nullable=True)
    duration_ms = Column(Numeric(10, 3), nullable=True)

    # State and context
    active_states = Column(ARRAY(Text), nullable=True)
    screenshot_path = Column(Text, nullable=True)
    is_start_screenshot = Column(Boolean, nullable=False, default=False, index=True)

    # Full action data from action_log.json
    action_data_json = Column(JSONB, nullable=False)

    # Relationship
    snapshot_run = relationship("SnapshotRun", back_populates="actions")

    # Indexes
    __table_args__ = (
        Index("idx_snapshot_actions_run_sequence", "snapshot_run_id", "sequence_number"),
        Index("idx_snapshot_actions_timestamp", "timestamp"),
        Index("idx_snapshot_actions_pattern_id", "pattern_id"),
        Index("idx_snapshot_actions_action_type", "action_type"),
        Index("idx_snapshot_actions_success", "success"),
    )

    def __repr__(self) -> str:
        return (
            f"<SnapshotAction(run_id={self.snapshot_run_id}, "
            f"seq={self.sequence_number}, type='{self.action_type}')>"
        )


class SnapshotPattern(Base):
    """
    Pattern statistics table.

    Aggregates pattern usage across a snapshot run.
    """

    __tablename__ = "snapshot_patterns"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    snapshot_run_id = Column(
        Integer,
        ForeignKey("snapshot_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Pattern identifiers
    pattern_id = Column(String(255), nullable=False, index=True)
    pattern_name = Column(String(255), nullable=False)

    # Statistics
    total_finds = Column(Integer, default=0)
    successful_finds = Column(Integer, default=0)
    failed_finds = Column(Integer, default=0)
    total_matches = Column(Integer, default=0)
    avg_duration_ms = Column(Numeric(10, 3), nullable=True)

    # Full pattern data from patterns/{pattern-id}/metadata.json
    pattern_data_json = Column(JSONB, nullable=False)

    # Relationships
    snapshot_run = relationship("SnapshotRun", back_populates="patterns")
    matches = relationship(
        "SnapshotMatch",
        back_populates="pattern",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_snapshot_patterns_run_pattern", "snapshot_run_id", "pattern_id"),
        Index("idx_snapshot_patterns_pattern_id", "pattern_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<SnapshotPattern(run_id={self.snapshot_run_id}, " f"pattern_id='{self.pattern_id}')>"
        )


class SnapshotMatch(Base):
    """
    Individual match records table.

    Stores detailed match information for pattern finds.
    """

    __tablename__ = "snapshot_matches"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    pattern_id = Column(
        Integer,
        ForeignKey("snapshot_patterns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action_id = Column(
        Integer,
        ForeignKey("snapshot_actions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Match info
    match_index = Column(Integer, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    score = Column(Numeric(5, 4), nullable=True)

    # Full match data from patterns/{pattern-id}/history.json
    match_data_json = Column(JSONB, nullable=False)

    # Relationships
    pattern = relationship("SnapshotPattern", back_populates="matches")

    # Indexes
    __table_args__ = (
        Index("idx_snapshot_matches_pattern", "pattern_id"),
        Index("idx_snapshot_matches_action", "action_id"),
    )

    def __repr__(self) -> str:
        return f"<SnapshotMatch(pattern_id={self.pattern_id}, x={self.x}, y={self.y})>"
