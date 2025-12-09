"""SQLAlchemy models for video capture and historical data.

This module defines the schema for storing:
- Video capture sessions with metadata
- Input events (mouse, keyboard) as time-series data
- Frame index for efficient video frame extraction
- Links between automation results and video timestamps

The design follows a hybrid approach:
- PostgreSQL stores metadata and queryable data
- Object storage (local filesystem or S3) stores video files
"""

from enum import Enum as PyEnum

from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Enum, Float,
                        ForeignKey, Index, Integer, Numeric, SmallInteger,
                        String, Text, UniqueConstraint)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class StorageBackend(PyEnum):
    """Storage backend types for video files."""

    LOCAL = "local"
    S3 = "s3"


class InputEventType(PyEnum):
    """Types of input events."""

    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    MOUSE_SCROLL = "mouse_scroll"
    MOUSE_DRAG = "mouse_drag"
    KEY_PRESS = "key_press"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"


class CaptureSession(Base):
    """
    Video capture session metadata.

    Represents a single recording session that captures:
    - Screen video
    - Input events (mouse, keyboard)
    - Automation results (linked via snapshot_run)

    The video file is stored externally (local filesystem or S3),
    with only the reference stored in the database.
    """

    __tablename__ = "capture_sessions"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Unique session identifier (UUID)
    session_id = Column(String(36), unique=True, nullable=False, index=True)

    # Timestamps
    started_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    ended_at = Column(DateTime, nullable=True)
    duration_ms = Column(BigInteger, nullable=True)  # Total duration in milliseconds

    # Video metadata
    video_width = Column(Integer, nullable=False)
    video_height = Column(Integer, nullable=False)
    video_fps = Column(Float, nullable=False, default=30.0)
    video_codec = Column(String(50), nullable=False, default="h264")
    video_format = Column(String(10), nullable=False, default="mp4")
    total_frames = Column(Integer, nullable=True)

    # Storage information
    storage_backend: Column[StorageBackend] = Column(
        Enum(StorageBackend), nullable=False, default=StorageBackend.LOCAL
    )  # type: ignore[assignment]
    video_path = Column(Text, nullable=False)  # Local path or S3 key
    video_size_bytes = Column(BigInteger, nullable=True)

    # Optional compressed/streaming version
    compressed_video_path = Column(Text, nullable=True)
    compressed_video_size_bytes = Column(BigInteger, nullable=True)

    # Monitor/display info
    monitor_id = Column(Integer, nullable=True)
    monitor_name = Column(String(255), nullable=True)
    monitor_scale_factor = Column(Float, nullable=True, default=1.0)

    # Association with snapshot run (automation results)
    snapshot_run_id = Column(
        Integer, ForeignKey("snapshot_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Association with workflow
    workflow_id = Column(
        Integer, ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Association with project
    project_id = Column(
        Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # User who created the session
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Session status
    is_complete = Column(Boolean, nullable=False, default=False)
    is_processed = Column(Boolean, nullable=False, default=False)  # Frame index built

    # Additional metadata
    metadata_json = Column(JSONB, nullable=True)
    tags: list[str] = Column(ARRAY(Text), nullable=True)  # type: ignore[assignment]
    notes = Column(Text, nullable=True)

    # Audit timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    # Relationships
    input_events = relationship(
        "InputEvent",
        back_populates="capture_session",
        cascade="all, delete-orphan",
        lazy="dynamic",  # For efficient querying of large event sets
    )
    frame_index = relationship(
        "FrameIndex", back_populates="capture_session", cascade="all, delete-orphan", lazy="dynamic"
    )
    action_frames = relationship(
        "ActionFrame", back_populates="capture_session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_capture_sessions_started_at", "started_at"),
        Index("idx_capture_sessions_workflow_id", "workflow_id"),
        Index("idx_capture_sessions_project_id", "project_id"),
        Index("idx_capture_sessions_snapshot_run_id", "snapshot_run_id"),
    )

    def __repr__(self) -> str:
        return f"<CaptureSession(id={self.id}, session_id='{self.session_id}')>"


class InputEvent(Base):
    """
    Input events (mouse and keyboard) captured during a session.

    Stored as time-series data with millisecond precision timestamps
    relative to session start. This allows efficient querying for
    events within a time range and playback synchronization with video.
    """

    __tablename__ = "input_events"

    # Primary key
    id = Column(BigInteger, primary_key=True, index=True)

    # Foreign key to capture session
    capture_session_id = Column(
        Integer, ForeignKey("capture_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Timestamp (milliseconds from session start)
    timestamp_ms = Column(BigInteger, nullable=False, index=True)

    # Event type
    event_type: Column[InputEventType] = Column(Enum(InputEventType), nullable=False, index=True)  # type: ignore[assignment]

    # Mouse position (for mouse events)
    mouse_x = Column(Integer, nullable=True)
    mouse_y = Column(Integer, nullable=True)

    # Mouse button (for click events): 1=left, 2=middle, 3=right
    mouse_button = Column(SmallInteger, nullable=True)

    # Scroll delta (for scroll events)
    scroll_dx = Column(Integer, nullable=True)
    scroll_dy = Column(Integer, nullable=True)

    # Keyboard data
    key_code = Column(String(50), nullable=True)  # Virtual key code
    key_name = Column(String(50), nullable=True)  # Human-readable key name
    key_char = Column(String(10), nullable=True)  # Character if printable

    # Modifier keys state
    shift_pressed = Column(Boolean, nullable=True, default=False)
    ctrl_pressed = Column(Boolean, nullable=True, default=False)
    alt_pressed = Column(Boolean, nullable=True, default=False)
    meta_pressed = Column(Boolean, nullable=True, default=False)  # Win/Cmd key

    # Additional event data (for complex events like drag)
    event_data_json = Column(JSONB, nullable=True)

    # Relationship
    capture_session = relationship("CaptureSession", back_populates="input_events")

    __table_args__ = (
        Index("idx_input_events_session_timestamp", "capture_session_id", "timestamp_ms"),
        Index("idx_input_events_type", "event_type"),
    )

    def __repr__(self) -> str:
        return f"<InputEvent(id={self.id}, type={self.event_type}, ts={self.timestamp_ms}ms)>"


class FrameIndex(Base):
    """
    Index mapping timestamps to video frame positions.

    This table enables efficient frame extraction by storing:
    - Keyframe positions (I-frames) for fast seeking
    - Timestamp to frame number mapping
    - Byte offsets for direct seeking (optional)

    Only keyframes and periodic samples are stored to keep the table size manageable.
    """

    __tablename__ = "frame_index"

    # Primary key
    id = Column(BigInteger, primary_key=True, index=True)

    # Foreign key to capture session
    capture_session_id = Column(
        Integer, ForeignKey("capture_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Frame information
    frame_number = Column(Integer, nullable=False)
    timestamp_ms = Column(BigInteger, nullable=False, index=True)  # Milliseconds from start

    # Video seeking information
    byte_offset = Column(BigInteger, nullable=True)  # Byte position in video file
    is_keyframe = Column(Boolean, nullable=False, default=False, index=True)

    # Frame metadata (optional, for important frames)
    frame_hash = Column(String(64), nullable=True)  # For deduplication/comparison

    # Relationship
    capture_session = relationship("CaptureSession", back_populates="frame_index")

    __table_args__ = (
        Index("idx_frame_index_session_timestamp", "capture_session_id", "timestamp_ms"),
        Index("idx_frame_index_session_frame", "capture_session_id", "frame_number"),
        Index("idx_frame_index_keyframes", "capture_session_id", "is_keyframe"),
        UniqueConstraint("capture_session_id", "frame_number", name="uq_frame_index_session_frame"),
    )

    def __repr__(self) -> str:
        return f"<FrameIndex(session={self.capture_session_id}, frame={self.frame_number})>"


class ActionFrame(Base):
    """
    Links automation actions to specific video frames.

    This table enables:
    - Retrieving the exact frame when an action occurred
    - Showing visual context for automation results
    - Integration test playback with screenshots

    Each automation action (from SnapshotAction) can have multiple
    associated frames (before, during, after the action).
    """

    __tablename__ = "action_frames"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    capture_session_id = Column(
        Integer, ForeignKey("capture_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_action_id = Column(
        Integer, ForeignKey("snapshot_actions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Frame timing
    frame_number = Column(Integer, nullable=False)
    timestamp_ms = Column(BigInteger, nullable=False)

    # Frame type relative to action
    frame_type = Column(
        String(20), nullable=False, default="action"
    )  # 'before', 'action', 'after', 'result'

    # Cached frame path (if extracted and cached)
    cached_frame_path = Column(Text, nullable=True)
    cache_storage_backend: Column[StorageBackend | None] = Column(
        Enum(StorageBackend), nullable=True
    )  # type: ignore[assignment]

    # Relationship
    capture_session = relationship("CaptureSession", back_populates="action_frames")

    __table_args__ = (
        Index("idx_action_frames_session", "capture_session_id"),
        Index("idx_action_frames_action", "snapshot_action_id"),
        Index("idx_action_frames_type", "frame_type"),
        UniqueConstraint(
            "capture_session_id",
            "snapshot_action_id",
            "frame_type",
            name="uq_action_frames_session_action_type",
        ),
    )

    def __repr__(self) -> str:
        return f"<ActionFrame(action={self.snapshot_action_id}, type={self.frame_type})>"


class HistoricalResult(Base):
    """
    Queryable historical results for integration testing.

    This table aggregates data from snapshot actions in a format
    optimized for random selection during mock mode execution.
    It includes denormalized fields for efficient querying without
    joins.

    Key features:
    - Indexed by pattern, state, and action type for fast lookups
    - Links to capture session for frame retrieval
    - Stores result data needed for mock responses
    """

    __tablename__ = "historical_results"

    # Primary key
    id = Column(BigInteger, primary_key=True, index=True)

    # Source references
    snapshot_run_id = Column(
        Integer, ForeignKey("snapshot_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_action_id = Column(
        Integer, ForeignKey("snapshot_actions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    capture_session_id = Column(
        Integer, ForeignKey("capture_sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Denormalized query fields (for efficient selection)
    pattern_id = Column(String(255), nullable=True, index=True)
    pattern_name = Column(String(255), nullable=True)
    action_type = Column(String(50), nullable=False, index=True)
    active_states: list[str] = Column(ARRAY(Text), nullable=True)  # type: ignore[assignment]

    # Result data
    success = Column(Boolean, nullable=False, index=True)
    match_count = Column(Integer, nullable=True)
    best_match_score = Column(Numeric(5, 4), nullable=True)
    duration_ms = Column(Numeric(10, 3), nullable=True)

    # Match location (for FIND actions)
    match_x = Column(Integer, nullable=True)
    match_y = Column(Integer, nullable=True)
    match_width = Column(Integer, nullable=True)
    match_height = Column(Integer, nullable=True)

    # Video frame info (for frame retrieval)
    frame_timestamp_ms = Column(BigInteger, nullable=True)
    frame_number = Column(Integer, nullable=True)

    # Full result data for mock responses
    result_data_json = Column(JSONB, nullable=False)

    # Workflow/project context
    workflow_id = Column(
        Integer, ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True, index=True
    )
    project_id = Column(
        Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Timestamps
    recorded_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=func.now())

    __table_args__ = (
        # Composite indexes for common query patterns
        Index("idx_historical_pattern_states", "pattern_id", "active_states"),
        Index("idx_historical_action_type_success", "action_type", "success"),
        Index("idx_historical_workflow_pattern", "workflow_id", "pattern_id"),
        Index("idx_historical_project_pattern", "project_id", "pattern_id"),
        # For random selection within a context
        Index("idx_historical_selection", "pattern_id", "action_type", "success", "recorded_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<HistoricalResult(id={self.id}, pattern={self.pattern_id}, type={self.action_type})>"
        )
