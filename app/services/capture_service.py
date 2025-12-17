"""Service for managing capture sessions and historical data.

This service provides:
- Capture session management (create, update, complete)
- Input event recording and retrieval
- Historical result indexing and querying
- Frame extraction for integration test playback
"""

import random
from datetime import datetime
from typing import Any, BinaryIO, Optional

from sqlalchemy.orm import Session

from app.models.capture import (
    ActionFrame,
    CaptureSession,
    FrameIndex,
    HistoricalResult,
    InputEvent,
    InputEventType,
    StorageBackend,
)
from app.models.snapshot import SnapshotAction, SnapshotRun
from app.services.storage import FrameExtractor, StorageBackendInterface, get_default_storage


class CaptureService:
    """Service for managing capture sessions and historical data."""

    def __init__(self, db: Session, storage: StorageBackendInterface | None = None):
        """Initialize capture service.

        Args:
            db: SQLAlchemy database session
            storage: Storage backend (uses default if not provided)
        """
        self.db = db
        self.storage = storage or get_default_storage()
        self.frame_extractor = FrameExtractor(self.storage)

    # =========================================================================
    # Capture Session Management
    # =========================================================================

    async def create_session(
        self,
        session_id: str,
        video_width: int,
        video_height: int,
        video_fps: float = 30.0,
        monitor_id: int | None = None,
        monitor_name: str | None = None,
        workflow_id: int | None = None,
        project_id: int | None = None,
        created_by: int | None = None,
        metadata: dict | None = None,
    ) -> CaptureSession:
        """Create a new capture session.

        Args:
            session_id: Unique session identifier (UUID)
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frames per second
            monitor_id: Monitor identifier
            monitor_name: Monitor name
            workflow_id: Associated workflow ID
            project_id: Associated project ID
            created_by: User ID who created the session
            metadata: Additional metadata

        Returns:
            Created CaptureSession
        """
        session = CaptureSession(
            session_id=session_id,
            video_width=video_width,
            video_height=video_height,
            video_fps=video_fps,
            monitor_id=monitor_id,
            monitor_name=monitor_name,
            workflow_id=workflow_id,
            project_id=project_id,
            created_by=created_by,
            metadata_json=metadata,
            storage_backend=(
                StorageBackend.LOCAL
                if isinstance(self.storage, type(get_default_storage()))
                else StorageBackend.S3
            ),
        )

        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        return session

    async def complete_session(
        self,
        session_id: str,
        video_data: BinaryIO,
        video_filename: str,
        total_frames: int | None = None,
        snapshot_run_id: int | None = None,
    ) -> CaptureSession:
        """Complete a capture session and store the video.

        Args:
            session_id: Session identifier
            video_data: Video file data
            video_filename: Video filename
            total_frames: Total frame count (if known)
            snapshot_run_id: Associated snapshot run ID

        Returns:
            Updated CaptureSession
        """
        session = self.db.query(CaptureSession).filter_by(session_id=session_id).first()

        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Store video
        stored_file = await self.storage.store_video(session_id, video_data, video_filename)

        # Update session
        session.ended_at = datetime.utcnow()  # type: ignore[assignment]
        session.duration_ms = int((session.ended_at - session.started_at).total_seconds() * 1000)  # type: ignore[assignment]
        session.video_path = stored_file.path  # type: ignore[assignment]
        session.video_size_bytes = stored_file.size_bytes  # type: ignore[assignment]
        session.total_frames = total_frames  # type: ignore[assignment]
        session.snapshot_run_id = snapshot_run_id  # type: ignore[assignment]
        session.is_complete = True  # type: ignore[assignment]

        self.db.commit()
        self.db.refresh(session)

        return session

    async def build_frame_index(self, session_id: str) -> int:
        """Build frame index for a capture session.

        Args:
            session_id: Session identifier

        Returns:
            Number of frames indexed
        """
        session = self.db.query(CaptureSession).filter_by(session_id=session_id).first()

        if not session or not session.video_path:
            raise ValueError(f"Session not found or no video: {session_id}")

        # Get video filename from path
        video_filename = session.video_path.split("/")[-1]

        # Build index using FFmpeg
        frames = await self.frame_extractor.build_frame_index(
            session_id,
            video_filename,
            keyframes_only=False,  # Index all frames for accurate seeking
        )

        # Store frame index in database
        for frame_info in frames:
            frame_index = FrameIndex(
                capture_session_id=session.id,
                frame_number=frame_info["frame_number"],
                timestamp_ms=frame_info["timestamp_ms"],
                byte_offset=frame_info.get("byte_offset"),
                is_keyframe=frame_info["is_keyframe"],
            )
            self.db.add(frame_index)

        session.is_processed = True  # type: ignore[assignment]
        self.db.commit()

        return len(frames)

    # =========================================================================
    # Input Event Recording
    # =========================================================================

    async def record_input_events(self, session_id: str, events: list[dict]) -> int:
        """Record input events for a capture session.

        Args:
            session_id: Session identifier
            events: List of event dictionaries

        Returns:
            Number of events recorded
        """
        session = self.db.query(CaptureSession).filter_by(session_id=session_id).first()

        if not session:
            raise ValueError(f"Session not found: {session_id}")

        for event_data in events:
            event = InputEvent(
                capture_session_id=session.id,
                timestamp_ms=event_data["timestamp_ms"],
                event_type=InputEventType(event_data["event_type"]),
                mouse_x=event_data.get("mouse_x"),
                mouse_y=event_data.get("mouse_y"),
                mouse_button=event_data.get("mouse_button"),
                scroll_dx=event_data.get("scroll_dx"),
                scroll_dy=event_data.get("scroll_dy"),
                key_code=event_data.get("key_code"),
                key_name=event_data.get("key_name"),
                key_char=event_data.get("key_char"),
                shift_pressed=event_data.get("shift_pressed", False),
                ctrl_pressed=event_data.get("ctrl_pressed", False),
                alt_pressed=event_data.get("alt_pressed", False),
                meta_pressed=event_data.get("meta_pressed", False),
                event_data_json=event_data.get("extra_data"),
            )
            self.db.add(event)

        self.db.commit()
        return len(events)

    async def get_input_events(
        self,
        session_id: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        event_types: list[str] | None = None,
    ) -> list[InputEvent]:
        """Get input events for a capture session.

        Args:
            session_id: Session identifier
            start_ms: Start timestamp filter (inclusive)
            end_ms: End timestamp filter (inclusive)
            event_types: Filter by event types

        Returns:
            List of InputEvent objects
        """
        session = self.db.query(CaptureSession).filter_by(session_id=session_id).first()

        if not session:
            raise ValueError(f"Session not found: {session_id}")

        query = self.db.query(InputEvent).filter_by(capture_session_id=session.id)

        if start_ms is not None:
            query = query.filter(InputEvent.timestamp_ms >= start_ms)

        if end_ms is not None:
            query = query.filter(InputEvent.timestamp_ms <= end_ms)

        if event_types:
            type_enums = [InputEventType(t) for t in event_types]
            query = query.filter(InputEvent.event_type.in_(type_enums))

        return query.order_by(InputEvent.timestamp_ms).all()

    # =========================================================================
    # Historical Result Management
    # =========================================================================

    async def index_historical_results(
        self, snapshot_run_id: int, capture_session_id: int | None = None
    ) -> int:
        """Index historical results from a snapshot run.

        Creates HistoricalResult entries for efficient querying
        during integration testing.

        Args:
            snapshot_run_id: Snapshot run ID to index
            capture_session_id: Associated capture session ID

        Returns:
            Number of results indexed
        """
        # Get snapshot run with actions
        snapshot_run = self.db.query(SnapshotRun).filter_by(id=snapshot_run_id).first()

        if not snapshot_run:
            raise ValueError(f"Snapshot run not found: {snapshot_run_id}")

        # Get all actions for this run
        actions = self.db.query(SnapshotAction).filter_by(snapshot_run_id=snapshot_run_id).all()

        count = 0
        for action in actions:
            # Extract data from action_data_json
            action_data: dict[str, Any] = action.action_data_json or {}  # type: ignore[assignment]

            # Get best match info if available
            matches = action_data.get("matches", [])
            best_match = matches[0] if matches else {}

            # Calculate frame timestamp (approximate from action timestamp)
            frame_timestamp_ms = None
            if capture_session_id:
                session = self.db.query(CaptureSession).filter_by(id=capture_session_id).first()
                if session:
                    # Calculate offset from session start
                    action_time = action.timestamp
                    session_start = session.started_at
                    if action_time and session_start:
                        delta = action_time - session_start
                        frame_timestamp_ms = int(delta.total_seconds() * 1000)

            historical = HistoricalResult(
                snapshot_run_id=snapshot_run_id,
                snapshot_action_id=action.id,
                capture_session_id=capture_session_id,
                pattern_id=action.pattern_id,
                pattern_name=action.pattern_name,
                action_type=action.action_type,
                active_states=action.active_states,
                success=action.success,
                match_count=action.match_count,
                best_match_score=best_match.get("score"),
                duration_ms=action.duration_ms,
                match_x=best_match.get("x"),
                match_y=best_match.get("y"),
                match_width=best_match.get("width"),
                match_height=best_match.get("height"),
                frame_timestamp_ms=frame_timestamp_ms,
                result_data_json=action_data,
                workflow_id=snapshot_run.workflow_id,
                project_id=None,  # Set from workflow if needed
                recorded_at=action.timestamp,
            )
            self.db.add(historical)
            count += 1

        self.db.commit()
        return count

    async def get_random_historical_result(
        self,
        pattern_id: str | None = None,
        action_type: str | None = None,
        active_states: list[str] | None = None,
        success_only: bool = True,
        workflow_id: int | None = None,
        project_id: int | None = None,
    ) -> HistoricalResult | None:
        """Get a random historical result matching criteria.

        This is the key method for integration testing - it returns
        a random result from historical data, making each test
        run different.

        Args:
            pattern_id: Filter by pattern ID
            action_type: Filter by action type
            active_states: Filter by active states (any match)
            success_only: Only return successful results
            workflow_id: Filter by workflow ID
            project_id: Filter by project ID

        Returns:
            Random HistoricalResult or None if no matches
        """
        query = self.db.query(HistoricalResult)

        if pattern_id:
            query = query.filter(HistoricalResult.pattern_id == pattern_id)

        if action_type:
            query = query.filter(HistoricalResult.action_type == action_type)

        if success_only:
            query = query.filter(HistoricalResult.success is True)  # type: ignore[arg-type]  # type: ignore[arg-type]

        if workflow_id:
            query = query.filter(HistoricalResult.workflow_id == workflow_id)

        if project_id:
            query = query.filter(HistoricalResult.project_id == project_id)

        if active_states:
            # Match any of the active states
            query = query.filter(HistoricalResult.active_states.overlap(active_states))  # type: ignore[attr-defined]  # type: ignore[attr-defined]

        # Get count and select random
        count = query.count()
        if count == 0:
            return None

        # Random offset for selection
        offset = random.randint(0, count - 1)
        return query.offset(offset).first()

    async def get_historical_results_for_action(
        self,
        pattern_id: str,
        action_type: str,
        active_states: list[str] | None = None,
        limit: int = 10,
    ) -> list[HistoricalResult]:
        """Get all historical results for an action context.

        Args:
            pattern_id: Pattern ID
            action_type: Action type
            active_states: Active states filter
            limit: Maximum results to return

        Returns:
            List of matching HistoricalResult objects
        """
        query = self.db.query(HistoricalResult).filter(
            HistoricalResult.pattern_id == pattern_id,
            HistoricalResult.action_type == action_type,
        )

        if active_states:
            query = query.filter(HistoricalResult.active_states.overlap(active_states))  # type: ignore[attr-defined]

        return query.order_by(HistoricalResult.recorded_at.desc()).limit(limit).all()

    # =========================================================================
    # Frame Extraction for Playback
    # =========================================================================

    async def get_frame_for_action(
        self, historical_result_id: int, frame_type: str = "action"
    ) -> bytes | None:
        """Get the frame image for a historical result.

        Args:
            historical_result_id: Historical result ID
            frame_type: Frame type ('before', 'action', 'after', 'result')

        Returns:
            Frame image data as bytes, or None if not available
        """
        result = self.db.query(HistoricalResult).filter_by(id=historical_result_id).first()

        if not result or not result.capture_session_id:
            return None

        # Check for cached action frame
        action_frame = (
            self.db.query(ActionFrame)
            .filter_by(
                capture_session_id=result.capture_session_id,
                snapshot_action_id=result.snapshot_action_id,
                frame_type=frame_type,
            )
            .first()
        )

        if action_frame and action_frame.cached_frame_path:
            return await self.storage.get_frame_data(str(action_frame.cached_frame_path))

        # Extract from video
        session = self.db.query(CaptureSession).filter_by(id=result.capture_session_id).first()

        if not session or not session.video_path:
            return None

        # Get video filename
        video_filename = str(session.video_path).split("/")[-1]

        # Extract frame at the action timestamp
        timestamp_ms = result.frame_timestamp_ms
        if not timestamp_ms:
            return None

        frame_data = await self.frame_extractor.extract_frame_by_timestamp(
            str(session.session_id), video_filename, int(timestamp_ms)
        )

        return frame_data

    async def get_frames_for_integration_test(self, historical_result_ids: list[int]) -> list[dict]:
        """Get frames for a sequence of integration test results.

        Args:
            historical_result_ids: List of historical result IDs in order

        Returns:
            List of dicts with frame data and metadata
        """
        frames = []

        for result_id in historical_result_ids:
            result = self.db.query(HistoricalResult).filter_by(id=result_id).first()

            if not result:
                continue

            frame_data = await self.get_frame_for_action(result_id)

            frames.append(
                {
                    "historical_result_id": result_id,
                    "action_type": result.action_type,
                    "pattern_id": result.pattern_id,
                    "pattern_name": result.pattern_name,
                    "success": result.success,
                    "match_x": result.match_x,
                    "match_y": result.match_y,
                    "match_width": result.match_width,
                    "match_height": result.match_height,
                    "timestamp_ms": result.frame_timestamp_ms,
                    "frame_data": frame_data,
                    "has_frame": frame_data is not None,
                }
            )

        return frames

    # =========================================================================
    # Session Queries
    # =========================================================================

    async def get_session(self, session_id: str) -> CaptureSession | None:
        """Get a capture session by ID."""
        return self.db.query(CaptureSession).filter_by(session_id=session_id).first()

    async def get_sessions_for_workflow(
        self, workflow_id: int, limit: int = 50
    ) -> list[CaptureSession]:
        """Get capture sessions for a workflow."""
        return (
            self.db.query(CaptureSession)
            .filter_by(workflow_id=workflow_id, is_complete=True)
            .order_by(CaptureSession.started_at.desc())
            .limit(limit)
            .all()
        )

    async def get_sessions_for_project(
        self, project_id: int, limit: int = 100
    ) -> list[CaptureSession]:
        """Get capture sessions for a project."""
        return (
            self.db.query(CaptureSession)
            .filter_by(project_id=project_id, is_complete=True)
            .order_by(CaptureSession.started_at.desc())
            .limit(limit)
            .all()
        )

    def get_random_historical_result_sync(
        self,
        pattern_id: str | None = None,
        action_type: str | None = None,
        active_states: list[str] | None = None,
        success_only: bool = True,
        workflow_id: int | None = None,
        project_id: int | None = None,
    ) -> HistoricalResult | None:
        """Synchronous version of get_random_historical_result.

        Used when called from non-async contexts.
        """
        query = self.db.query(HistoricalResult)

        if pattern_id:
            query = query.filter(HistoricalResult.pattern_id == pattern_id)

        if action_type:
            query = query.filter(HistoricalResult.action_type == action_type)

        if success_only:
            query = query.filter(HistoricalResult.success is True)  # type: ignore[arg-type]

        if workflow_id:
            query = query.filter(HistoricalResult.workflow_id == workflow_id)

        if project_id:
            query = query.filter(HistoricalResult.project_id == project_id)

        if active_states:
            query = query.filter(HistoricalResult.active_states.overlap(active_states))  # type: ignore[attr-defined]

        count = query.count()
        if count == 0:
            return None

        offset = random.randint(0, count - 1)
        return query.offset(offset).first()


# Factory function for dependency injection
_capture_service: Optional["CaptureService"] = None


def get_capture_service(db: Session | None = None) -> CaptureService:
    """Get or create a CaptureService instance.

    Args:
        db: SQLAlchemy database session (creates new if not provided)

    Returns:
        CaptureService instance
    """
    global _capture_service

    if db is not None:
        # Create new service with provided session
        return CaptureService(db)

    # Use cached service with fresh session
    if _capture_service is None:
        from app.core.database import SessionLocal

        _capture_service = CaptureService(SessionLocal())

    return _capture_service
