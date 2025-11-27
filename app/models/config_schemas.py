"""Pydantic schemas for snapshot configuration validation.

This module provides Pydantic models for validating snapshot data structures
loaded from JSON files. These schemas ensure type safety and data validation
when syncing snapshot data to the database.

Key schemas:
- SnapshotMetadataSchema: Validates metadata.json files
- ActionLogItemSchema: Validates individual action log entries
- PatternMetadataSchema: Validates pattern metadata files
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StatisticsSchema(BaseModel):
    """Statistics section of snapshot metadata."""

    total_actions: int = Field(ge=0, description="Total number of actions executed")
    successful_actions: int = Field(ge=0, description="Number of successful actions")
    failed_actions: int = Field(ge=0, description="Number of failed actions")
    total_screenshots: int = Field(ge=0, description="Total number of screenshots captured")

    @field_validator("failed_actions")
    @classmethod
    def validate_failed_actions(cls, v: int, info) -> int:
        """Ensure failed_actions + successful_actions <= total_actions."""
        if "total_actions" in info.data:
            total = info.data["total_actions"]
            successful = info.data.get("successful_actions", 0)
            if v + successful > total:
                raise ValueError(
                    f"failed_actions ({v}) + successful_actions ({successful}) "
                    f"cannot exceed total_actions ({total})"
                )
        return v


class SnapshotMetadataSchema(BaseModel):
    """Schema for snapshot metadata.json files.

    This validates the top-level metadata file that contains information about
    the entire snapshot run.
    """

    run_id: str = Field(min_length=1, description="Unique identifier for this snapshot run")
    start_time: str = Field(description="ISO 8601 timestamp when snapshot started")
    end_time: str | None = Field(None, description="ISO 8601 timestamp when snapshot ended")
    duration_seconds: float | None = Field(None, ge=0, description="Total duration in seconds")
    execution_mode: str = Field(
        min_length=1, description="Execution mode (e.g., 'live', 'mock', 'screenshot')"
    )
    statistics: StatisticsSchema = Field(description="Statistics about the snapshot run")
    patterns: dict[str, Any] = Field(
        default_factory=dict, description="Pattern information by pattern ID"
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        """Validate timestamp is in ISO 8601 format."""
        if v is None:
            return v
        try:
            datetime.fromisoformat(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e
        return v

    @field_validator("end_time")
    @classmethod
    def validate_end_after_start(cls, v: str | None, info) -> str | None:
        """Ensure end_time is after start_time if both are present."""
        if v is None or "start_time" not in info.data:
            return v

        start = datetime.fromisoformat(info.data["start_time"])
        end = datetime.fromisoformat(v)

        if end < start:
            raise ValueError(
                f"end_time ({v}) cannot be before start_time ({info.data['start_time']})"
            )

        return v

    model_config = {"extra": "allow"}  # Allow additional fields for extensibility


class ActionLogItemSchema(BaseModel):
    """Schema for individual action log entries from action_log.json.

    Each entry represents a single action execution during the snapshot run.
    """

    timestamp: str = Field(description="ISO 8601 timestamp when action was executed")
    action_type: str = Field(min_length=1, description="Type of action (e.g., 'FIND', 'CLICK')")
    pattern_id: str | None = Field(None, description="ID of pattern used, if applicable")
    pattern_name: str | None = Field(None, description="Name of pattern used, if applicable")
    success: bool = Field(default=True, description="Whether action completed successfully")
    match_count: int | None = Field(None, ge=0, description="Number of matches found")
    duration_ms: float | None = Field(None, ge=0, description="Action duration in milliseconds")
    active_states: list[str] = Field(
        default_factory=list, description="List of active state IDs during action"
    )
    screenshot_path: str | None = Field(
        None, description="Path to screenshot associated with action"
    )
    is_start_screenshot: bool = Field(
        default=False, description="Whether this is the initial screenshot"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is in ISO 8601 format."""
        try:
            datetime.fromisoformat(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e
        return v

    model_config = {"extra": "allow"}  # Allow additional fields for extensibility


class MatchHistoryItemSchema(BaseModel):
    """Schema for individual match items in pattern history."""

    x: int = Field(ge=0, description="X coordinate of match")
    y: int = Field(ge=0, description="Y coordinate of match")
    width: int = Field(gt=0, description="Width of match region")
    height: int = Field(gt=0, description="Height of match region")
    score: float | None = Field(None, ge=0, le=1, description="Match confidence score (0-1)")

    model_config = {"extra": "allow"}  # Allow additional fields for extensibility


class PatternMetadataSchema(BaseModel):
    """Schema for pattern metadata files (patterns/{pattern-id}/metadata.json).

    Contains aggregated statistics about pattern usage during the snapshot run.
    """

    pattern_name: str = Field(min_length=1, description="Human-readable pattern name")
    total_finds: int = Field(ge=0, description="Total number of find attempts")
    successful_finds: int = Field(ge=0, description="Number of successful finds")
    total_matches: int = Field(ge=0, description="Total number of matches found")
    avg_duration_ms: float | None = Field(
        None, ge=0, description="Average duration in milliseconds"
    )

    @field_validator("successful_finds")
    @classmethod
    def validate_successful_finds(cls, v: int, info) -> int:
        """Ensure successful_finds <= total_finds."""
        if "total_finds" in info.data and v > info.data["total_finds"]:
            raise ValueError(
                f"successful_finds ({v}) cannot exceed total_finds ({info.data['total_finds']})"
            )
        return v

    model_config = {"extra": "allow"}  # Allow additional fields for extensibility


class ValidationError(Exception):
    """Exception raised when snapshot data validation fails.

    Attributes:
        message: Human-readable error message
        errors: List of validation errors from Pydantic
        file_path: Path to file that failed validation (if applicable)
    """

    def __init__(
        self, message: str, errors: list[dict[str, Any]] | None = None, file_path: str | None = None
    ):
        """Initialize validation error.

        Args:
            message: Human-readable error message
            errors: List of validation errors from Pydantic
            file_path: Path to file that failed validation
        """
        self.message = message
        self.errors = errors or []
        self.file_path = file_path
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of error."""
        error_details = "\n".join(
            f"  - {err.get('loc', '')}: {err.get('msg', '')}" for err in self.errors
        )
        file_info = f" in {self.file_path}" if self.file_path else ""
        return f"{self.message}{file_info}\n{error_details}" if error_details else self.message


def validate_snapshot_metadata(
    data: dict[str, Any], file_path: str | None = None
) -> SnapshotMetadataSchema:
    """Validate snapshot metadata against schema.

    Args:
        data: Raw metadata dictionary from JSON
        file_path: Optional file path for error reporting

    Returns:
        Validated SnapshotMetadataSchema instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        return SnapshotMetadataSchema.model_validate(data)
    except Exception as e:
        errors = e.errors() if hasattr(e, "errors") else [{"msg": str(e)}]
        raise ValidationError(
            "Snapshot metadata validation failed", errors=errors, file_path=file_path
        ) from e


def validate_action_log_item(
    data: dict[str, Any], file_path: str | None = None
) -> ActionLogItemSchema:
    """Validate action log item against schema.

    Args:
        data: Raw action log item dictionary from JSON
        file_path: Optional file path for error reporting

    Returns:
        Validated ActionLogItemSchema instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        return ActionLogItemSchema.model_validate(data)
    except Exception as e:
        errors = e.errors() if hasattr(e, "errors") else [{"msg": str(e)}]
        raise ValidationError(
            "Action log item validation failed", errors=errors, file_path=file_path
        ) from e


def validate_pattern_metadata(
    data: dict[str, Any], file_path: str | None = None
) -> PatternMetadataSchema:
    """Validate pattern metadata against schema.

    Args:
        data: Raw pattern metadata dictionary from JSON
        file_path: Optional file path for error reporting

    Returns:
        Validated PatternMetadataSchema instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        return PatternMetadataSchema.model_validate(data)
    except Exception as e:
        errors = e.errors() if hasattr(e, "errors") else [{"msg": str(e)}]
        raise ValidationError(
            "Pattern metadata validation failed", errors=errors, file_path=file_path
        ) from e


__all__ = [
    "SnapshotMetadataSchema",
    "ActionLogItemSchema",
    "PatternMetadataSchema",
    "MatchHistoryItemSchema",
    "StatisticsSchema",
    "ValidationError",
    "validate_snapshot_metadata",
    "validate_action_log_item",
    "validate_pattern_metadata",
]
