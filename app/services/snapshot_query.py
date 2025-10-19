"""Query service for advanced snapshot search and analytics."""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.models.snapshot import (
    SnapshotAction,
    SnapshotPattern,
    SnapshotRun,
)
from app.services.filter_builder import FilterBuilder


class SnapshotQueryService:
    """Service for advanced snapshot queries, search, and analytics."""

    def __init__(self, db: Session):
        """Initialize the query service.

        Args:
            db: Database session
        """
        self.db = db
        self.filter_builder = FilterBuilder()

    def search_snapshots(
        self,
        query_text: str | None = None,
        filters: dict[str, Any] | None = None,
        sort_by: str = "start_time",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SnapshotRun], int]:
        """
        Search snapshots with advanced filtering.

        Args:
            query_text: Text search in metadata (optional)
            filters: Dictionary of filter specifications
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (matching snapshots, total count)
        """
        # Start with base query
        query = self.db.query(SnapshotRun)

        # Apply text search if provided
        if query_text:
            # Search in run_id, execution_mode, and metadata_json
            search_pattern = f"%{query_text}%"
            query = query.filter(
                func.or_(
                    SnapshotRun.run_id.ilike(search_pattern),
                    SnapshotRun.execution_mode.ilike(search_pattern),
                    SnapshotRun.notes.ilike(search_pattern),
                    # JSONB text search
                    func.cast(SnapshotRun.metadata_json, func.Text()).ilike(search_pattern),
                )
            )

        # Apply filters
        if filters:
            query = self.filter_builder.apply_filters(query, filters)

        # Get total count before pagination
        total = query.count()

        # Apply sorting
        query = self.filter_builder.apply_sorting(query, sort_by, sort_order)

        # Apply pagination
        query = self.filter_builder.apply_pagination(query, limit, offset)

        # Execute query
        snapshots = query.all()

        return snapshots, total

    def get_snapshot_statistics(self, run_id: str) -> dict[str, Any] | None:
        """
        Get detailed statistics for a single snapshot.

        Args:
            run_id: Snapshot run ID

        Returns:
            Statistics dictionary or None if not found
        """
        # Get snapshot run
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            return None

        # Get action breakdown
        action_breakdown = self._get_action_breakdown(snapshot.id)

        # Get pattern usage
        pattern_usage = self._get_pattern_usage(snapshot.id)

        # Get timeline
        timeline = self._get_action_timeline(snapshot.id)

        # Get performance metrics
        performance = self._get_performance_metrics(snapshot.id)

        return {
            "run_id": run_id,
            "action_breakdown": action_breakdown,
            "pattern_usage": pattern_usage,
            "timeline": timeline,
            "performance": performance,
        }

    def _get_action_breakdown(self, snapshot_run_id: int) -> dict[str, dict[str, int]]:
        """Get action counts broken down by type and success."""
        results = (
            self.db.query(
                SnapshotAction.action_type,
                SnapshotAction.success,
                func.count(SnapshotAction.id).label("count"),
            )
            .filter_by(snapshot_run_id=snapshot_run_id)
            .group_by(SnapshotAction.action_type, SnapshotAction.success)
            .all()
        )

        breakdown = {}
        for action_type, success, count in results:
            if action_type not in breakdown:
                breakdown[action_type] = {"total": 0, "successful": 0, "failed": 0}

            breakdown[action_type]["total"] += count
            if success:
                breakdown[action_type]["successful"] += count
            else:
                breakdown[action_type]["failed"] += count

        return breakdown

    def _get_pattern_usage(self, snapshot_run_id: int) -> list[dict[str, Any]]:
        """Get pattern usage statistics for a snapshot."""
        patterns = self.db.query(SnapshotPattern).filter_by(snapshot_run_id=snapshot_run_id).all()

        usage = []
        for pattern in patterns:
            success_rate = (
                float(pattern.successful_finds) / pattern.total_finds
                if pattern.total_finds > 0
                else 0.0
            )

            usage.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "uses": pattern.total_finds,
                    "success_rate": round(success_rate, 3),
                    "avg_duration_ms": (
                        float(pattern.avg_duration_ms) if pattern.avg_duration_ms else 0.0
                    ),
                    "total_matches": pattern.total_matches,
                }
            )

        # Sort by usage count
        usage.sort(key=lambda x: x["uses"], reverse=True)

        return usage

    def _get_action_timeline(self, snapshot_run_id: int) -> list[dict[str, Any]]:
        """Get chronological action timeline."""
        actions = (
            self.db.query(SnapshotAction)
            .filter_by(snapshot_run_id=snapshot_run_id)
            .order_by(SnapshotAction.sequence_number)
            .all()
        )

        timeline = []
        for action in actions:
            timeline.append(
                {
                    "sequence": action.sequence_number,
                    "timestamp": action.timestamp.isoformat(),
                    "action_type": action.action_type,
                    "pattern_id": action.pattern_id,
                    "pattern_name": action.pattern_name,
                    "success": action.success,
                    "duration_ms": float(action.duration_ms) if action.duration_ms else 0.0,
                    "match_count": action.match_count,
                    "active_states": action.active_states or [],
                }
            )

        return timeline

    def _get_performance_metrics(self, snapshot_run_id: int) -> dict[str, Any]:
        """Get performance metrics for a snapshot."""
        snapshot = self.db.query(SnapshotRun).filter_by(id=snapshot_run_id).first()
        if not snapshot:
            return {}

        # Get action durations
        actions = (
            self.db.query(SnapshotAction.duration_ms)
            .filter_by(snapshot_run_id=snapshot_run_id)
            .filter(SnapshotAction.duration_ms.isnot(None))
            .all()
        )

        durations = [float(a.duration_ms) for a in actions if a.duration_ms]

        if not durations:
            return {
                "total_duration_seconds": (
                    float(snapshot.duration_seconds) if snapshot.duration_seconds else 0.0
                ),
                "avg_action_duration_ms": 0.0,
                "slowest_action_ms": 0.0,
                "fastest_action_ms": 0.0,
            }

        return {
            "total_duration_seconds": (
                float(snapshot.duration_seconds) if snapshot.duration_seconds else 0.0
            ),
            "avg_action_duration_ms": round(sum(durations) / len(durations), 2),
            "slowest_action_ms": round(max(durations), 2),
            "fastest_action_ms": round(min(durations), 2),
        }

    def get_pattern_analytics(
        self,
        pattern_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        workflow_id: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Get analytics for a specific pattern across snapshots.

        Args:
            pattern_id: Pattern identifier
            start_date: Filter snapshots from this date
            end_date: Filter snapshots until this date
            workflow_id: Filter by workflow

        Returns:
            Analytics dictionary or None if pattern not found
        """
        # Build query with filters
        query = self.db.query(SnapshotPattern).filter_by(pattern_id=pattern_id)

        # Join with snapshot_runs for filtering
        query = query.join(SnapshotRun, SnapshotPattern.snapshot_run_id == SnapshotRun.id)

        if start_date:
            query = query.filter(SnapshotRun.start_time >= start_date)
        if end_date:
            query = query.filter(SnapshotRun.start_time <= end_date)
        if workflow_id:
            query = query.filter(SnapshotRun.workflow_id == workflow_id)

        patterns = query.all()

        if not patterns:
            return None

        # Aggregate statistics
        total_snapshots = len(patterns)
        total_finds = sum(p.total_finds for p in patterns)
        successful_finds = sum(p.successful_finds for p in patterns)
        total_matches = sum(p.total_matches for p in patterns)

        durations = [float(p.avg_duration_ms) for p in patterns if p.avg_duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        success_rate = successful_finds / total_finds if total_finds > 0 else 0.0
        match_rate = total_matches / total_finds if total_finds > 0 else 0.0

        # Get pattern name from first result
        pattern_name = patterns[0].pattern_name if patterns else pattern_id

        # Build time series for reliability trend
        reliability_trend = self._get_pattern_reliability_trend(
            pattern_id, start_date, end_date, workflow_id
        )

        return {
            "pattern_id": pattern_id,
            "pattern_name": pattern_name,
            "total_snapshots": total_snapshots,
            "total_finds": total_finds,
            "successful_finds": successful_finds,
            "success_rate": round(success_rate, 3),
            "avg_duration_ms": round(avg_duration, 2),
            "match_rate": round(match_rate, 2),
            "reliability_trend": reliability_trend,
        }

    def _get_pattern_reliability_trend(
        self,
        pattern_id: str,
        start_date: datetime | None,
        end_date: datetime | None,
        workflow_id: int | None,
    ) -> list[dict[str, Any]]:
        """Get daily reliability trend for a pattern."""
        # Query pattern usage by date
        query = (
            self.db.query(
                func.date(SnapshotRun.start_time).label("date"),
                func.count(SnapshotPattern.id).label("usage_count"),
                func.sum(SnapshotPattern.successful_finds).label("successful"),
                func.sum(SnapshotPattern.total_finds).label("total"),
            )
            .join(SnapshotRun, SnapshotPattern.snapshot_run_id == SnapshotRun.id)
            .filter(SnapshotPattern.pattern_id == pattern_id)
        )

        if start_date:
            query = query.filter(SnapshotRun.start_time >= start_date)
        if end_date:
            query = query.filter(SnapshotRun.start_time <= end_date)
        if workflow_id:
            query = query.filter(SnapshotRun.workflow_id == workflow_id)

        query = query.group_by(func.date(SnapshotRun.start_time)).order_by(
            func.date(SnapshotRun.start_time).desc()
        )

        results = query.limit(30).all()  # Last 30 days

        trend = []
        for date, usage_count, successful, total in results:
            success_rate = successful / total if total > 0 else 0.0
            trend.append(
                {
                    "date": date.isoformat(),
                    "usage_count": usage_count,
                    "success_rate": round(success_rate, 3),
                }
            )

        return trend

    def get_execution_trends(
        self,
        period_days: int = 30,
        group_by: str = "day",
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get execution trends over time.

        Args:
            period_days: Number of days to analyze
            group_by: Grouping interval ("day", "week")
            filters: Additional filters

        Returns:
            Trends dictionary with time series data
        """
        start_date = datetime.now() - timedelta(days=period_days)

        # Base query
        query = self.db.query(SnapshotRun).filter(SnapshotRun.start_time >= start_date)

        # Apply additional filters
        if filters:
            query = self.filter_builder.apply_filters(query, filters)

        # Get success rate trend
        success_rate_trend = self._get_success_rate_trend(query, group_by)

        # Get execution count trend
        execution_count_trend = self._get_execution_count_trend(query, group_by)

        # Get common failures
        common_failures = self._get_common_failures(start_date, filters)

        # Get pattern usage trend
        pattern_usage_trend = self._get_pattern_usage_trend(start_date, filters)

        return {
            "period_days": period_days,
            "group_by": group_by,
            "success_rate_trend": success_rate_trend,
            "execution_count_trend": execution_count_trend,
            "common_failures": common_failures,
            "pattern_usage_trend": pattern_usage_trend,
        }

    def _get_success_rate_trend(self, base_query, group_by: str) -> list[dict[str, Any]]:
        """Get success rate trend from base query."""
        date_func = func.date(SnapshotRun.start_time)

        results = (
            base_query.with_entities(
                date_func.label("date"),
                func.count(SnapshotRun.id).label("total_runs"),
                func.sum(
                    case((SnapshotRun.successful_actions == SnapshotRun.total_actions, 1), else_=0)
                ).label("successful_runs"),
                func.avg(
                    func.cast(SnapshotRun.successful_actions, func.Float())
                    / func.nullif(SnapshotRun.total_actions, 0)
                ).label("avg_success_rate"),
            )
            .group_by(date_func)
            .order_by(date_func.desc())
            .limit(30)
            .all()
        )

        trend = []
        for date, total_runs, successful_runs, avg_success_rate in results:
            trend.append(
                {
                    "date": date.isoformat(),
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "success_rate": round(float(avg_success_rate or 0.0), 3),
                }
            )

        return trend

    def _get_execution_count_trend(self, base_query, group_by: str) -> list[dict[str, Any]]:
        """Get execution count by mode over time."""
        date_func = func.date(SnapshotRun.start_time)

        results = (
            base_query.with_entities(
                date_func.label("date"),
                SnapshotRun.execution_mode,
                func.count(SnapshotRun.id).label("count"),
            )
            .group_by(date_func, SnapshotRun.execution_mode)
            .order_by(date_func.desc())
            .limit(100)
            .all()
        )

        # Group by date
        by_date = {}
        for date, mode, count in results:
            date_str = date.isoformat()
            if date_str not in by_date:
                by_date[date_str] = {"date": date_str, "total": 0}
            by_date[date_str][mode] = count
            by_date[date_str]["total"] += count

        return list(by_date.values())

    def _get_common_failures(
        self, start_date: datetime, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Get most commonly failing patterns."""
        query = (
            self.db.query(
                SnapshotPattern.pattern_id,
                SnapshotPattern.pattern_name,
                func.sum(SnapshotPattern.failed_finds).label("failure_count"),
                func.count(func.distinct(SnapshotPattern.snapshot_run_id)).label("affected_runs"),
            )
            .join(SnapshotRun, SnapshotPattern.snapshot_run_id == SnapshotRun.id)
            .filter(SnapshotRun.start_time >= start_date)
            .filter(SnapshotPattern.failed_finds > 0)
        )

        if filters:
            query = self.filter_builder.apply_filters(query, filters)

        results = (
            query.group_by(SnapshotPattern.pattern_id, SnapshotPattern.pattern_name)
            .order_by(func.sum(SnapshotPattern.failed_finds).desc())
            .limit(10)
            .all()
        )

        failures = []
        for pattern_id, pattern_name, failure_count, affected_runs in results:
            failures.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_name": pattern_name,
                    "failure_count": failure_count,
                    "affected_runs": affected_runs,
                }
            )

        return failures

    def _get_pattern_usage_trend(
        self, start_date: datetime, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Get most used patterns in period."""
        query = (
            self.db.query(
                SnapshotPattern.pattern_id,
                SnapshotPattern.pattern_name,
                func.count(SnapshotPattern.id).label("usage_count"),
            )
            .join(SnapshotRun, SnapshotPattern.snapshot_run_id == SnapshotRun.id)
            .filter(SnapshotRun.start_time >= start_date)
        )

        if filters:
            query = self.filter_builder.apply_filters(query, filters)

        results = (
            query.group_by(SnapshotPattern.pattern_id, SnapshotPattern.pattern_name)
            .order_by(func.count(SnapshotPattern.id).desc())
            .limit(10)
            .all()
        )

        usage = []
        for pattern_id, pattern_name, usage_count in results:
            usage.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_name": pattern_name,
                    "usage_count": usage_count,
                }
            )

        return usage
