"""Filter builder for constructing SQLAlchemy queries from filter dictionaries."""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, func
from sqlalchemy.orm import Query

from app.models.snapshot import SnapshotRun


class FilterBuilder:
    """Build SQLAlchemy query filters from dictionary specifications."""

    @staticmethod
    def apply_filters(query: Query, filters: dict[str, Any]) -> Query:
        """
        Apply filters to a SQLAlchemy query.

        Args:
            query: Base SQLAlchemy query
            filters: Dictionary of filter specifications

        Returns:
            Query with filters applied

        Filter Format:
            {
                "field_name": {
                    "operator": value
                },
                "field_name": value  # Shorthand for {"eq": value}
            }

        Supported Operators:
            - eq: Equal to
            - ne: Not equal to
            - gt: Greater than
            - gte: Greater than or equal
            - lt: Less than
            - lte: Less than or equal
            - in: In list
            - not_in: Not in list
            - like: SQL LIKE (case-sensitive)
            - ilike: SQL LIKE (case-insensitive)
            - contains: Array contains (for array fields)
            - contained_by: Array contained by (for array fields)
            - overlap: Array overlap (for array fields)
            - is_null: Is null (boolean value)
            - jsonb_contains: JSONB contains (@>)
            - jsonb_contained: JSONB contained by (<@)
            - jsonb_has_key: JSONB has key (?)
        """
        if not filters:
            return query

        filter_conditions = []

        for field_name, filter_spec in filters.items():
            # Handle shorthand: {"field": value} -> {"field": {"eq": value}}
            if not isinstance(filter_spec, dict):
                filter_spec = {"eq": filter_spec}

            # Get the model field
            if not hasattr(SnapshotRun, field_name):
                continue  # Skip unknown fields

            field = getattr(SnapshotRun, field_name)

            # Apply each operator
            for operator, value in filter_spec.items():
                condition = FilterBuilder._build_condition(field, field_name, operator, value)
                if condition is not None:
                    filter_conditions.append(condition)

        # Combine all conditions with AND
        if filter_conditions:
            query = query.filter(and_(*filter_conditions))

        return query

    @staticmethod
    def _build_condition(field, field_name: str, operator: str, value: Any):
        """Build a single filter condition."""
        # Comparison operators
        if operator == "eq":
            return field == value

        elif operator == "ne":
            return field != value

        elif operator == "gt":
            return field > value

        elif operator == "gte":
            return field >= value

        elif operator == "lt":
            return field < value

        elif operator == "lte":
            return field <= value

        # List operators
        elif operator == "in":
            if not isinstance(value, list | tuple):
                value = [value]
            return field.in_(value)

        elif operator == "not_in":
            if not isinstance(value, list | tuple):
                value = [value]
            return field.notin_(value)

        # String operators
        elif operator == "like":
            return field.like(value)

        elif operator == "ilike":
            return field.ilike(value)

        # Null check
        elif operator == "is_null":
            if value:
                return field.is_(None)
            else:
                return field.isnot(None)

        # Array operators (PostgreSQL)
        elif operator == "contains":
            # Array contains value (for ARRAY columns)
            if not isinstance(value, list):
                value = [value]
            return field.contains(value)

        elif operator == "contained_by":
            # Array is contained by value
            if not isinstance(value, list):
                value = [value]
            return field.contained_by(value)

        elif operator == "overlap":
            # Array overlaps with value
            if not isinstance(value, list):
                value = [value]
            return field.overlap(value)

        # JSONB operators (PostgreSQL)
        elif operator == "jsonb_contains":
            # JSONB contains value (@>)
            return field.contains(value)

        elif operator == "jsonb_contained":
            # JSONB contained by value (<@)
            return field.contained_by(value)

        elif operator == "jsonb_has_key":
            # JSONB has key (?)
            return field.has_key(value)

        # Unknown operator
        else:
            return None

    @staticmethod
    def apply_date_range_filter(
        query: Query,
        field_name: str,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
    ) -> Query:
        """
        Apply a date range filter to a query.

        Args:
            query: Base query
            field_name: Name of the datetime field
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            Query with date filter applied
        """
        if not hasattr(SnapshotRun, field_name):
            return query

        field = getattr(SnapshotRun, field_name)

        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        conditions = []
        if start_date is not None:
            conditions.append(field >= start_date)
        if end_date is not None:
            conditions.append(field <= end_date)

        if conditions:
            query = query.filter(and_(*conditions))

        return query

    @staticmethod
    def apply_success_rate_filter(
        query: Query,
        min_rate: float | None = None,
        max_rate: float | None = None,
    ) -> Query:
        """
        Apply success rate filter (calculated from successful_actions / total_actions).

        Args:
            query: Base query
            min_rate: Minimum success rate (0.0-1.0)
            max_rate: Maximum success rate (0.0-1.0)

        Returns:
            Query with success rate filter applied
        """
        conditions = []

        if min_rate is not None:
            # successful_actions / total_actions >= min_rate
            # Note: Use NULLIF to avoid division by zero
            success_rate = func.cast(SnapshotRun.successful_actions, func.Float()) / func.nullif(
                SnapshotRun.total_actions, 0
            )
            conditions.append(success_rate >= min_rate)

        if max_rate is not None:
            success_rate = func.cast(SnapshotRun.successful_actions, func.Float()) / func.nullif(
                SnapshotRun.total_actions, 0
            )
            conditions.append(success_rate <= max_rate)

        if conditions:
            query = query.filter(and_(*conditions))

        return query

    @staticmethod
    def apply_sorting(
        query: Query,
        sort_by: str = "start_time",
        sort_order: str = "desc",
    ) -> Query:
        """
        Apply sorting to a query.

        Args:
            query: Base query
            sort_by: Field name to sort by
            sort_order: "asc" or "desc"

        Returns:
            Query with sorting applied
        """
        if not hasattr(SnapshotRun, sort_by):
            # Default to start_time if field doesn't exist
            sort_by = "start_time"

        field = getattr(SnapshotRun, sort_by)

        if sort_order.lower() == "asc":
            query = query.order_by(field.asc())
        else:
            query = query.order_by(field.desc())

        return query

    @staticmethod
    def apply_pagination(
        query: Query,
        limit: int = 50,
        offset: int = 0,
    ) -> Query:
        """
        Apply pagination to a query.

        Args:
            query: Base query
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Query with pagination applied
        """
        # Clamp limit to reasonable range
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        return query.limit(limit).offset(offset)
