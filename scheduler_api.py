"""
Scheduler API router for qontinui-api.

Provides REST endpoints for scheduler statistics and monitoring.
Schedules are stored as part of project configurations, so this router
focuses on statistics and execution history retrieval.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from auth_simple import User, get_current_user, load_users

router = APIRouter(tags=["scheduler"])


@router.get("/scheduler/statistics/{project_id}")
async def get_scheduler_statistics(
    project_id: str, current_user: User = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Get scheduler statistics for a specific project.

    Returns:
        - total_schedules: Total number of schedules
        - active_schedules: Number of enabled schedules
        - total_executions: Total execution count
        - successful_executions: Number of successful executions
        - failed_executions: Number of failed executions
        - average_iteration_count: Average iterations per execution
    """
    users_data = load_users()
    user = users_data["users"].get(current_user.username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find the project
    project = None
    for proj in user.get("projects", []):
        if proj.get("id") == project_id:
            project = proj
            break

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get schedules and execution records from project configuration
    config_data = project.get("config_data", {})
    schedules = config_data.get("schedules", [])
    execution_records = config_data.get("executionRecords", [])

    # Calculate statistics
    total_schedules = len(schedules)
    active_schedules = sum(1 for s in schedules if s.get("enabled", False))

    total_executions = len(execution_records)
    successful_executions = sum(1 for r in execution_records if r.get("success", False))
    failed_executions = total_executions - successful_executions

    # Calculate average iteration count
    average_iteration_count = 0.0
    if execution_records:
        total_iterations = sum(r.get("iterationCount", 0) for r in execution_records)
        average_iteration_count = total_iterations / len(execution_records)

    return {
        "total_schedules": total_schedules,
        "active_schedules": active_schedules,
        "total_executions": total_executions,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "average_iteration_count": round(average_iteration_count, 2),
    }


@router.get("/scheduler/status/{project_id}")
async def get_scheduler_status(
    project_id: str, current_user: User = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Get scheduler status for a specific project.

    Note: The actual scheduler running state is managed by qontinui-runner.
    This endpoint returns configuration-level status.

    Returns:
        - has_schedules: Whether the project has any schedules configured
        - total_schedules: Total number of schedules
        - active_schedules: Number of enabled schedules
        - schedules: List of schedule summaries
    """
    users_data = load_users()
    user = users_data["users"].get(current_user.username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find the project
    project = None
    for proj in user.get("projects", []):
        if proj.get("id") == project_id:
            project = proj
            break

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get schedules from project configuration
    config_data = project.get("config_data", {})
    schedules = config_data.get("schedules", [])

    # Create schedule summaries
    schedule_summaries = []
    for schedule in schedules:
        schedule_summaries.append(
            {
                "id": schedule.get("id"),
                "name": schedule.get("name"),
                "processId": schedule.get("processId"),
                "triggerType": schedule.get("triggerType"),
                "enabled": schedule.get("enabled", False),
                "lastExecutedAt": schedule.get("lastExecutedAt"),
            }
        )

    return {
        "has_schedules": len(schedules) > 0,
        "total_schedules": len(schedules),
        "active_schedules": sum(1 for s in schedules if s.get("enabled", False)),
        "schedules": schedule_summaries,
    }


@router.get("/scheduler/executions/{project_id}")
async def get_execution_history(
    project_id: str,
    schedule_id: str | None = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get execution history for a specific project.

    Args:
        project_id: The project ID
        schedule_id: Optional schedule ID to filter by specific schedule
        limit: Maximum number of records to return (default 50)

    Returns:
        - executions: List of execution records
        - total_count: Total number of execution records
    """
    users_data = load_users()
    user = users_data["users"].get(current_user.username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find the project
    project = None
    for proj in user.get("projects", []):
        if proj.get("id") == project_id:
            project = proj
            break

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get execution records from project configuration
    config_data = project.get("config_data", {})
    execution_records = config_data.get("executionRecords", [])

    # Filter by schedule_id if provided
    if schedule_id:
        execution_records = [r for r in execution_records if r.get("scheduleId") == schedule_id]

    # Sort by startTime (most recent first)
    try:
        execution_records = sorted(
            execution_records, key=lambda r: r.get("startTime", ""), reverse=True
        )
    except Exception:
        # If sorting fails, just use the original order
        pass

    # Apply limit
    limited_records = execution_records[:limit]

    return {
        "executions": limited_records,
        "total_count": len(execution_records),
        "returned_count": len(limited_records),
    }


@router.get("/scheduler/schedule/{project_id}/{schedule_id}")
async def get_schedule_details(
    project_id: str, schedule_id: str, current_user: User = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Get detailed information about a specific schedule.

    Returns:
        - schedule: The schedule configuration
        - recent_executions: Recent execution records for this schedule
        - statistics: Schedule-specific statistics
    """
    users_data = load_users()
    user = users_data["users"].get(current_user.username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find the project
    project = None
    for proj in user.get("projects", []):
        if proj.get("id") == project_id:
            project = proj
            break

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get schedules and execution records
    config_data = project.get("config_data", {})
    schedules = config_data.get("schedules", [])
    execution_records = config_data.get("executionRecords", [])

    # Find the specific schedule
    schedule = None
    for s in schedules:
        if s.get("id") == schedule_id:
            schedule = s
            break

    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Get executions for this schedule
    schedule_executions = [r for r in execution_records if r.get("scheduleId") == schedule_id]

    # Sort and limit to recent executions
    try:
        schedule_executions = sorted(
            schedule_executions, key=lambda r: r.get("startTime", ""), reverse=True
        )[
            :10
        ]  # Last 10 executions
    except Exception:
        schedule_executions = schedule_executions[:10]

    # Calculate schedule-specific statistics
    total_executions = len([r for r in execution_records if r.get("scheduleId") == schedule_id])
    successful = sum(
        1
        for r in execution_records
        if r.get("scheduleId") == schedule_id and r.get("success", False)
    )
    failed = total_executions - successful

    avg_iterations = 0.0
    if schedule_executions:
        total_iters = sum(r.get("iterationCount", 0) for r in schedule_executions)
        avg_iterations = total_iters / len(schedule_executions)

    return {
        "schedule": schedule,
        "recent_executions": schedule_executions,
        "statistics": {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "average_iteration_count": round(avg_iterations, 2),
            "success_rate": round(
                (successful / total_executions * 100) if total_executions > 0 else 0, 1
            ),
        },
    }
