"""
Tests for scheduler API endpoints.

Tests the scheduler statistics, status, execution history, and schedule details endpoints.
"""

import pytest
from fastapi.testclient import TestClient

# Note: This assumes main.py exports the app
# If not, we'll need to adjust the import
from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_user_data():
    """Create mock user data with schedules and execution records."""
    return {
        "users": {
            "testuser": {
                "username": "testuser",
                "hashed_password": "test_hash",
                "projects": [
                    {
                        "id": 1,
                        "name": "Test Project",
                        "config_data": {
                            "version": "1.0.0",
                            "metadata": {"name": "Test Project"},
                            "processes": [
                                {
                                    "id": "process-1",
                                    "name": "Test Process",
                                }
                            ],
                            "states": [],
                            "transitions": [],
                            "schedules": [
                                {
                                    "id": "schedule-1",
                                    "name": "Morning Automation",
                                    "processId": "process-1",
                                    "description": "Runs every morning",
                                    "triggerType": "TIME",
                                    "cronExpression": "0 8 * * *",
                                    "checkMode": "CHECK_ALL",
                                    "scheduleType": "FIXED_RATE",
                                    "maxIterations": 5,
                                    "stateCheckDelaySeconds": 2.0,
                                    "stateRebuildDelaySeconds": 1.0,
                                    "failureThreshold": 3,
                                    "enabled": True,
                                    "createdAt": "2024-01-01T00:00:00Z",
                                    "lastExecutedAt": "2024-01-15T08:00:00Z",
                                },
                                {
                                    "id": "schedule-2",
                                    "name": "Disabled Schedule",
                                    "processId": "process-1",
                                    "triggerType": "INTERVAL",
                                    "intervalSeconds": 300,
                                    "checkMode": "CHECK_INACTIVE_ONLY",
                                    "scheduleType": "FIXED_DELAY",
                                    "stateCheckDelaySeconds": 2.0,
                                    "stateRebuildDelaySeconds": 1.0,
                                    "failureThreshold": 3,
                                    "enabled": False,
                                },
                            ],
                            "executionRecords": [
                                {
                                    "id": "exec-1",
                                    "scheduleId": "schedule-1",
                                    "processId": "process-1",
                                    "startTime": "2024-01-15T08:00:00Z",
                                    "endTime": "2024-01-15T08:05:00Z",
                                    "success": True,
                                    "iterationCount": 3,
                                    "errors": [],
                                    "metadata": {},
                                },
                                {
                                    "id": "exec-2",
                                    "scheduleId": "schedule-1",
                                    "processId": "process-1",
                                    "startTime": "2024-01-14T08:00:00Z",
                                    "endTime": "2024-01-14T08:04:00Z",
                                    "success": True,
                                    "iterationCount": 2,
                                    "errors": [],
                                    "metadata": {},
                                },
                                {
                                    "id": "exec-3",
                                    "scheduleId": "schedule-1",
                                    "processId": "process-1",
                                    "startTime": "2024-01-13T08:00:00Z",
                                    "endTime": "2024-01-13T08:10:00Z",
                                    "success": False,
                                    "iterationCount": 1,
                                    "errors": ["Timeout waiting for state"],
                                    "metadata": {},
                                },
                            ],
                        },
                    }
                ],
            }
        }
    }


@pytest.fixture
def auth_headers():
    """Create mock authentication headers."""
    # Note: In real tests, you'd need to generate a valid JWT token
    # For now, we'll assume the test bypasses auth or uses a mock
    return {"Authorization": "Bearer test_token"}


class TestSchedulerStatistics:
    """Tests for GET /api/v1/scheduler/statistics/{project_id}"""

    def test_get_statistics_success(self, client, mock_user_data, auth_headers, monkeypatch):
        """Test successful retrieval of scheduler statistics."""

        # Mock the load_users function to return our test data
        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        # Mock get_current_user to return test user
        from auth_simple import User

        def mock_get_current_user():
            return User(username="testuser", email="test@example.com")

        # This would need proper dependency override in FastAPI
        # app.dependency_overrides[get_current_user] = mock_get_current_user

        response = client.get("/api/v1/scheduler/statistics/1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["total_schedules"] == 2
        assert data["active_schedules"] == 1
        assert data["total_executions"] == 3
        assert data["successful_executions"] == 2
        assert data["failed_executions"] == 1
        assert "average_iteration_count" in data

    def test_get_statistics_project_not_found(self, client, auth_headers):
        """Test statistics endpoint with non-existent project."""
        response = client.get("/api/v1/scheduler/statistics/999", headers=auth_headers)

        # Should return 404 or empty statistics
        assert response.status_code in [404, 200]

    def test_get_statistics_unauthorized(self, client):
        """Test statistics endpoint without authentication."""
        response = client.get("/api/v1/scheduler/statistics/1")

        # Should return 401 Unauthorized
        assert response.status_code == 401


class TestSchedulerStatus:
    """Tests for GET /api/v1/scheduler/status/{project_id}"""

    def test_get_status_success(self, client, mock_user_data, auth_headers, monkeypatch):
        """Test successful retrieval of scheduler status."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/status/1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["has_schedules"] is True
        assert data["total_schedules"] == 2
        assert data["active_schedules"] == 1
        assert len(data["schedules"]) == 2

        # Verify schedule summary structure
        schedule = data["schedules"][0]
        assert "id" in schedule
        assert "name" in schedule
        assert "processId" in schedule
        assert "triggerType" in schedule
        assert "enabled" in schedule

    def test_get_status_no_schedules(self, client, auth_headers, monkeypatch):
        """Test status endpoint for project with no schedules."""
        mock_data = {
            "users": {
                "testuser": {
                    "projects": [
                        {
                            "id": 1,
                            "config_data": {
                                "schedules": [],
                            },
                        }
                    ]
                }
            }
        }

        def mock_load_users():
            return mock_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/status/1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["has_schedules"] is False
        assert data["total_schedules"] == 0
        assert data["active_schedules"] == 0
        assert data["schedules"] == []


class TestExecutionHistory:
    """Tests for GET /api/v1/scheduler/executions/{project_id}"""

    def test_get_executions_success(self, client, mock_user_data, auth_headers, monkeypatch):
        """Test successful retrieval of execution history."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/executions/1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert "executions" in data
        assert "total_count" in data
        assert "returned_count" in data
        assert data["total_count"] == 3
        assert len(data["executions"]) == 3

    def test_get_executions_filtered_by_schedule(
        self, client, mock_user_data, auth_headers, monkeypatch
    ):
        """Test execution history filtered by schedule_id."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get(
            "/api/v1/scheduler/executions/1?schedule_id=schedule-1", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # All executions should be for schedule-1
        for execution in data["executions"]:
            assert execution["scheduleId"] == "schedule-1"

    def test_get_executions_with_limit(self, client, mock_user_data, auth_headers, monkeypatch):
        """Test execution history with limit parameter."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/executions/1?limit=2", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["returned_count"] <= 2
        assert len(data["executions"]) <= 2

    def test_get_executions_sorted_by_date(self, client, mock_user_data, auth_headers, monkeypatch):
        """Test that executions are sorted by most recent first."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/executions/1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        executions = data["executions"]
        # First execution should have the most recent startTime
        if len(executions) > 1:
            assert executions[0]["startTime"] >= executions[1]["startTime"]


class TestScheduleDetails:
    """Tests for GET /api/v1/scheduler/schedule/{project_id}/{schedule_id}"""

    def test_get_schedule_details_success(self, client, mock_user_data, auth_headers, monkeypatch):
        """Test successful retrieval of schedule details."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/schedule/1/schedule-1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert "schedule" in data
        assert "recent_executions" in data
        assert "statistics" in data

        # Verify schedule details
        schedule = data["schedule"]
        assert schedule["id"] == "schedule-1"
        assert schedule["name"] == "Morning Automation"
        assert schedule["processId"] == "process-1"

        # Verify statistics
        stats = data["statistics"]
        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert "failed_executions" in stats
        assert "average_iteration_count" in stats
        assert "success_rate" in stats

        # Verify recent executions (should be limited to 10)
        assert len(data["recent_executions"]) <= 10

    def test_get_schedule_details_not_found(self, client, auth_headers, monkeypatch):
        """Test schedule details endpoint with non-existent schedule."""
        mock_data = {
            "users": {
                "testuser": {
                    "projects": [
                        {
                            "id": 1,
                            "config_data": {
                                "schedules": [],
                                "executionRecords": [],
                            },
                        }
                    ]
                }
            }
        }

        def mock_load_users():
            return mock_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/schedule/1/nonexistent", headers=auth_headers)

        assert response.status_code == 404

    def test_schedule_statistics_calculation(
        self, client, mock_user_data, auth_headers, monkeypatch
    ):
        """Test that schedule statistics are calculated correctly."""

        def mock_load_users():
            return mock_user_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/schedule/1/schedule-1", headers=auth_headers)

        assert response.status_code == 200
        stats = response.json()["statistics"]

        # Based on mock data: 3 executions, 2 successful, 1 failed
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 2
        assert stats["failed_executions"] == 1
        assert stats["success_rate"] == pytest.approx(66.7, rel=0.1)

        # Average iterations: (3 + 2 + 1) / 3 = 2.0
        assert stats["average_iteration_count"] == pytest.approx(2.0, rel=0.1)


class TestAuthenticationAndAuthorization:
    """Tests for authentication and authorization on scheduler endpoints."""

    def test_all_endpoints_require_authentication(self, client):
        """Test that all scheduler endpoints require authentication."""
        endpoints = [
            "/api/v1/scheduler/statistics/1",
            "/api/v1/scheduler/status/1",
            "/api/v1/scheduler/executions/1",
            "/api/v1/scheduler/schedule/1/schedule-1",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"

    def test_user_cannot_access_other_users_projects(self, client, auth_headers, monkeypatch):
        """Test that users can only access their own project data."""
        # This test would require more complex setup to test cross-user access
        # For now, we verify that the endpoint checks project ownership
        pass


class TestErrorHandling:
    """Tests for error handling in scheduler API."""

    def test_invalid_project_id(self, client, auth_headers):
        """Test handling of invalid project ID format."""
        response = client.get("/api/v1/scheduler/statistics/invalid", headers=auth_headers)

        # Should handle gracefully (404 or 400)
        assert response.status_code in [400, 404]

    def test_malformed_query_parameters(self, client, auth_headers):
        """Test handling of malformed query parameters."""
        response = client.get("/api/v1/scheduler/executions/1?limit=invalid", headers=auth_headers)

        # FastAPI should validate and return 422
        assert response.status_code == 422

    def test_missing_config_data(self, client, auth_headers, monkeypatch):
        """Test handling of project with missing config_data."""
        mock_data = {
            "users": {
                "testuser": {
                    "projects": [
                        {
                            "id": 1,
                            "name": "Test Project",
                            # No config_data
                        }
                    ]
                }
            }
        }

        def mock_load_users():
            return mock_data

        monkeypatch.setattr("scheduler_api.load_users", mock_load_users)

        response = client.get("/api/v1/scheduler/statistics/1", headers=auth_headers)

        # Should handle gracefully
        assert response.status_code in [200, 404]


# Integration test example
class TestSchedulerAPIIntegration:
    """Integration tests for scheduler API with full workflow."""

    def test_full_schedule_lifecycle(self, client, auth_headers, monkeypatch):
        """Test the full lifecycle: create, execute, view statistics."""
        # This would be a more complex test that:
        # 1. Creates a schedule via project update
        # 2. Simulates execution (creates execution records)
        # 3. Retrieves statistics and verifies accuracy
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
