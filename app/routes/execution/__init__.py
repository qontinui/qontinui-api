"""FastAPI routers for workflow execution endpoints.

This package contains focused router modules for different execution API endpoint groups:
- health_router: Health check and version endpoints
- execution_router: Execution control endpoints (start, pause, resume, cancel, step)
- history_router: Execution history endpoints
- state_router: State management endpoints
- websocket_router: WebSocket streaming endpoint

These routers were migrated from qontinui core library (Phase 2: Core Library Cleanup).
The ExecutionManager and execution logic remain in qontinui - only HTTP routing moved here.

NO backward compatibility - clean FastAPI router pattern.
"""

from . import (execution_router, health_router, history_router, state_router,
               websocket_router)

__all__ = [
    "health_router",
    "execution_router",
    "history_router",
    "state_router",
    "websocket_router",
]
