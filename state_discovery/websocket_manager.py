"""WebSocket connection manager for state discovery.

This module handles all WebSocket connections and message broadcasting.
Single Responsibility: Manage WebSocket connections and communication.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class ConnectionManager:
    """Manages WebSocket connections for analysis updates."""

    active_connections: dict[str, WebSocket] = field(default_factory=dict)

    async def connect(self, analysis_id: str, websocket: WebSocket):
        """Accept and store a new WebSocket connection."""
        try:
            await websocket.accept()
            self.active_connections[analysis_id] = websocket
            logger.info(f"WebSocket connected for analysis {analysis_id}")
            print(f"[WS] Connected: {analysis_id}")
        except Exception as e:
            logger.error(f"Failed to connect WebSocket for {analysis_id}: {e}")
            print(f"[WS] Connection failed for {analysis_id}: {e}")
            raise

    def disconnect(self, analysis_id: str):
        """Remove a WebSocket connection."""
        if analysis_id in self.active_connections:
            del self.active_connections[analysis_id]
            logger.info(f"WebSocket disconnected for analysis {analysis_id}")
            print(f"[WS] Disconnected: {analysis_id}")

    async def send_update(self, analysis_id: str, message: dict[str, Any]):
        """Send an update message to a specific connection."""
        if analysis_id not in self.active_connections:
            logger.warning(f"No active connection for analysis {analysis_id}")
            print(f"[WS] No connection for {analysis_id}")
            return

        websocket = self.active_connections[analysis_id]
        try:
            message_str = json.dumps(message)
            await websocket.send_text(message_str)
            logger.debug(f"Sent update to {analysis_id}: {message.get('type')}")
            print(f"[WS] Sent to {analysis_id}: {message.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send update to {analysis_id}: {e}")
            print(f"[WS] Send failed to {analysis_id}: {e}")
            self.disconnect(analysis_id)

    async def send_progress(self, analysis_id: str, stage: str, percentage: int, message: str):
        """Send a progress update."""
        await self.send_update(
            analysis_id,
            {
                "type": "progress",
                "data": {"stage": stage, "percentage": percentage, "message": message},
            },
        )

    async def send_error(self, analysis_id: str, error_message: str):
        """Send an error message."""
        await self.send_update(analysis_id, {"type": "error", "data": {"message": error_message}})

    async def send_complete(self, analysis_id: str, result: dict[str, Any]):
        """Send analysis completion message."""
        await self.send_update(analysis_id, {"type": "complete", "data": result})


# Global instance
manager = ConnectionManager()
