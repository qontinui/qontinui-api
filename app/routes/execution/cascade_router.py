"""Cascade detection events SSE endpoint.

Streams CascadeDetector events (started, backend_tried, hit, miss) to
subscribers in real time via Server-Sent Events. The supervisor dashboard
consumes this to show cascade detection activity.
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cascade", tags=["cascade"])

# In-memory event buffer — recent cascade events for late-joining clients
_event_buffer: deque[dict[str, Any]] = deque(maxlen=200)

# Async queue for push-based streaming to SSE clients
_subscribers: list[asyncio.Queue[dict[str, Any]]] = []


def _on_cascade_event(event: Any) -> None:
    """Callback registered with qontinui's EventRegistry.

    Receives cascade events from the Python event system and pushes
    them to all active SSE subscribers.
    """
    data = {
        "type": event.type.value,
        "data": event.data,
        "timestamp": event.timestamp,
    }
    _event_buffer.append(data)

    # Push to all active subscribers (non-blocking)
    dead: list[asyncio.Queue] = []
    for q in _subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.remove(q)


def _register_event_callbacks() -> None:
    """Register our callback with the qontinui event system.

    Called once at import time. Safe to call multiple times.
    """
    try:
        from qontinui.reporting.events import EventType, register_callback

        for et in (
            EventType.CASCADE_STARTED,
            EventType.CASCADE_BACKEND_TRIED,
            EventType.CASCADE_HIT,
            EventType.CASCADE_MISS,
        ):
            register_callback(et, _on_cascade_event)
        logger.info("Cascade event callbacks registered")
    except ImportError:
        logger.debug("qontinui.reporting not available, cascade events disabled")
    except Exception as e:
        logger.warning(f"Failed to register cascade callbacks: {e}")


# Register on module load
_register_event_callbacks()


@router.get("/events")
async def get_recent_events():
    """Return recent cascade events as JSON array."""
    return list(_event_buffer)


@router.get("/stream")
async def stream_cascade_events():
    """Stream cascade events via Server-Sent Events.

    Sends recent events as a burst, then streams new events in real time.
    Each SSE message has event type "cascade" and JSON data payload.
    """

    async def event_generator():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=50)
        _subscribers.append(queue)
        try:
            # Send recent events as initial burst
            for evt in _event_buffer:
                yield f"event: cascade\ndata: {json.dumps(evt)}\n\n"

            # Stream new events
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: cascade\ndata: {json.dumps(evt)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield f": keepalive {time.time()}\n\n"
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
