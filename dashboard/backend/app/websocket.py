"""
WebSocket handler for real-time updates.

Supports:
- Run status updates
- Metrics streaming during evaluation
- Progress notifications
"""

import json
import logging
from typing import Any, Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Supports broadcasting messages to specific run subscribers.
    """

    def __init__(self):
        # Active connections: {connection: run_id}
        self.active_connections: Dict[WebSocket, str] = {}
        # Subscribers per run: {run_id: set of connections}
        self.run_subscribers: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str | None = None):
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            run_id: Optional run ID to subscribe to
        """
        try:
            await websocket.accept()
            self.active_connections[websocket] = run_id

            if run_id:
                if run_id not in self.run_subscribers:
                    self.run_subscribers[run_id] = set()
                self.run_subscribers[run_id].add(websocket)

            logger.info(f"WebSocket connected: run_id={run_id}, total_connections={len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            raise

    def disconnect(self, websocket: WebSocket):
        """
        Handle WebSocket disconnection.

        Args:
            websocket: The disconnected connection
        """
        try:
            run_id = self.active_connections.pop(websocket, None)

            if run_id and websocket in self.run_subscribers.get(run_id, set()):
                self.run_subscribers[run_id].remove(websocket)
                if not self.run_subscribers[run_id]:
                    del self.run_subscribers[run_id]

            logger.info(f"WebSocket disconnected: run_id={run_id}, total_connections={len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")

    async def broadcast_to_run(self, run_id: str, message: dict):
        """
        Broadcast a message to all subscribers of a run.

        Args:
            run_id: The run ID to broadcast to
            message: Message dict to send
        """
        if run_id in self.run_subscribers:
            # Create a copy since we might modify the set during iteration
            subscribers = self.run_subscribers[run_id].copy()
            for connection in subscribers:
                try:
                    await self.send_json(connection, message)
                except WebSocketDisconnect:
                    logger.debug(f"WebSocket disconnected while sending: run_id={run_id}")
                    self.disconnect(connection)
                except Exception as e:
                    logger.error(f"Error broadcasting to run {run_id}: {e}")
                    self.disconnect(connection)

    async def broadcast_evaluations(self, message: dict):
        """
        Broadcast to all evaluation subscribers.

        Args:
            message: Message dict to send
        """
        for connection in list(self.active_connections.keys()):
            try:
                await self.send_json(connection, message)
            except WebSocketDisconnect:
                logger.debug("WebSocket disconnected during evaluations broadcast")
                self.disconnect(connection)
            except Exception as e:
                logger.error(f"Error broadcasting evaluations: {e}")
                self.disconnect(connection)

    async def send_json(self, websocket: WebSocket, message: dict):
        """
        Send a JSON message to a specific connection.

        Args:
            websocket: Target connection
            message: Message dict to send
        """
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected while sending JSON")
            raise
        except Exception as e:
            logger.error(f"Error sending JSON message: {e}")
            raise

    async def send_text(self, websocket: WebSocket, message: str):
        """
        Send a text message to a specific connection.

        Args:
            websocket: Target connection
            message: Message string to send
        """
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected while sending text")
            raise
        except Exception as e:
            logger.error(f"Error sending text message: {e}")
            raise

    def get_subscriber_count(self, run_id: str) -> int:
        """Get number of subscribers for a run."""
        return len(self.run_subscribers.get(run_id, set()))

    def get_total_connections(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, run_id: str | None = None):
    """
    WebSocket endpoint for real-time updates.

    Connect to /ws/v1/evaluations?run_id=<id> to subscribe to specific run updates.
    Connect to /ws/v1/evaluations to receive all evaluation updates.
    """
    await manager.connect(websocket, run_id)

    try:
        while True:
            # Handle incoming messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_client_message(websocket, message)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data[:100]}")
                await websocket.send_json({"error": "Invalid JSON"})
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected normally: run_id={run_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Always clean up connection
        manager.disconnect(websocket)
        logger.info(f"WebSocket cleanup completed: run_id={run_id}")


async def handle_client_message(websocket: WebSocket, message: dict):
    """
    Handle incoming messages from WebSocket clients.

    Supported message types:
    - subscribe: Subscribe to a run's updates
    - unsubscribe: Unsubscribe from a run
    - ping: Health check
    """
    try:
        msg_type = message.get("type")

        if msg_type == "subscribe":
            run_id = message.get("run_id")
            if run_id:
                manager.active_connections[websocket] = run_id
                if run_id not in manager.run_subscribers:
                    manager.run_subscribers[run_id] = set()
                manager.run_subscribers[run_id].add(websocket)
                await websocket.send_json({
                    "type": "subscribed",
                    "run_id": run_id,
                })
                logger.debug(f"Client subscribed to run: {run_id}")

        elif msg_type == "unsubscribe":
            run_id = manager.active_connections.get(websocket)
            if run_id and run_id in manager.run_subscribers:
                manager.run_subscribers[run_id].discard(websocket)
            manager.active_connections[websocket] = None
            await websocket.send_json({"type": "unsubscribed"})
            logger.debug(f"Client unsubscribed from run: {run_id}")

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

        else:
            logger.warning(f"Unknown message type received: {msg_type}")
            await websocket.send_json({
                "error": f"Unknown message type: {msg_type}",
            })
    except WebSocketDisconnect:
        logger.debug("Client disconnected while handling message")
        raise
    except Exception as e:
        logger.error(f"Error handling client message: {e}")
        try:
            await websocket.send_json({"error": f"Failed to process message: {str(e)}"})
        except Exception:
            pass  # Connection might be closed


# Helper functions for emitting events


async def emit_run_status(
    run_id: str,
    status: str,
    message: str | None = None,
    data: dict | None = None,
):
    """
    Emit a run status update event.

    Args:
        run_id: The run ID
        status: New status (pending, running, completed, failed)
        message: Optional status message
        data: Optional additional data
    """
    event = {
        "event_type": "run_status",
        "run_id": run_id,
        "status": status,
        "message": message,
    }
    if data:
        event["data"] = data

    await manager.broadcast_to_run(run_id, event)


async def emit_metric_update(
    run_id: str,
    metrics: list[dict],
):
    """
    Emit a metrics update event.

    Args:
        run_id: The run ID
        metrics: List of metric data dicts
    """
    event = {
        "event_type": "metrics_stream",
        "run_id": run_id,
        "metrics": metrics,
    }

    await manager.broadcast_to_run(run_id, event)


async def emit_progress(
    run_id: str,
    progress: float,
    message: str | None = None,
):
    """
    Emit a progress update event.

    Args:
        run_id: The run ID
        progress: Progress percentage (0-100)
        message: Optional progress message
    """
    event = {
        "event_type": "progress",
        "run_id": run_id,
        "progress": progress,
        "message": message,
    }

    await manager.broadcast_to_run(run_id, event)
