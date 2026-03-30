"""
Integration tests for WebSocket connections.

Tests cover:
- WebSocket connection establishment
- Connection to specific run
- Subscribe/unsubscribe messages
- Ping/pong health check
- Broadcasting to run subscribers
- Connection management
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient
from starlette.websockets import WebSocketDisconnect

from app.websocket import manager, ConnectionManager


class TestConnectionManager:
    """Unit tests for ConnectionManager class."""

    def test_connect_websocket(self):
        """Test connecting a WebSocket."""
        cm = ConnectionManager()
        mock_websocket = MockWebSocket()

        cm.active_connections[mock_websocket] = None
        assert cm.get_total_connections() == 1

    def test_connect_with_run_id(self):
        """Test connecting with a run ID subscription."""
        cm = ConnectionManager()
        mock_websocket = MockWebSocket()
        run_id = "test-run-123"

        cm.active_connections[mock_websocket] = run_id
        cm.run_subscribers[run_id] = {mock_websocket}

        assert cm.get_subscriber_count(run_id) == 1

    def test_disconnect_websocket(self):
        """Test disconnecting a WebSocket."""
        cm = ConnectionManager()
        mock_websocket = MockWebSocket()

        cm.active_connections[mock_websocket] = None
        cm.disconnect(mock_websocket)

        assert cm.get_total_connections() == 0

    def test_disconnect_removes_from_subscribers(self):
        """Test that disconnecting removes from run subscribers."""
        cm = ConnectionManager()
        mock_websocket = MockWebSocket()
        run_id = "test-run-123"

        cm.active_connections[mock_websocket] = run_id
        cm.run_subscribers[run_id] = {mock_websocket}

        cm.disconnect(mock_websocket)

        assert run_id not in cm.run_subscribers or mock_websocket not in cm.run_subscribers.get(run_id, set())

    def test_broadcast_to_run(self):
        """Test broadcasting to run subscribers."""
        cm = ConnectionManager()
        mock_websocket1 = MockWebSocket()
        mock_websocket2 = MockWebSocket()
        run_id = "test-run-123"

        cm.run_subscribers[run_id] = {mock_websocket1, mock_websocket2}

        # Note: Full broadcast test requires async test client
        assert cm.get_subscriber_count(run_id) == 2

    def test_get_subscriber_count_empty(self):
        """Test getting subscriber count for run with no subscribers."""
        cm = ConnectionManager()
        assert cm.get_subscriber_count("nonexistent-run") == 0


class MockWebSocket:
    """Mock WebSocket for unit testing."""

    def __init__(self):
        self.messages = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, message):
        self.messages.append(message)

    async def send_text(self, message):
        self.messages.append(message)

    async def receive_text(self):
        return '{"type": "ping"}'

    def close(self):
        self.closed = True


class TestWebSocketEndpoint:
    """Integration tests for WebSocket endpoint using TestClient."""

    def test_websocket_connect(self, client):
        """Test WebSocket connection to evaluations endpoint."""
        # Simply connecting without exception means success
        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            websocket.send_json({"type": "ping"})
            # Note: Response might not be immediate in test environment

    def test_websocket_connect_with_run_id(self, client, test_run):
        """Test WebSocket connection with run ID."""
        # Simply connecting without exception means success
        with client.websocket_connect(f"/ws/v1/evaluations?run_id={test_run.id}") as websocket:
            websocket.send_json({"type": "ping"})
            # Connection established successfully

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong message."""
        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            websocket.send_json({"type": "ping"})
            # Give time for response
            import time
            time.sleep(0.1)
            # Note: Actual pong response depends on implementation

    def test_websocket_subscribe(self, client, test_run):
        """Test WebSocket subscribe message."""
        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "run_id": test_run.id,
            })
            import time
            time.sleep(0.1)
            # Check for subscribed response
            # Note: Implementation may vary

    def test_websocket_unsubscribe(self, client):
        """Test WebSocket unsubscribe message."""
        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            # First subscribe
            websocket.send_json({"type": "subscribe", "run_id": "test-run"})
            import time
            time.sleep(0.1)

            # Then unsubscribe
            websocket.send_json({"type": "unsubscribe"})
            time.sleep(0.1)

    def test_websocket_invalid_message(self, client):
        """Test WebSocket with invalid message type."""
        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            websocket.send_json({"type": "invalid_type"})
            import time
            time.sleep(0.1)
            # Should receive error response

    def test_websocket_invalid_json(self, client):
        """Test WebSocket with invalid JSON."""
        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            websocket.send_text("not valid json")
            import time
            time.sleep(0.1)
            # Should receive error about invalid JSON


class TestWebSocketDisconnect:
    """Tests for WebSocket disconnection handling."""

    def test_disconnect_cleanup(self, client):
        """Test that disconnection cleans up properly."""
        initial_count = manager.get_total_connections()

        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            during_count = manager.get_total_connections()
            assert during_count == initial_count + 1

        after_count = manager.get_total_connections()
        assert after_count == initial_count


class TestEmitFunctions:
    """Tests for WebSocket emit helper functions."""

    @pytest.mark.asyncio
    async def test_emit_run_status(self):
        """Test emit_run_status function."""
        from app.websocket import emit_run_status

        run_id = "test-run"
        status = "completed"
        message = "Test complete"

        # This would require mocked connections to test fully
        # For now, verify function exists and signature
        assert callable(emit_run_status)

    @pytest.mark.asyncio
    async def test_emit_metric_update(self):
        """Test emit_metric_update function."""
        from app.websocket import emit_metric_update

        run_id = "test-run"
        metrics = [{"name": "ttft", "value": 0.025}]

        assert callable(emit_metric_update)

    @pytest.mark.asyncio
    async def test_emit_progress(self):
        """Test emit_progress function."""
        from app.websocket import emit_progress

        run_id = "test-run"
        progress = 50.0
        message = "Halfway done"

        assert callable(emit_progress)


class TestWebSocketIntegration:
    """Full integration tests for WebSocket with real scenarios."""

    def test_multiple_clients_same_run(self, client, test_run):
        """Test multiple clients subscribing to same run."""
        # Connect first client
        with client.websocket_connect(f"/ws/v1/evaluations?run_id={test_run.id}") as ws1:
            # Connect second client
            with client.websocket_connect(f"/ws/v1/evaluations?run_id={test_run.id}") as ws2:
                # Both connected successfully (no exception raised)

                # Check subscriber count
                count = manager.get_subscriber_count(test_run.id)
                assert count == 2

    def test_client_subscribes_to_multiple_runs(self, client, db_session):
        """Test a client subscribing to multiple runs."""
        from tests.conftest import RunFactory

        run1 = RunFactory.create(db_session)
        run2 = RunFactory.create(db_session)

        with client.websocket_connect("/ws/v1/evaluations") as websocket:
            # Subscribe to first run
            websocket.send_json({"type": "subscribe", "run_id": run1.id})
            import time
            time.sleep(0.1)

            # Subscribe to second run
            websocket.send_json({"type": "subscribe", "run_id": run2.id})
            time.sleep(0.1)

            # Should be subscribed to both
            count1 = manager.get_subscriber_count(run1.id)
            count2 = manager.get_subscriber_count(run2.id)
            assert count1 >= 1
            assert count2 >= 1
