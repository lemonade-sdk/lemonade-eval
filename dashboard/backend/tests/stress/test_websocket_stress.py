"""
Stress tests for WebSocket real-time updates.

Tests cover:
- High concurrent connection handling
- Message throughput under load
- Broadcast efficiency
- Connection stability during load
- Memory usage with many connections
"""

import pytest
import asyncio
import time
from typing import List, Dict
from unittest.mock import MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from app.websocket import ConnectionManager, manager, emit_run_status, emit_metric_update


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def connection_manager():
    """Create a fresh connection manager for testing."""
    return ConnectionManager()


@pytest.fixture
def mock_websockets():
    """Create mock WebSocket connections."""
    websockets = []
    for i in range(20):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        ws.send_text = AsyncMock()
        ws.close = MagicMock()
        websockets.append(ws)
    return websockets


# ============================================================================
# CONNECTION STRESS TESTS
# ============================================================================

class TestWebSocketConnectionStress:
    """Stress tests for WebSocket connections."""

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, connection_manager):
        """Test handling many concurrent connections."""
        num_connections = 50

        async def connect(i):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            ws.send_text = AsyncMock()
            await connection_manager.connect(ws, f"run-{i % 10}")

        # Connect all clients concurrently
        tasks = [connect(i) for i in range(num_connections)]
        await asyncio.gather(*tasks)

        # Verify all connections established
        assert connection_manager.get_total_connections() == num_connections

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self, connection_manager):
        """Test rapid connect/disconnect cycles."""
        async def connect_disconnect(i):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            ws.send_text = AsyncMock()
            await connection_manager.connect(ws, f"run-{i}")
            await asyncio.sleep(0.001)
            connection_manager.disconnect(ws)

        # Rapid connect/disconnect cycles
        tasks = [connect_disconnect(i) for i in range(100)]
        await asyncio.gather(*tasks)

        # All connections should be cleaned up
        assert connection_manager.get_total_connections() == 0

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, connection_manager):
        """Test connection cleanup when errors occur."""
        ws = MagicMock()
        ws.send_json = AsyncMock(side_effect=Exception("Connection error"))
        ws.send_text = AsyncMock()

        await connection_manager.connect(ws, "run-1")
        assert connection_manager.get_total_connections() == 1

        # Simulate error during send
        try:
            await connection_manager.send_json(ws, {"type": "test"})
        except Exception:
            pass

        # Connection should still be tracked (cleanup happens on disconnect)
        connection_manager.disconnect(ws)
        assert connection_manager.get_total_connections() == 0


# ============================================================================
# BROADCAST STRESS TESTS
# ============================================================================

class TestWebSocketBroadcastStress:
    """Stress tests for WebSocket broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_many_subscribers(self, connection_manager, mock_websockets):
        """Test broadcasting to many subscribers."""
        run_id = "test-run"

        # Connect all mock websockets to same run
        for ws in mock_websockets:
            ws.send_json = AsyncMock()
            await connection_manager.connect(ws, run_id)

        # Broadcast message
        message = {"type": "update", "data": "test"}
        await connection_manager.broadcast_to_run(run_id, message)

        # All subscribers should receive message
        for ws in mock_websockets:
            ws.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_broadcast_under_load(self, connection_manager):
        """Test broadcast performance under load."""
        run_id = "load-test-run"
        num_subscribers = 100

        # Create and connect many subscribers
        websockets = []
        for i in range(num_subscribers):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await connection_manager.connect(ws, run_id)

        # Measure broadcast time
        start = time.time()
        message = {"type": "metrics", "data": list(range(100))}
        await connection_manager.broadcast_to_run(run_id, message)
        elapsed = time.time() - start

        # Broadcast should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Broadcast took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(self, connection_manager, mock_websockets):
        """Test concurrent broadcasts to different runs."""
        # Connect websockets to different runs
        for i, ws in enumerate(mock_websockets):
            ws.send_json = AsyncMock()
            run_id = f"run-{i % 5}"  # 5 different runs
            await connection_manager.connect(ws, run_id)

        # Concurrent broadcasts
        async def broadcast(run_id, message):
            await connection_manager.broadcast_to_run(run_id, message)

        tasks = [
            broadcast(f"run-{i}", {"type": "update", "run": i})
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

        # All broadcasts should complete
        assert connection_manager.get_total_connections() == len(mock_websockets)


# ============================================================================
# EMIT FUNCTION TESTS
# ============================================================================

class TestWebSocketEmitFunctions:
    """Tests for WebSocket emit helper functions."""

    @pytest.mark.asyncio
    async def test_emit_run_status_performance(self):
        """Test emit_run_status performance."""
        run_id = "status-test-run"

        # Connect subscribers
        websockets = []
        for i in range(20):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await manager.connect(ws, run_id)

        # Measure emit performance
        start = time.time()
        for i in range(10):
            await emit_run_status(run_id, "running", message=f"Update {i}")
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0, f"Emit took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_emit_metric_update_performance(self):
        """Test emit_metric_update performance."""
        run_id = "metrics-test-run"

        # Connect subscribers
        websockets = []
        for i in range(20):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await manager.connect(ws, run_id)

        # Measure emit performance with large metric payload
        metrics = [{"name": f"metric-{i}", "value": i * 0.1} for i in range(50)]

        start = time.time()
        await emit_metric_update(run_id, metrics)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 2.0, f"Emit took {elapsed:.2f}s"


# ============================================================================
# MEMORY AND RESOURCE TESTS
# ============================================================================

class TestWebSocketResources:
    """Tests for WebSocket resource management."""

    @pytest.mark.asyncio
    async def test_memory_cleanup_after_disconnect(self, connection_manager):
        """Test memory cleanup after disconnections."""
        import gc

        # Connect many clients
        websockets = []
        for i in range(100):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await connection_manager.connect(ws, f"run-{i % 10}")

        # Disconnect all
        for ws in websockets:
            connection_manager.disconnect(ws)

        # Force garbage collection
        gc.collect()

        # All connections should be cleaned up
        assert connection_manager.get_total_connections() == 0
        assert len(connection_manager.run_subscribers) == 0

    @pytest.mark.asyncio
    async def test_subscriber_count_accuracy(self, connection_manager):
        """Test subscriber count accuracy under load."""
        run_id = "count-test"

        # Connect varying numbers of subscribers
        for i in range(50):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            await connection_manager.connect(ws, run_id)
            assert connection_manager.get_subscriber_count(run_id) == i + 1

        # Disconnect half
        for i, ws in enumerate(list(connection_manager.active_connections.keys())):
            if i % 2 == 0:
                connection_manager.disconnect(ws)

        expected = 25  # Half of 50
        assert connection_manager.get_subscriber_count(run_id) == expected


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestWebSocketErrorHandling:
    """Tests for WebSocket error handling under stress."""

    @pytest.mark.asyncio
    async def test_broadcast_with_failing_connections(self, connection_manager):
        """Test broadcast when some connections fail."""
        run_id = "failing-run"

        # Mix of working and failing connections
        for i in range(20):
            ws = MagicMock()
            if i % 3 == 0:
                # Every third connection fails
                ws.send_json = AsyncMock(side_effect=Exception("Connection lost"))
            else:
                ws.send_json = AsyncMock()
            await connection_manager.connect(ws, run_id)

        # Broadcast should complete despite failures
        message = {"type": "update"}
        await connection_manager.broadcast_to_run(run_id, message)

        # Failed connections should be removed
        remaining = connection_manager.get_subscriber_count(run_id)
        assert remaining < 20

    @pytest.mark.asyncio
    async def test_disconnect_during_broadcast(self, connection_manager):
        """Test disconnect happening during broadcast."""
        run_id = "concurrent-run"

        websockets = []
        for i in range(20):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await connection_manager.connect(ws, run_id)

        # Disconnect some during broadcast
        async def disconnect_half():
            await asyncio.sleep(0.01)
            for i, ws in enumerate(websockets):
                if i % 2 == 0:
                    connection_manager.disconnect(ws)

        # Broadcast and disconnect concurrently
        message = {"type": "update"}
        await asyncio.gather(
            connection_manager.broadcast_to_run(run_id, message),
            disconnect_half(),
        )


# ============================================================================
# REAL-TIME UPDATE SCENARIOS
# ============================================================================

class TestRealTimeUpdateScenarios:
    """Tests for real-time update scenarios."""

    @pytest.mark.asyncio
    async def test_metrics_streaming_scenario(self):
        """Test metrics streaming scenario."""
        run_id = "streaming-run"

        # Simulate metrics streaming during evaluation
        websockets = []
        for i in range(10):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await manager.connect(ws, run_id)

        # Stream metrics
        for i in range(50):
            metrics = [
                {"name": "seconds_to_first_token", "value": 0.025 + i * 0.001},
                {"name": "tokens_per_second", "value": 45.5 - i * 0.1},
            ]
            await emit_metric_update(run_id, metrics)

        # All subscribers should receive all updates
        for ws in websockets:
            assert ws.send_json.call_count >= 50

    @pytest.mark.asyncio
    async def test_progress_tracking_scenario(self):
        """Test progress tracking scenario."""
        from app.websocket import emit_progress

        run_id = "progress-run"

        # Connect subscribers
        websockets = []
        for i in range(5):
            ws = MagicMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)
            await manager.connect(ws, run_id)

        # Emit progress updates
        for i in range(0, 101, 5):
            await emit_progress(run_id, progress=i, message=f"Step {i}%")

        # Verify progress updates sent
        for ws in websockets:
            assert ws.send_json.call_count == 21  # 0, 5, 10, ... 100


# Cleanup after tests
@pytest.fixture(autouse=True)
def cleanup_manager():
    """Cleanup connection manager after each test."""
    yield
    # Clear all connections
    manager.active_connections.clear()
    manager.run_subscribers.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
