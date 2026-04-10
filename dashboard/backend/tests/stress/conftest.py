"""
Pytest configuration and fixtures for stress testing.

Provides specialized fixtures for load and performance testing.
"""

import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator

# Set testing environment variable
os.environ["TESTING"] = "true"
os.environ["TEST_DATABASE_URL"] = "sqlite:///:memory:"


# ============================================================================
# STRESS TEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session")
def stress_test_config():
    """Configuration for stress tests."""
    return {
        "concurrent_users": 50,
        "requests_per_user": 100,
        "max_response_time_seconds": 2.0,
        "min_success_rate": 0.95,
    }


@pytest.fixture(scope="session")
def load_test_config():
    """Configuration for load tests."""
    return {
        "ramp_up_users": 10,
        "steady_state_users": 20,
        "ramp_down_users": 10,
        "test_duration_seconds": 60,
    }


# ============================================================================
# PERFORMANCE FIXTURES
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.response_times = []
            self.throughput = 0
            self.errors = 0

        def record_response_time(self, seconds: float):
            self.response_times.append(seconds)

        def get_avg_response_time(self) -> float:
            if not self.response_times:
                return 0.0
            return sum(self.response_times) / len(self.response_times)

        def get_p95_response_time(self) -> float:
            if not self.response_times:
                return 0.0
            sorted_times = sorted(self.response_times)
            p95_index = int(len(sorted_times) * 0.95)
            return sorted_times[min(p95_index, len(sorted_times) - 1)]

        def get_p99_response_time(self) -> float:
            if not self.response_times:
                return 0.0
            sorted_times = sorted(self.response_times)
            p99_index = int(len(sorted_times) * 0.99)
            return sorted_times[min(p99_index, len(sorted_times) - 1)]

    return PerformanceMonitor()


@pytest.fixture
def load_generator():
    """Generate load for stress testing."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    class LoadGenerator:
        def __init__(self, max_workers: int = 50):
            self.max_workers = max_workers
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        def run_load_test(
            self,
            func,
            num_requests: int,
            *args,
            **kwargs
        ) -> dict:
            """Run load test with specified number of requests."""
            start_time = time.time()
            results = []
            errors = []

            # Submit all requests
            futures = [
                self.executor.submit(func, *args, **kwargs)
                for _ in range(num_requests)
            ]

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))

            end_time = time.time()
            duration = end_time - start_time

            return {
                "total_requests": num_requests,
                "successful_requests": len(results),
                "failed_requests": len(errors),
                "duration_seconds": duration,
                "requests_per_second": num_requests / duration if duration > 0 else 0,
                "errors": errors[:10],  # First 10 errors
            }

        def shutdown(self):
            self.executor.shutdown(wait=True)

    generator = LoadGenerator()
    yield generator
    generator.shutdown()


# ============================================================================
# ASYNC STRESS FIXTURES
# ============================================================================

@pytest.fixture
def async_load_generator():
    """Generate async load for stress testing."""
    import time

    class AsyncLoadGenerator:
        async def run_load_test(
            self,
            coro_func,
            num_requests: int,
            *args,
            **kwargs
        ) -> dict:
            """Run async load test with specified number of requests."""
            start_time = time.time()
            results = []
            errors = []

            # Create tasks
            tasks = [
                coro_func(*args, **kwargs)
                for _ in range(num_requests)
            ]

            # Run all tasks concurrently
            coro_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Separate results and errors
            for result in coro_results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                else:
                    results.append(result)

            end_time = time.time()
            duration = end_time - start_time

            return {
                "total_requests": num_requests,
                "successful_requests": len(results),
                "failed_requests": len(errors),
                "duration_seconds": duration,
                "requests_per_second": num_requests / duration if duration > 0 else 0,
                "errors": errors[:10],
            }

    return AsyncLoadGenerator()


# ============================================================================
# METRICS COLLECTION
# ============================================================================

@pytest.fixture
def metrics_collector():
    """Collect and analyze test metrics."""
    class MetricsCollector:
        def __init__(self):
            self.metrics = {}

        def record(self, name: str, value: float):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

        def get_stats(self, name: str) -> dict:
            if name not in self.metrics or not self.metrics[name]:
                return {}

            values = self.metrics[name]
            sorted_values = sorted(values)

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "p50": sorted_values[len(values) // 2],
                "p95": sorted_values[int(len(values) * 0.95)] if len(values) > 1 else sorted_values[0],
                "p99": sorted_values[int(len(values) * 0.99)] if len(values) > 1 else sorted_values[0],
            }

        def summary(self) -> dict:
            return {
                name: self.get_stats(name)
                for name in self.metrics
            }

    return MetricsCollector()


# ============================================================================
# DATABASE LOAD FIXTURES
# ============================================================================

@pytest.fixture
def db_load_test_data(test_engine, db_session):
    """Create test data for database load testing."""
    from app.models import Model, Run, Metric
    from tests.conftest import ModelFactory, RunFactory, MetricFactory
    from uuid import uuid4

    # Create bulk test data
    models = []
    runs = []
    metrics = []

    for i in range(100):
        model = ModelFactory.create(
            db_session,
            name=f"Load Test Model {i}",
            checkpoint=f"test/model-{i}",
        )
        models.append(model)

        run = RunFactory.create(
            db_session,
            model_id=model.id,
            build_name=f"load-test-run-{i}",
        )
        runs.append(run)

        metric = MetricFactory.create(
            db_session,
            run_id=run.id,
            name="load_test_metric",
            value_numeric=i * 0.1,
        )
        metrics.append(metric)

    db_session.commit()

    yield {
        "models": models,
        "runs": runs,
        "metrics": metrics,
    }

    # Cleanup
    db_session.rollback()


# ============================================================================
# WEBSOCKET LOAD FIXTURES
# ============================================================================

@pytest.fixture
def websocket_load_test():
    """WebSocket load testing utilities."""
    class WebSocketLoadTest:
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url
            self.connections = []

        async def create_connections(
            self,
            num_connections: int,
            run_id: str = None
        ) -> list:
            """Create multiple WebSocket connections."""
            import websockets

            ws_url = f"ws://{self.base_url}/ws/v1/evaluations"
            if run_id:
                ws_url += f"?run_id={run_id}"

            connections = []
            for i in range(num_connections):
                ws = await websockets.connect(ws_url)
                connections.append(ws)

            self.connections = connections
            return connections

        async def close_all_connections(self):
            """Close all WebSocket connections."""
            import asyncio

            close_tasks = [ws.close() for ws in self.connections]
            await asyncio.gather(*close_tasks, return_exceptions=True)
            self.connections = []

    return WebSocketLoadTest()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_throughput(requests: int, duration_seconds: float) -> float:
    """Calculate requests per second."""
    return requests / duration_seconds if duration_seconds > 0 else 0


def calculate_success_rate(successful: int, total: int) -> float:
    """Calculate success rate."""
    return successful / total if total > 0 else 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
