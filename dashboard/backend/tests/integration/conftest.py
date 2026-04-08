"""
Pytest configuration and fixtures for integration testing.

Provides fixtures for end-to-end integration tests.
"""

import os
import pytest
from typing import Generator

# Set testing environment variable
os.environ["TESTING"] = "true"
os.environ["TEST_DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "base_url": "http://localhost:8000",
        "api_prefix": "/api/v1",
        "ws_prefix": "/ws/v1",
        "test_timeout_seconds": 30,
    }


@pytest.fixture
def api_client(client):
    """Extended API client for integration testing."""
    class IntegrationClient:
        def __init__(self, test_client):
            self.client = test_client
            self.base_url = "/api/v1"
            self.headers = {}

        def set_auth_token(self, token: str):
            """Set authentication token for requests."""
            self.headers["Authorization"] = f"Bearer {token}"

        def create_model(self, model_data: dict) -> dict:
            """Create a model and return response."""
            response = self.client.post(
                f"{self.base_url}/models",
                json=model_data,
                headers=self.headers,
            )
            return response.json()

        def create_run(self, run_data: dict) -> dict:
            """Create a run and return response."""
            response = self.client.post(
                f"{self.base_url}/runs",
                json=run_data,
                headers=self.headers,
            )
            return response.json()

        def submit_metrics(self, run_id: str, metrics: list) -> dict:
            """Submit metrics for a run."""
            response = self.client.post(
                f"{self.base_url}/metrics/bulk",
                json={"run_id": run_id, "metrics": metrics},
                headers=self.headers,
            )
            return response.json()

        def get_run(self, run_id: str) -> dict:
            """Get run details."""
            response = self.client.get(
                f"{self.base_url}/runs/{run_id}",
                headers=self.headers,
            )
            return response.json()

        def list_models(self) -> list:
            """List all models."""
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self.headers,
            )
            return response.json()

        def list_runs(self, model_id: str = None) -> list:
            """List runs, optionally filtered by model."""
            params = {}
            if model_id:
                params["model_id"] = model_id
            response = self.client.get(
                f"{self.base_url}/runs",
                params=params,
                headers=self.headers,
            )
            return response.json()

    return IntegrationClient(client)


@pytest.fixture
def integration_test_data(db_session):
    """Create standard test data for integration tests."""
    from tests.conftest import UserFactory, ModelFactory, RunFactory, MetricFactory

    # Create test user
    user = UserFactory.create(db_session, role="editor")

    # Create test models
    models = []
    for i in range(3):
        model = ModelFactory.create(
            db_session,
            name=f"Integration Test Model {i}",
            checkpoint=f"integration/model-{i}",
        )
        models.append(model)

    # Create test runs
    runs = []
    for model in models:
        run = RunFactory.create(
            db_session,
            model_id=model.id,
            build_name=f"integration-run-{model.id[:8]}",
            status="completed",
        )
        runs.append(run)

    # Create metrics for each run
    for run in runs:
        MetricFactory.create_performance_metrics(db_session, run.id)

    db_session.commit()

    return {
        "user": user,
        "models": models,
        "runs": runs,
    }


@pytest.fixture
def cli_integration_data():
    """Test data for CLI integration tests."""
    return {
        "evaluation_data": {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "run_type": "benchmark",
            "build_name": "integration-cli-test",
            "metrics": [
                {
                    "name": "seconds_to_first_token",
                    "value": 0.025,
                    "unit": "seconds",
                    "category": "performance",
                },
                {
                    "name": "token_generation_tokens_per_second",
                    "value": 45.5,
                    "unit": "tokens/s",
                    "category": "performance",
                },
            ],
            "device": "gpu",
            "backend": "ort",
            "dtype": "float16",
            "status": "completed",
        },
        "bulk_data": {
            "evaluations": [
                {
                    "model_checkpoint": "model/checkpoint-1",
                    "run_type": "benchmark",
                    "build_name": "bulk-integration-1",
                    "metrics": [{"name": "ttft", "value": 0.025}],
                },
                {
                    "model_checkpoint": "model/checkpoint-2",
                    "run_type": "accuracy-mmlu",
                    "build_name": "bulk-integration-2",
                    "metrics": [{"name": "mmlu_score", "value": 68.5}],
                },
            ],
            "skip_duplicates": True,
        },
    }


@pytest.fixture
def websocket_integration_config():
    """Configuration for WebSocket integration tests."""
    return {
        "ws_endpoint": "/ws/v1/evaluations",
        "progress_endpoint": "/ws/v1/evaluation-progress",
        "ping_interval_seconds": 30,
        "message_timeout_seconds": 5,
    }


@pytest.fixture
def cache_integration_test():
    """Cache integration test utilities."""
    from app.cache.cache_manager import CacheManager

    cache = CacheManager(redis_url="redis://localhost:6379/0")

    # Try to connect
    connected = cache.connect()

    yield {
        "cache": cache,
        "connected": connected,
    }

    # Cleanup
    if connected:
        cache.disconnect()


@pytest.fixture
def rate_limiter_integration_test():
    """Rate limiter integration test utilities."""
    from app.middleware.rate_limiter import RateLimiter

    limiter = RateLimiter(
        redis_url="redis://localhost:6379/0",
        default_rate=100,
        default_burst=200,
    )

    # Try to connect
    limiter.connect()

    yield {
        "limiter": limiter,
        "connected": limiter.redis is not None,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
