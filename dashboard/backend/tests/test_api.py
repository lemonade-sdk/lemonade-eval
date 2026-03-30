"""
Basic tests for the Dashboard Backend API.

Note: These tests verify basic API functionality.
Full integration tests require PostgreSQL.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ModelCreate, RunCreate, MetricCreate


# Create test client
@pytest.fixture(scope="function")
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


class TestHealth:
    """Tests for health check endpoints."""

    def test_root(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_api_info(self, client):
        """Test the API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "endpoints" in data


class TestModelsUnit:
    """Unit tests for model schemas."""

    def test_model_create_schema(self):
        """Test ModelCreate schema validation."""
        model_data = ModelCreate(
            name="Test Model",
            checkpoint="test/checkpoint-1b",
            model_type="llm",
            family="Test",
            parameters=1000000000,
        )
        assert model_data.name == "Test Model"
        assert model_data.checkpoint == "test/checkpoint-1b"
        assert model_data.model_type == "llm"

    def test_model_create_schema_minimal(self):
        """Test ModelCreate schema with minimal data."""
        model_data = ModelCreate(
            name="Minimal Model",
            checkpoint="minimal/model",
        )
        assert model_data.name == "Minimal Model"
        assert model_data.checkpoint == "minimal/model"
        assert model_data.model_type == "llm"  # default

    def test_model_create_schema_with_metadata(self):
        """Test ModelCreate schema with metadata."""
        model_data = ModelCreate(
            name="Model with Metadata",
            checkpoint="test/metadata-model",
            model_metadata={"key": "value"},
        )
        assert model_data.model_metadata == {"key": "value"}


class TestRunsUnit:
    """Unit tests for run schemas."""

    def test_run_create_schema(self):
        """Test RunCreate schema validation."""
        run_data = RunCreate(
            model_id="test-model-id",
            build_name="test_build_2026",
            run_type="benchmark",
            status="completed",
        )
        assert run_data.model_id == "test-model-id"
        assert run_data.build_name == "test_build_2026"
        assert run_data.run_type == "benchmark"

    def test_run_create_schema_defaults(self):
        """Test RunCreate schema with default values."""
        run_data = RunCreate(
            model_id="test-model-id",
            build_name="test_build",
            run_type="benchmark",
        )
        assert run_data.status == "pending"  # default
        assert run_data.config == {}  # default
        assert run_data.system_info == {}  # default


class TestMetricsUnit:
    """Unit tests for metric schemas."""

    def test_metric_create_schema(self):
        """Test MetricCreate schema validation."""
        metric_data = MetricCreate(
            run_id="test-run-id",
            category="performance",
            name="seconds_to_first_token",
            value_numeric=0.025,
            unit="seconds",
        )
        assert metric_data.run_id == "test-run-id"
        assert metric_data.category == "performance"
        assert metric_data.value_numeric == 0.025

    def test_metric_create_schema_text_value(self):
        """Test MetricCreate schema with text value."""
        metric_data = MetricCreate(
            run_id="test-run-id",
            category="accuracy",
            name="test_metric",
            value_text="pass",
        )
        assert metric_data.value_text == "pass"


class TestAPIResponse:
    """Tests for API response schema."""

    def test_api_response_success(self):
        """Test successful API response."""
        from app.schemas import APIResponse

        response = APIResponse(
            success=True,
            data={"key": "value"},
        )
        assert response.success is True
        assert response.data == {"key": "value"}
        assert response.errors == []

    def test_api_response_with_meta(self):
        """Test API response with pagination metadata."""
        from app.schemas import APIResponse, PaginationMeta

        meta = PaginationMeta(
            page=1,
            per_page=20,
            total=100,
            total_pages=5,
        )
        response = APIResponse(
            success=True,
            data=[],
            meta=meta,
        )
        assert response.meta.page == 1
        assert response.meta.total == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
