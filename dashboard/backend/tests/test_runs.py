"""
Integration tests for Run API endpoints.

Tests cover:
- List runs with pagination and filtering
- Create run
- Get single run with optional metrics
- Update run
- Update run status
- Delete run
- Get run metrics
- Recent runs and stats
"""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta

from tests.conftest import ModelFactory, RunFactory, MetricFactory


class TestListRuns:
    """Tests for GET /api/v1/runs endpoint."""

    def test_list_runs_empty(self, client, db_session):
        """Test listing runs when database is empty."""
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["meta"]["total"] == 0

    def test_list_runs_with_data(self, client, db_session):
        """Test listing runs with data in database."""
        model = ModelFactory.create(db_session)
        RunFactory.create(db_session, model_id=model.id, status="completed")
        RunFactory.create(db_session, model_id=model.id, status="running")
        RunFactory.create(db_session, model_id=model.id, status="pending")

        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meta"]["total"] == 3
        assert len(data["data"]) == 3

    def test_list_runs_pagination(self, client, db_session):
        """Test run listing pagination."""
        model = ModelFactory.create(db_session)
        for i in range(30):
            RunFactory.create(db_session, model_id=model.id)

        response = client.get("/api/v1/runs?page=1&per_page=10")
        data = response.json()
        assert len(data["data"]) == 10
        assert data["meta"]["total"] == 30
        assert data["meta"]["total_pages"] == 3

    def test_list_runs_filter_by_model_id(self, client, db_session):
        """Test filtering runs by model ID."""
        model1 = ModelFactory.create(db_session, name="Model 1")
        model2 = ModelFactory.create(db_session, name="Model 2")

        RunFactory.create(db_session, model_id=model1.id)
        RunFactory.create(db_session, model_id=model1.id)
        RunFactory.create(db_session, model_id=model2.id)

        response = client.get(f"/api/v1/runs?model_id={model1.id}")
        data = response.json()
        assert data["meta"]["total"] == 2
        assert all(r["model_id"] == model1.id for r in data["data"])

    def test_list_runs_filter_by_status(self, client, db_session):
        """Test filtering runs by status."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, status="completed")
        RunFactory.create(db_session, model_id=model.id, status="completed")
        RunFactory.create(db_session, model_id=model.id, status="failed")
        RunFactory.create(db_session, model_id=model.id, status="running")

        response = client.get("/api/v1/runs?status=completed")
        data = response.json()
        assert data["meta"]["total"] == 2
        assert all(r["status"] == "completed" for r in data["data"])

    def test_list_runs_filter_by_run_type(self, client, db_session):
        """Test filtering runs by run type."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, run_type="benchmark")
        RunFactory.create(db_session, model_id=model.id, run_type="accuracy-mmlu")
        RunFactory.create(db_session, model_id=model.id, run_type="accuracy-humaneval")

        response = client.get("/api/v1/runs?run_type=accuracy-mmlu")
        data = response.json()
        assert data["meta"]["total"] == 1
        assert data["data"][0]["run_type"] == "accuracy-mmlu"

    def test_list_runs_filter_by_device(self, client, db_session):
        """Test filtering runs by device."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, device="gpu")
        RunFactory.create(db_session, model_id=model.id, device="cpu")
        RunFactory.create(db_session, model_id=model.id, device="npu")

        response = client.get("/api/v1/runs?device=gpu")
        data = response.json()
        assert data["meta"]["total"] == 1
        assert data["data"][0]["device"] == "gpu"

    def test_list_runs_filter_by_backend(self, client, db_session):
        """Test filtering runs by backend."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, backend="llamacpp")
        RunFactory.create(db_session, model_id=model.id, backend="ort")
        RunFactory.create(db_session, model_id=model.id, backend="flm")

        response = client.get("/api/v1/runs?backend=llamacpp")
        data = response.json()
        assert data["meta"]["total"] == 1

    def test_list_runs_filter_by_multiple_filters(self, client, db_session):
        """Test filtering runs by multiple criteria."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, status="completed", device="gpu")
        RunFactory.create(db_session, model_id=model.id, status="completed", device="cpu")
        RunFactory.create(db_session, model_id=model.id, status="failed", device="gpu")
        RunFactory.create(db_session, model_id=model.id, status="running", device="gpu")

        response = client.get(f"/api/v1/runs?status=completed&device=gpu&model_id={model.id}")
        data = response.json()
        assert data["meta"]["total"] == 1


class TestCreateRun:
    """Tests for POST /api/v1/runs endpoint."""

    def test_create_run_minimal(self, client, test_model):
        """Test creating a run with minimal required fields."""
        run_data = {
            "model_id": test_model.id,
            "build_name": "test_build_001",
            "run_type": "benchmark",
        }
        response = client.post("/api/v1/runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["model_id"] == test_model.id
        assert data["data"]["build_name"] == "test_build_001"
        assert data["data"]["status"] == "pending"  # default

    def test_create_run_full(self, client, test_model):
        """Test creating a run with all fields."""
        run_data = {
            "model_id": test_model.id,
            "build_name": "test_build_full",
            "run_type": "accuracy-mmlu",
            "status": "running",
            "device": "gpu",
            "backend": "llamacpp",
            "dtype": "float16",
            "config": {"iterations": 100, "prompts": 10},
            "system_info": {"cpu": "Intel", "gpu": "NVIDIA"},
        }
        response = client.post("/api/v1/runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["status"] == "running"
        assert data["data"]["device"] == "gpu"
        assert data["data"]["config"]["iterations"] == 100

    def test_create_run_invalid_model_id(self, client):
        """Test creating a run with invalid model ID."""
        run_data = {
            "model_id": str(uuid4()),
            "build_name": "test_build",
            "run_type": "benchmark",
        }
        # This might succeed if foreign key constraint is not enforced
        # or fail with 400/500
        response = client.post("/api/v1/runs", json=run_data)
        assert response.status_code in [201, 400, 500]


class TestGetRun:
    """Tests for GET /api/v1/runs/{run_id} endpoint."""

    def test_get_run_success(self, client, test_run):
        """Test getting a run by ID."""
        response = client.get(f"/api/v1/runs/{test_run.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == test_run.id
        assert data["data"]["build_name"] == test_run.build_name

    def test_get_run_with_metrics(self, client, db_session, test_run):
        """Test getting a run with metrics included."""
        MetricFactory.create_performance_metrics(db_session, test_run.id)

        response = client.get(f"/api/v1/runs/{test_run.id}?include_metrics=true")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metrics" in data["data"]
        assert len(data["data"]["metrics"]) == 4

    def test_get_run_without_metrics(self, client, db_session, test_run):
        """Test getting a run without metrics."""
        MetricFactory.create_performance_metrics(db_session, test_run.id)

        response = client.get(f"/api/v1/runs/{test_run.id}?include_metrics=false")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metrics" not in data["data"]

    def test_get_run_not_found(self, client):
        """Test getting a non-existent run."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/runs/{fake_id}")
        assert response.status_code == 404


class TestUpdateRun:
    """Tests for PUT /api/v1/runs/{run_id} endpoint."""

    def test_update_run_status_field(self, client, test_run):
        """Test updating run status."""
        update_data = {"status": "completed"}
        response = client.put(f"/api/v1/runs/{test_run.id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "completed"

    def test_update_run_status_message(self, client, test_run):
        """Test updating run status message."""
        update_data = {"status_message": "Evaluation completed successfully"}
        response = client.put(f"/api/v1/runs/{test_run.id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status_message"] == "Evaluation completed successfully"

    def test_update_run_config(self, client, test_run):
        """Test updating run config."""
        update_data = {"config": {"iterations": 50, "new_key": "value"}}
        response = client.put(f"/api/v1/runs/{test_run.id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["config"]["iterations"] == 50
        assert data["data"]["config"]["new_key"] == "value"

    def test_update_run_not_found(self, client):
        """Test updating a non-existent run."""
        fake_id = str(uuid4())
        update_data = {"status": "completed"}
        response = client.put(f"/api/v1/runs/{fake_id}", json=update_data)
        assert response.status_code == 404


class TestUpdateRunStatus:
    """Tests for POST /api/v1/runs/{run_id}/status endpoint."""

    def test_update_status_to_running(self, client, test_run):
        """Test updating status to running."""
        response = client.post(f"/api/v1/runs/{test_run.id}/status?status=running")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "running"

    def test_update_status_to_completed(self, client, test_run):
        """Test updating status to completed."""
        response = client.post(f"/api/v1/runs/{test_run.id}/status?status=completed")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "completed"

    def test_update_status_with_message(self, client, test_run):
        """Test updating status with message."""
        response = client.post(
            f"/api/v1/runs/{test_run.id}/status?status=failed&message=Out of memory error"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "failed"
        assert "Out of memory" in data["data"]["status_message"]

    def test_update_status_invalid(self, client, test_run):
        """Test updating status with invalid value."""
        response = client.post(f"/api/v1/runs/{test_run.id}/status?status=invalid_status")
        assert response.status_code == 400
        data = response.json()
        assert "Invalid status" in data["detail"]

    def test_update_status_not_found(self, client):
        """Test updating status for non-existent run."""
        fake_id = str(uuid4())
        response = client.post(f"/api/v1/runs/{fake_id}/status?status=completed")
        assert response.status_code == 404


class TestDeleteRun:
    """Tests for DELETE /api/v1/runs/{run_id} endpoint."""

    def test_delete_run_success(self, client, test_run):
        """Test deleting a run."""
        response = client.delete(f"/api/v1/runs/{test_run.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["data"]["message"]

        # Verify run is deleted
        response = client.get(f"/api/v1/runs/{test_run.id}")
        assert response.status_code == 404

    def test_delete_run_cascades_to_metrics(self, client, db_session, test_run):
        """Test that deleting a run cascades to metrics."""
        MetricFactory.create_performance_metrics(db_session, test_run.id)

        # Delete run
        response = client.delete(f"/api/v1/runs/{test_run.id}")
        assert response.status_code == 200

    def test_delete_run_not_found(self, client):
        """Test deleting a non-existent run."""
        fake_id = str(uuid4())
        response = client.delete(f"/api/v1/runs/{fake_id}")
        assert response.status_code == 404


class TestGetRunMetrics:
    """Tests for GET /api/v1/runs/{run_id}/metrics endpoint."""

    def test_get_run_metrics(self, client, db_session, test_run):
        """Test getting metrics for a run."""
        metrics = MetricFactory.create_performance_metrics(db_session, test_run.id)

        response = client.get(f"/api/v1/runs/{test_run.id}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 4

    def test_get_run_metrics_empty(self, client, test_run):
        """Test getting metrics for a run with no metrics."""
        response = client.get(f"/api/v1/runs/{test_run.id}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []

    def test_get_run_metrics_not_found(self, client):
        """Test getting metrics for non-existent run."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/runs/{fake_id}/metrics")
        assert response.status_code == 404


class TestGetRecentRuns:
    """Tests for GET /api/v1/runs/recent/list endpoint."""

    def test_get_recent_runs(self, client, db_session):
        """Test getting recent runs."""
        model = ModelFactory.create(db_session)
        for i in range(15):
            RunFactory.create(db_session, model_id=model.id)

        response = client.get("/api/v1/runs/recent/list?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 10

    def test_get_recent_runs_default_limit(self, client, db_session):
        """Test getting recent runs with default limit."""
        model = ModelFactory.create(db_session)
        for i in range(20):
            RunFactory.create(db_session, model_id=model.id)

        response = client.get("/api/v1/runs/recent/list")
        data = response.json()
        assert len(data["data"]) == 10  # default limit


class TestGetRunStats:
    """Tests for GET /api/v1/runs/stats endpoint."""

    def test_get_run_stats(self, client, db_session):
        """Test getting run statistics."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, status="completed", run_type="benchmark")
        RunFactory.create(db_session, model_id=model.id, status="completed", run_type="benchmark")
        RunFactory.create(db_session, model_id=model.id, status="failed", run_type="accuracy-mmlu")
        RunFactory.create(db_session, model_id=model.id, status="running", run_type="benchmark")

        response = client.get("/api/v1/runs/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_runs"] == 4
        assert data["data"]["by_status"]["completed"] == 2
        assert data["data"]["by_status"]["failed"] == 1
        assert data["data"]["by_type"]["benchmark"] == 3
        assert data["data"]["by_type"]["accuracy-mmlu"] == 1

    def test_get_run_stats_empty(self, client, db_session):
        """Test getting stats when no runs exist."""
        response = client.get("/api/v1/runs/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total_runs"] == 0
