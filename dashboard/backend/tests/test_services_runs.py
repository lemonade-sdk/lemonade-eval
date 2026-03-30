"""
Unit tests for Run Service.

Tests cover:
- Run CRUD operations
- Pagination and filtering
- Status updates
- Metrics retrieval
- Statistics
- Edge cases
"""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta

from app.services.runs import RunService
from app.schemas import RunCreate, RunUpdate
from tests.conftest import ModelFactory, RunFactory, MetricFactory


class TestRunServiceGetRuns:
    """Tests for RunService.get_runs method."""

    def test_get_runs_empty(self, db_session):
        """Test getting runs when database is empty."""
        service = RunService(db_session)
        runs, meta = service.get_runs()

        assert runs == []
        assert meta.total == 0

    def test_get_runs_with_data(self, db_session):
        """Test getting runs with data."""
        model = ModelFactory.create(db_session)
        RunFactory.create(db_session, model_id=model.id)
        RunFactory.create(db_session, model_id=model.id)
        RunFactory.create(db_session, model_id=model.id)

        service = RunService(db_session)
        runs, meta = service.get_runs()

        assert len(runs) == 3
        assert meta.total == 3

    def test_get_runs_pagination(self, db_session):
        """Test pagination."""
        model = ModelFactory.create(db_session)
        for i in range(30):
            RunFactory.create(db_session, model_id=model.id)

        service = RunService(db_session)
        runs, meta = service.get_runs(page=1, per_page=10)

        assert len(runs) == 10
        assert meta.total == 30
        assert meta.total_pages == 3

    def test_get_runs_filter_by_model_id(self, db_session):
        """Test filtering by model ID."""
        model1 = ModelFactory.create(db_session, name="Model 1")
        model2 = ModelFactory.create(db_session, name="Model 2")

        RunFactory.create(db_session, model_id=model1.id)
        RunFactory.create(db_session, model_id=model1.id)
        RunFactory.create(db_session, model_id=model2.id)

        service = RunService(db_session)
        runs, meta = service.get_runs(model_id=model1.id)

        assert len(runs) == 2
        assert all(r.model_id == model1.id for r in runs)

    def test_get_runs_filter_by_status(self, db_session):
        """Test filtering by status."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, status="completed")
        RunFactory.create(db_session, model_id=model.id, status="completed")
        RunFactory.create(db_session, model_id=model.id, status="failed")

        service = RunService(db_session)
        runs, meta = service.get_runs(status="completed")

        assert len(runs) == 2
        assert all(r.status == "completed" for r in runs)

    def test_get_runs_filter_by_run_type(self, db_session):
        """Test filtering by run type."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, run_type="benchmark")
        RunFactory.create(db_session, model_id=model.id, run_type="accuracy-mmlu")

        service = RunService(db_session)
        runs, meta = service.get_runs(run_type="accuracy-mmlu")

        assert len(runs) == 1

    def test_get_runs_filter_by_device(self, db_session):
        """Test filtering by device."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, device="gpu")
        RunFactory.create(db_session, model_id=model.id, device="cpu")

        service = RunService(db_session)
        runs, meta = service.get_runs(device="gpu")

        assert len(runs) == 1

    def test_get_runs_filter_by_backend(self, db_session):
        """Test filtering by backend."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, backend="llamacpp")
        RunFactory.create(db_session, model_id=model.id, backend="ort")

        service = RunService(db_session)
        runs, meta = service.get_runs(backend="llamacpp")

        assert len(runs) == 1

    def test_get_runs_filter_combined(self, db_session):
        """Test combined filters."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, status="completed", device="gpu")
        RunFactory.create(db_session, model_id=model.id, status="completed", device="cpu")
        RunFactory.create(db_session, model_id=model.id, status="failed", device="gpu")

        service = RunService(db_session)
        runs, meta = service.get_runs(status="completed", device="gpu", model_id=model.id)

        assert len(runs) == 1


class TestRunServiceGetRun:
    """Tests for RunService.get_run method."""

    def test_get_run_success(self, db_session):
        """Test getting a run by ID."""
        run = RunFactory.create(db_session)

        service = RunService(db_session)
        result = service.get_run(run.id)

        assert result is not None
        assert result["id"] == run.id

    def test_get_run_with_metrics(self, db_session):
        """Test getting run with metrics."""
        run = RunFactory.create(db_session)
        MetricFactory.create_performance_metrics(db_session, run.id)

        service = RunService(db_session)
        result = service.get_run(run.id, include_metrics=True)

        assert result is not None
        assert "metrics" in result
        assert len(result["metrics"]) == 4

    def test_get_run_without_metrics(self, db_session):
        """Test getting run without metrics."""
        run = RunFactory.create(db_session)
        MetricFactory.create_performance_metrics(db_session, run.id)

        service = RunService(db_session)
        result = service.get_run(run.id, include_metrics=False)

        assert result is not None
        assert "metrics" not in result

    def test_get_run_not_found(self, db_session):
        """Test getting non-existent run."""
        service = RunService(db_session)
        result = service.get_run(str(uuid4()))

        assert result is None


class TestRunServiceCreateRun:
    """Tests for RunService.create_run method."""

    def test_create_run_minimal(self, db_session):
        """Test creating run with minimal data."""
        model = ModelFactory.create(db_session)

        run_data = RunCreate(
            model_id=model.id,
            build_name="test_build",
            run_type="benchmark",
        )

        service = RunService(db_session)
        result = service.create_run(run_data)

        assert result.id is not None
        assert result.status == "pending"  # default

    def test_create_run_full(self, db_session):
        """Test creating run with full data."""
        model = ModelFactory.create(db_session)

        run_data = RunCreate(
            model_id=model.id,
            build_name="test_build_full",
            run_type="accuracy-mmlu",
            status="running",
            device="gpu",
            backend="llamacpp",
            dtype="float16",
        )

        service = RunService(db_session)
        result = service.create_run(run_data)

        assert result.status == "running"
        assert result.device == "gpu"


class TestRunServiceUpdateRun:
    """Tests for RunService.update_run method."""

    def test_update_run_status(self, db_session):
        """Test updating run status."""
        run = RunFactory.create(db_session, status="pending")

        service = RunService(db_session)
        update_data = RunUpdate(status="completed")
        result = service.update_run(run.id, update_data)

        assert result is not None
        assert result.status == "completed"

    def test_update_run_status_message(self, db_session):
        """Test updating status message."""
        run = RunFactory.create(db_session)

        service = RunService(db_session)
        update_data = RunUpdate(status_message="New status message")
        result = service.update_run(run.id, update_data)

        assert result is not None
        assert result.status_message == "New status message"

    def test_update_run_not_found(self, db_session):
        """Test updating non-existent run."""
        service = RunService(db_session)
        update_data = RunUpdate(status="completed")
        result = service.update_run(str(uuid4()), update_data)

        assert result is None


class TestRunServiceUpdateStatus:
    """Tests for RunService.update_status method."""

    def test_update_status_to_running(self, db_session):
        """Test updating status to running."""
        run = RunFactory.create(db_session, status="pending")

        service = RunService(db_session)
        result = service.update_status(run.id, "running")

        assert result is not None
        assert result.status == "running"

    def test_update_status_to_completed(self, db_session):
        """Test updating status to completed."""
        run = RunFactory.create(db_session, status="running")

        service = RunService(db_session)
        result = service.update_status(run.id, "completed")

        assert result is not None
        assert result.status == "completed"

    def test_update_status_with_message(self, db_session):
        """Test updating status with message."""
        run = RunFactory.create(db_session)

        service = RunService(db_session)
        result = service.update_status(run.id, "failed", message="Out of memory")

        assert result is not None
        assert result.status == "failed"
        assert "Out of memory" in result.status_message

    def test_update_status_invalid(self, db_session):
        """Test updating to invalid status - service accepts any string."""
        run = RunFactory.create(db_session)

        service = RunService(db_session)
        # Service doesn't validate status - API layer does
        result = service.update_status(run.id, "invalid_status")

        assert result is not None
        assert result.status == "invalid_status"

    def test_update_status_not_found(self, db_session):
        """Test updating non-existent run status."""
        service = RunService(db_session)
        result = service.update_status(str(uuid4()), "completed")

        assert result is None


class TestRunServiceDeleteRun:
    """Tests for RunService.delete_run method."""

    def test_delete_run_success(self, db_session):
        """Test deleting a run."""
        run = RunFactory.create(db_session)

        service = RunService(db_session)
        result = service.delete_run(run.id)

        assert result is True

        # Verify deleted
        assert service.get_run(run.id) is None

    def test_delete_run_not_found(self, db_session):
        """Test deleting non-existent run."""
        service = RunService(db_session)
        result = service.delete_run(str(uuid4()))

        assert result is False


class TestRunServiceGetRunMetrics:
    """Tests for RunService.get_run_metrics method."""

    def test_get_run_metrics(self, db_session):
        """Test getting run metrics."""
        run = RunFactory.create(db_session)
        MetricFactory.create_performance_metrics(db_session, run.id)

        service = RunService(db_session)
        metrics = service.get_run_metrics(run.id)

        assert len(metrics) == 4

    def test_get_run_metrics_empty(self, db_session):
        """Test getting metrics when none exist."""
        run = RunFactory.create(db_session)

        service = RunService(db_session)
        metrics = service.get_run_metrics(run.id)

        assert metrics == []


class TestRunServiceGetRecentRuns:
    """Tests for RunService.get_recent_runs method."""

    def test_get_recent_runs(self, db_session):
        """Test getting recent runs."""
        model = ModelFactory.create(db_session)
        for i in range(15):
            RunFactory.create(db_session, model_id=model.id)

        service = RunService(db_session)
        runs = service.get_recent_runs(limit=10)

        assert len(runs) == 10

    def test_get_recent_runs_default_limit(self, db_session):
        """Test default limit."""
        model = ModelFactory.create(db_session)
        for i in range(20):
            RunFactory.create(db_session, model_id=model.id)

        service = RunService(db_session)
        runs = service.get_recent_runs()

        assert len(runs) == 10


class TestRunServiceGetRunStats:
    """Tests for RunService.get_run_stats method."""

    def test_get_run_stats(self, db_session):
        """Test getting run statistics."""
        model = ModelFactory.create(db_session)

        RunFactory.create(db_session, model_id=model.id, status="completed", run_type="benchmark")
        RunFactory.create(db_session, model_id=model.id, status="completed", run_type="benchmark")
        RunFactory.create(db_session, model_id=model.id, status="failed", run_type="accuracy-mmlu")

        service = RunService(db_session)
        stats = service.get_run_stats()

        assert stats["total_runs"] == 3
        assert stats["by_status"]["completed"] == 2
        assert stats["by_status"]["failed"] == 1
        assert stats["by_type"]["benchmark"] == 2

    def test_get_run_stats_empty(self, db_session):
        """Test stats when no runs exist."""
        service = RunService(db_session)
        stats = service.get_run_stats()

        assert stats["total_runs"] == 0
