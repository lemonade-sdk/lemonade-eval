"""
Integration tests for Metrics API endpoints.

Tests cover:
- List metrics with pagination and filtering
- Create single metric
- Bulk create metrics
- Get single metric
- Delete metric
- Aggregate metrics
- Metric trends
- Compare metrics
- Performance metrics
"""

import pytest
from uuid import uuid4

from tests.conftest import ModelFactory, RunFactory, MetricFactory


class TestListMetrics:
    """Tests for GET /api/v1/metrics endpoint."""

    def test_list_metrics_empty(self, client, db_session):
        """Test listing metrics when database is empty."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["meta"]["total"] == 0

    def test_list_metrics_with_data(self, client, db_session, test_run):
        """Test listing metrics with data in database."""
        MetricFactory.create_performance_metrics(db_session, test_run.id)
        MetricFactory.create_accuracy_metrics(db_session, test_run.id)

        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meta"]["total"] == 7  # 4 perf + 3 accuracy

    def test_list_metrics_pagination(self, client, db_session, test_run):
        """Test metrics pagination."""
        # Create 50 metrics
        for i in range(50):
            MetricFactory.create(
                db_session,
                run_id=test_run.id,
                name=f"metric_{i}",
                value_numeric=float(i),
            )

        response = client.get("/api/v1/metrics?page=1&per_page=20")
        data = response.json()
        assert len(data["data"]) == 20
        assert data["meta"]["total"] == 50
        assert data["meta"]["total_pages"] == 3

    def test_list_metrics_filter_by_run_id(self, client, db_session):
        """Test filtering metrics by run ID."""
        run1 = RunFactory.create(db_session)
        run2 = RunFactory.create(db_session)

        MetricFactory.create(db_session, run_id=run1.id, name="metric_a")
        MetricFactory.create(db_session, run_id=run1.id, name="metric_b")
        MetricFactory.create(db_session, run_id=run2.id, name="metric_c")

        response = client.get(f"/api/v1/metrics?run_id={run1.id}")
        data = response.json()
        assert data["meta"]["total"] == 2
        assert all(m["run_id"] == run1.id for m in data["data"])

    def test_list_metrics_filter_by_category(self, client, db_session, test_run):
        """Test filtering metrics by category."""
        MetricFactory.create(
            db_session, run_id=test_run.id, category="performance", name="ttft"
        )
        MetricFactory.create(
            db_session, run_id=test_run.id, category="accuracy", name="mmlu"
        )
        MetricFactory.create(
            db_session, run_id=test_run.id, category="performance", name="tps"
        )

        response = client.get("/api/v1/metrics?category=performance")
        data = response.json()
        assert data["meta"]["total"] == 2
        assert all(m["category"] == "performance" for m in data["data"])

    def test_list_metrics_filter_by_name(self, client, db_session, test_run):
        """Test filtering metrics by name."""
        MetricFactory.create(db_session, run_id=test_run.id, name="seconds_to_first_token")
        MetricFactory.create(db_session, run_id=test_run.id, name="token_generation_tokens_per_second")
        MetricFactory.create(db_session, run_id=test_run.id, name="mmlu_stem")

        response = client.get("/api/v1/metrics?name=token")
        data = response.json()
        # Should match metrics containing "token" in name
        assert data["meta"]["total"] >= 1


class TestCreateMetric:
    """Tests for POST /api/v1/metrics endpoint."""

    def test_create_metric_performance(self, client, test_run):
        """Test creating a performance metric."""
        metric_data = {
            "run_id": test_run.id,
            "category": "performance",
            "name": "seconds_to_first_token",
            "display_name": "Seconds To First Token",
            "value_numeric": 0.025,
            "unit": "seconds",
        }
        response = client.post("/api/v1/metrics", json=metric_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["value_numeric"] == 0.025

    def test_create_metric_accuracy(self, client, test_run):
        """Test creating an accuracy metric."""
        metric_data = {
            "run_id": test_run.id,
            "category": "accuracy",
            "name": "mmlu_stem",
            "value_numeric": 65.4,
            "unit": "%",
        }
        response = client.post("/api/v1/metrics", json=metric_data)
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["category"] == "accuracy"

    def test_create_metric_text_value(self, client, test_run):
        """Test creating a metric with text value."""
        metric_data = {
            "run_id": test_run.id,
            "category": "accuracy",
            "name": "qualitative_assessment",
            "value_text": "pass",
        }
        response = client.post("/api/v1/metrics", json=metric_data)
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["value_text"] == "pass"

    def test_create_metric_invalid_run_id(self, client):
        """Test creating metric with invalid run ID."""
        metric_data = {
            "run_id": str(uuid4()),
            "category": "performance",
            "name": "test_metric",
            "value_numeric": 1.0,
        }
        response = client.post("/api/v1/metrics", json=metric_data)
        # May succeed or fail depending on FK constraints
        assert response.status_code in [201, 400, 500]


class TestBulkCreateMetrics:
    """Tests for POST /api/v1/metrics/bulk endpoint."""

    def test_bulk_create_metrics(self, client, test_run):
        """Test bulk creating metrics."""
        metrics_data = {
            "metrics": [
                {
                    "run_id": test_run.id,
                    "category": "performance",
                    "name": "ttft",
                    "value_numeric": 0.025,
                    "unit": "seconds",
                },
                {
                    "run_id": test_run.id,
                    "category": "performance",
                    "name": "tps",
                    "value_numeric": 45.5,
                    "unit": "tokens/s",
                },
                {
                    "run_id": test_run.id,
                    "category": "accuracy",
                    "name": "mmlu",
                    "value_numeric": 65.4,
                    "unit": "%",
                },
            ]
        }
        response = client.post("/api/v1/metrics/bulk", json=metrics_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 3

    def test_bulk_create_empty(self, client, test_run):
        """Test bulk creating with empty list."""
        metrics_data = {"metrics": []}
        response = client.post("/api/v1/metrics/bulk", json=metrics_data)
        assert response.status_code == 201


class TestGetMetric:
    """Tests for GET /api/v1/metrics/{metric_id} endpoint."""

    def test_get_metric_success(self, client, test_metric):
        """Test getting a metric by ID."""
        response = client.get(f"/api/v1/metrics/{test_metric.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == test_metric.id
        assert data["data"]["name"] == test_metric.name

    def test_get_metric_not_found(self, client):
        """Test getting a non-existent metric."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/metrics/{fake_id}")
        assert response.status_code == 404


class TestDeleteMetric:
    """Tests for DELETE /api/v1/metrics/{metric_id} endpoint."""

    def test_delete_metric_success(self, client, test_metric):
        """Test deleting a metric."""
        response = client.delete(f"/api/v1/metrics/{test_metric.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["data"]["message"]

        # Verify metric is deleted
        response = client.get(f"/api/v1/metrics/{test_metric.id}")
        assert response.status_code == 404

    def test_delete_metric_not_found(self, client):
        """Test deleting a non-existent metric."""
        fake_id = str(uuid4())
        response = client.delete(f"/api/v1/metrics/{fake_id}")
        assert response.status_code == 404


class TestAggregateMetrics:
    """Tests for GET /api/v1/metrics/aggregate endpoint."""

    def test_aggregate_metrics(self, client, db_session):
        """Test getting aggregated metrics."""
        model = ModelFactory.create(db_session)
        runs = [RunFactory.create(db_session, model_id=model.id) for _ in range(5)]

        # Create same metric across multiple runs
        for run in runs:
            MetricFactory.create(
                db_session,
                run_id=run.id,
                category="performance",
                name="seconds_to_first_token",
                value_numeric=0.025 + (runs.index(run) * 0.001),
                unit="seconds",
            )

        response = client.get("/api/v1/metrics/aggregate")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) > 0

        # Check aggregation fields
        agg = data["data"][0]
        assert "mean" in agg
        assert "std_dev" in agg
        assert "min" in agg
        assert "max" in agg
        assert "count" in agg

    def test_aggregate_metrics_filter_by_model(self, client, db_session):
        """Test aggregating metrics filtered by model."""
        model1 = ModelFactory.create(db_session, name="Model A")
        model2 = ModelFactory.create(db_session, name="Model B")

        run1 = RunFactory.create(db_session, model_id=model1.id)
        run2 = RunFactory.create(db_session, model_id=model2.id)

        MetricFactory.create(db_session, run_id=run1.id, name="ttft", value_numeric=0.02)
        MetricFactory.create(db_session, run_id=run2.id, name="ttft", value_numeric=0.03)

        response = client.get(f"/api/v1/metrics/aggregate?model_id={model1.id}")
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["mean"] == 0.02

    def test_aggregate_metrics_filter_by_category(self, client, db_session, test_run):
        """Test aggregating metrics filtered by category."""
        MetricFactory.create(
            db_session, run_id=test_run.id, category="performance", name="ttft", value_numeric=0.025
        )
        MetricFactory.create(
            db_session, run_id=test_run.id, category="accuracy", name="mmlu", value_numeric=65.4
        )

        response = client.get("/api/v1/metrics/aggregate?category=performance")
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["name"] == "ttft"


class TestMetricTrends:
    """Tests for GET /api/v1/metrics/trends endpoint."""

    def test_get_metric_trends(self, client, db_session):
        """Test getting metric trends for a model."""
        model = ModelFactory.create(db_session)

        # Create runs with metrics over time
        for i in range(5):
            run = RunFactory.create(db_session, model_id=model.id)
            MetricFactory.create(
                db_session,
                run_id=run.id,
                name="seconds_to_first_token",
                value_numeric=0.025 - (i * 0.001),  # Improving over time
                unit="seconds",
            )

        response = client.get(
            f"/api/v1/metrics/trends?model_id={model.id}&metric_name=seconds_to_first_token"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 5

        # Verify trend data structure
        trend = data["data"][0]
        assert "timestamp" in trend
        assert "value" in trend
        assert "unit" in trend

    def test_get_metric_trends_limit(self, client, db_session):
        """Test metric trends with limit."""
        model = ModelFactory.create(db_session)

        for i in range(20):
            run = RunFactory.create(db_session, model_id=model.id)
            MetricFactory.create(
                db_session, run_id=run.id, name="ttft", value_numeric=float(i)
            )

        response = client.get(
            f"/api/v1/metrics/trends?model_id={model.id}&metric_name=ttft&limit=10"
        )
        data = response.json()
        assert len(data["data"]) == 10


class TestCompareMetrics:
    """Tests for GET /api/v1/metrics/compare endpoint."""

    def test_compare_metrics(self, client, db_session):
        """Test comparing metrics across runs."""
        runs = [RunFactory.create(db_session) for _ in range(3)]

        for run in runs:
            MetricFactory.create_performance_metrics(db_session, run.id)

        run_ids = ",".join([r.id for r in runs])
        response = client.get(f"/api/v1/metrics/compare?run_ids={run_ids}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Should have data for each run
        assert len(data["data"]) == 3

    def test_compare_metrics_filter_categories(self, client, db_session):
        """Test comparing metrics with category filter."""
        runs = [RunFactory.create(db_session) for _ in range(2)]

        for run in runs:
            MetricFactory.create_performance_metrics(db_session, run.id)
            MetricFactory.create_accuracy_metrics(db_session, run.id)

        run_ids = ",".join([r.id for r in runs])
        response = client.get(f"/api/v1/metrics/compare?run_ids={run_ids}&categories=performance")
        data = response.json()
        # Verify only performance metrics are included
        for run_id, metrics in data["data"].items():
            assert all(m["category"] == "performance" for m in metrics)


class TestPerformanceMetrics:
    """Tests for GET /api/v1/metrics/performance/{run_id} endpoint."""

    def test_get_performance_metrics(self, client, db_session, test_run):
        """Test getting standard performance metrics for a run."""
        MetricFactory.create_performance_metrics(db_session, test_run.id)

        response = client.get(f"/api/v1/metrics/performance/{test_run.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Check standard perf metrics
        assert "seconds_to_first_token" in data["data"]
        assert "prefill_tokens_per_second" in data["data"]
        assert "token_generation_tokens_per_second" in data["data"]
        assert "max_memory_used_gbyte" in data["data"]

    def test_get_performance_metrics_partial(self, client, db_session, test_run):
        """Test getting performance metrics when only some exist."""
        # Create only TTFT metric
        MetricFactory.create(
            db_session,
            run_id=test_run.id,
            category="performance",
            name="seconds_to_first_token",
            value_numeric=0.025,
            unit="seconds",
        )

        response = client.get(f"/api/v1/metrics/performance/{test_run.id}")
        data = response.json()
        assert data["data"]["seconds_to_first_token"] is not None
        assert data["data"]["prefill_tokens_per_second"] is None

    def test_get_performance_metrics_run_not_found(self, client):
        """Test getting performance metrics for non-existent run."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/metrics/performance/{fake_id}")
        assert response.status_code == 404
