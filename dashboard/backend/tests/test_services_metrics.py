"""
Unit tests for Metrics Service.

Tests cover:
- Metric CRUD operations
- Pagination and filtering
- Bulk creation
- Aggregation
- Trends
- Comparison
- Performance metrics
"""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta

from app.services.metrics import MetricService
from app.schemas import MetricCreate
from tests.conftest import ModelFactory, RunFactory, MetricFactory


class TestMetricServiceGetMetrics:
    """Tests for MetricService.get_metrics method."""

    def test_get_metrics_empty(self, db_session):
        """Test getting metrics when database is empty."""
        service = MetricService(db_session)
        metrics, meta = service.get_metrics()

        assert metrics == []
        assert meta.total == 0

    def test_get_metrics_with_data(self, db_session):
        """Test getting metrics with data."""
        run = RunFactory.create(db_session)
        MetricFactory.create_performance_metrics(db_session, run.id)

        service = MetricService(db_session)
        metrics, meta = service.get_metrics()

        assert len(metrics) == 4
        assert meta.total == 4

    def test_get_metrics_pagination(self, db_session):
        """Test pagination."""
        run = RunFactory.create(db_session)
        for i in range(50):
            MetricFactory.create(db_session, run_id=run.id, name=f"metric_{i}")

        service = MetricService(db_session)
        metrics, meta = service.get_metrics(page=1, per_page=20)

        assert len(metrics) == 20
        assert meta.total == 50
        assert meta.total_pages == 3

    def test_get_metrics_filter_by_run_id(self, db_session):
        """Test filtering by run ID."""
        run1 = RunFactory.create(db_session)
        run2 = RunFactory.create(db_session)

        MetricFactory.create(db_session, run_id=run1.id, name="metric_a")
        MetricFactory.create(db_session, run_id=run1.id, name="metric_b")
        MetricFactory.create(db_session, run_id=run2.id, name="metric_c")

        service = MetricService(db_session)
        metrics, meta = service.get_metrics(run_id=run1.id)

        assert len(metrics) == 2
        assert all(m.run_id == run1.id for m in metrics)

    def test_get_metrics_filter_by_category(self, db_session):
        """Test filtering by category."""
        run = RunFactory.create(db_session)

        MetricFactory.create(db_session, run_id=run.id, category="performance", name="ttft")
        MetricFactory.create(db_session, run_id=run.id, category="accuracy", name="mmlu")
        MetricFactory.create(db_session, run_id=run.id, category="performance", name="tps")

        service = MetricService(db_session)
        metrics, meta = service.get_metrics(category="performance")

        assert len(metrics) == 2
        assert all(m.category == "performance" for m in metrics)

    def test_get_metrics_filter_by_name(self, db_session):
        """Test filtering by name."""
        run = RunFactory.create(db_session)

        MetricFactory.create(db_session, run_id=run.id, name="seconds_to_first_token")
        MetricFactory.create(db_session, run_id=run.id, name="token_generation_tokens_per_second")
        MetricFactory.create(db_session, run_id=run.id, name="mmlu_stem")

        service = MetricService(db_session)
        metrics, meta = service.get_metrics(name="token")

        # Should match metrics containing "token"
        assert len(metrics) >= 1


class TestMetricServiceGetMetric:
    """Tests for MetricService.get_metric method."""

    def test_get_metric_success(self, db_session):
        """Test getting a metric by ID."""
        run = RunFactory.create(db_session)
        metric = MetricFactory.create(db_session, run_id=run.id, name="test_metric")

        service = MetricService(db_session)
        result = service.get_metric(metric.id)

        assert result is not None
        assert result.id == metric.id
        assert result.name == "test_metric"

    def test_get_metric_not_found(self, db_session):
        """Test getting non-existent metric."""
        service = MetricService(db_session)
        result = service.get_metric(str(uuid4()))

        assert result is None


class TestMetricServiceCreateMetric:
    """Tests for MetricService.create_metric method."""

    def test_create_metric_performance(self, db_session):
        """Test creating performance metric."""
        run = RunFactory.create(db_session)

        metric_data = MetricCreate(
            run_id=run.id,
            category="performance",
            name="seconds_to_first_token",
            value_numeric=0.025,
            unit="seconds",
        )

        service = MetricService(db_session)
        result = service.create_metric(metric_data)

        assert result is not None
        assert result.value_numeric == 0.025

    def test_create_metric_accuracy(self, db_session):
        """Test creating accuracy metric."""
        run = RunFactory.create(db_session)

        metric_data = MetricCreate(
            run_id=run.id,
            category="accuracy",
            name="mmlu_stem",
            value_numeric=65.4,
            unit="%",
        )

        service = MetricService(db_session)
        result = service.create_metric(metric_data)

        assert result is not None
        assert result.category == "accuracy"

    def test_create_metric_text_value(self, db_session):
        """Test creating metric with text value."""
        run = RunFactory.create(db_session)

        metric_data = MetricCreate(
            run_id=run.id,
            category="accuracy",
            name="qualitative",
            value_text="pass",
        )

        service = MetricService(db_session)
        result = service.create_metric(metric_data)

        assert result is not None
        assert result.value_text == "pass"


class TestMetricServiceCreateMetricsBulk:
    """Tests for MetricService.create_metrics_bulk method."""

    def test_bulk_create_metrics(self, db_session):
        """Test bulk creating metrics."""
        run = RunFactory.create(db_session)

        metrics_data = [
            MetricCreate(
                run_id=run.id,
                category="performance",
                name="ttft",
                value_numeric=0.025,
                unit="seconds",
            ),
            MetricCreate(
                run_id=run.id,
                category="performance",
                name="tps",
                value_numeric=45.5,
                unit="tokens/s",
            ),
            MetricCreate(
                run_id=run.id,
                category="accuracy",
                name="mmlu",
                value_numeric=65.4,
                unit="%",
            ),
        ]

        service = MetricService(db_session)
        results = service.create_metrics_bulk(metrics_data)

        assert len(results) == 3

    def test_bulk_create_empty(self, db_session):
        """Test bulk creating with empty list."""
        service = MetricService(db_session)
        results = service.create_metrics_bulk([])

        assert results == []


class TestMetricServiceDeleteMetric:
    """Tests for MetricService.delete_metric method."""

    def test_delete_metric_success(self, db_session):
        """Test deleting a metric."""
        run = RunFactory.create(db_session)
        metric = MetricFactory.create(db_session, run_id=run.id)

        service = MetricService(db_session)
        result = service.delete_metric(metric.id)

        assert result is True

        # Verify deleted
        assert service.get_metric(metric.id) is None

    def test_delete_metric_not_found(self, db_session):
        """Test deleting non-existent metric."""
        service = MetricService(db_session)
        result = service.delete_metric(str(uuid4()))

        assert result is False


class TestMetricServiceGetAggregateMetrics:
    """Tests for MetricService.get_aggregate_metrics method."""

    def test_aggregate_metrics(self, db_session):
        """Test getting aggregated metrics."""
        model = ModelFactory.create(db_session)
        runs = [RunFactory.create(db_session, model_id=model.id) for _ in range(5)]

        for run in runs:
            MetricFactory.create(
                db_session,
                run_id=run.id,
                name="seconds_to_first_token",
                value_numeric=0.025 + (runs.index(run) * 0.001),
                unit="seconds",
            )

        service = MetricService(db_session)
        aggregates = service.get_aggregate_metrics()

        assert len(aggregates) > 0
        agg = aggregates[0]
        assert "mean" in agg
        assert "std_dev" in agg
        assert "min" in agg
        assert "max" in agg
        assert "count" in agg

    def test_aggregate_filter_by_model(self, db_session):
        """Test aggregating filtered by model."""
        model1 = ModelFactory.create(db_session, name="Model A")
        model2 = ModelFactory.create(db_session, name="Model B")

        run1 = RunFactory.create(db_session, model_id=model1.id)
        run2 = RunFactory.create(db_session, model_id=model2.id)

        MetricFactory.create(db_session, run_id=run1.id, name="ttft", value_numeric=0.02)
        MetricFactory.create(db_session, run_id=run2.id, name="ttft", value_numeric=0.03)

        service = MetricService(db_session)
        aggregates = service.get_aggregate_metrics(model_id=model1.id)

        assert len(aggregates) == 1
        assert aggregates[0]["mean"] == 0.02

    def test_aggregate_filter_by_category(self, db_session):
        """Test aggregating filtered by category."""
        run = RunFactory.create(db_session)

        MetricFactory.create(db_session, run_id=run.id, category="performance", name="ttft", value_numeric=0.025)
        MetricFactory.create(db_session, run_id=run.id, category="accuracy", name="mmlu", value_numeric=65.4)

        service = MetricService(db_session)
        aggregates = service.get_aggregate_metrics(category="performance")

        assert len(aggregates) == 1
        assert aggregates[0]["name"] == "ttft"


class TestMetricServiceGetMetricTrends:
    """Tests for MetricService.get_metric_trends method."""

    def test_get_metric_trends(self, db_session):
        """Test getting metric trends."""
        model = ModelFactory.create(db_session)

        for i in range(5):
            run = RunFactory.create(db_session, model_id=model.id)
            MetricFactory.create(
                db_session,
                run_id=run.id,
                name="seconds_to_first_token",
                value_numeric=0.025 - (i * 0.001),
                unit="seconds",
            )

        service = MetricService(db_session)
        trends = service.get_metric_trends(
            model_id=model.id,
            metric_name="seconds_to_first_token",
        )

        assert len(trends) == 5
        trend = trends[0]
        assert "timestamp" in trend
        assert "value" in trend
        assert "unit" in trend

    def test_get_metric_trends_limit(self, db_session):
        """Test trends with limit."""
        model = ModelFactory.create(db_session)

        for i in range(20):
            run = RunFactory.create(db_session, model_id=model.id)
            MetricFactory.create(db_session, run_id=run.id, name="ttft", value_numeric=float(i))

        service = MetricService(db_session)
        trends = service.get_metric_trends(
            model_id=model.id,
            metric_name="ttft",
            limit=10,
        )

        assert len(trends) == 10


class TestMetricServiceCompareMetrics:
    """Tests for MetricService.compare_metrics method."""

    def test_compare_metrics(self, db_session):
        """Test comparing metrics across runs."""
        runs = [RunFactory.create(db_session) for _ in range(3)]

        for run in runs:
            MetricFactory.create_performance_metrics(db_session, run.id)

        service = MetricService(db_session)
        comparison = service.compare_metrics([r.id for r in runs])

        assert len(comparison) == 3

    def test_compare_metrics_filter_categories(self, db_session):
        """Test comparing with category filter."""
        runs = [RunFactory.create(db_session) for _ in range(2)]

        for run in runs:
            MetricFactory.create_performance_metrics(db_session, run.id)
            MetricFactory.create_accuracy_metrics(db_session, run.id)

        service = MetricService(db_session)
        comparison = service.compare_metrics([r.id for r in runs], categories=["performance"])

        # Verify only performance metrics
        for run_id, metrics in comparison.items():
            assert all(m["category"] == "performance" for m in metrics)


class TestMetricServiceGetPerformanceMetrics:
    """Tests for MetricService.get_performance_metrics method."""

    def test_get_performance_metrics(self, db_session):
        """Test getting standard performance metrics."""
        run = RunFactory.create(db_session)
        MetricFactory.create_performance_metrics(db_session, run.id)

        service = MetricService(db_session)
        perf_metrics = service.get_performance_metrics(run.id)

        assert "seconds_to_first_token" in perf_metrics
        assert "prefill_tokens_per_second" in perf_metrics
        assert "token_generation_tokens_per_second" in perf_metrics
        assert "max_memory_used_gbyte" in perf_metrics

    def test_get_performance_metrics_partial(self, db_session):
        """Test getting partial performance metrics."""
        run = RunFactory.create(db_session)

        # Create only TTFT
        MetricFactory.create(
            db_session,
            run_id=run.id,
            category="performance",
            name="seconds_to_first_token",
            value_numeric=0.025,
        )

        service = MetricService(db_session)
        perf_metrics = service.get_performance_metrics(run.id)

        assert perf_metrics["seconds_to_first_token"] is not None
        assert perf_metrics["prefill_tokens_per_second"] is None
