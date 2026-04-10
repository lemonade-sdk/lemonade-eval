"""
Metric service for business logic related to evaluation metrics.

Includes automatic cache invalidation after mutations to ensure
cache consistency with database state.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from fastapi import HTTPException, status

from app.models import Metric, Run
from app.schemas import MetricCreate, MetricResponse, PaginationMeta


def _is_sqlite_mode() -> bool:
    """Check if running in SQLite test mode."""
    return (
        os.environ.get("TESTING", "false").lower() == "true" or
        os.environ.get("TEST_DATABASE_URL", "").startswith("sqlite")
    )


class MetricService:
    """Service class for metric operations."""

    def __init__(self, db: Session, cache_service=None):
        """
        Initialize metric service.

        Args:
            db: Database session
            cache_service: Optional CacheService instance for auto-invalidation
        """
        self.db = db
        self.cache_service = cache_service

    def _invalidate_metric_cache(self, run_id: Optional[str] = None) -> None:
        """
        Invalidate cache entries related to metrics.

        Args:
            run_id: Optional run ID for specific invalidation
        """
        if self.cache_service is None:
            return

        try:
            from app.cache import get_cache_manager

            cache = get_cache_manager()
            if cache and cache.connect():
                # Invalidate run metrics cache
                if run_id:
                    cache.delete(f"cache:metrics:run:{run_id}")

                # Invalidate aggregated metrics caches
                cache.invalidate_prefix("cache:metrics:aggregated")
        except Exception as e:
            # Log but don't fail the operation if cache invalidation fails
            pass

    def get_metrics(
        self,
        run_id: str | None = None,
        category: str | None = None,
        name: str | None = None,
        page: int = 1,
        per_page: int = 100,
    ) -> tuple[list[MetricResponse], PaginationMeta]:
        """
        Get paginated list of metrics with optional filtering.

        Args:
            run_id: Filter by run ID
            category: Filter by category
            name: Filter by metric name
            page: Page number (1-indexed)
            per_page: Items per page

        Returns:
            Tuple of (list of metrics, pagination metadata)
        """
        query = select(Metric)

        # Apply filters
        if run_id:
            query = query.where(Metric.run_id == run_id)
        if category:
            query = query.where(Metric.category == category)
        if name:
            query = query.where(Metric.name.ilike(f"%{name}%"))

        # Get total count
        total_query = select(func.count()).select_from(query.subquery())
        total = self.db.execute(total_query).scalar()

        # Apply pagination
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page)

        # Execute query
        results = self.db.execute(query).scalars().all()

        # Calculate total pages
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0

        meta = PaginationMeta(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
        )

        return [MetricResponse.model_validate(m) for m in results], meta

    def get_metric(self, metric_id: str) -> MetricResponse | None:
        """
        Get a single metric by ID.

        Args:
            metric_id: Metric UUID

        Returns:
            Metric response or None if not found
        """
        query = select(Metric).where(Metric.id == metric_id)
        metric = self.db.execute(query).scalar_one_or_none()

        if metric:
            return MetricResponse.model_validate(metric)
        return None

    def create_metric(self, metric_data: MetricCreate) -> MetricResponse:
        """
        Create a new metric.

        Args:
            metric_data: Metric creation data

        Returns:
            Created metric response

        Raises:
            HTTPException: If database error occurs
        """
        try:
            metric = Metric(**metric_data.model_dump())
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)

            # Invalidate cache after successful commit
            self._invalidate_metric_cache(str(metric.run_id))

            return MetricResponse.model_validate(metric)
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Metric already exists or foreign key constraint failed",
            ) from e
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while creating metric: {str(e)}",
            ) from e

    def create_metrics_bulk(
        self, metrics_data: list[MetricCreate]
    ) -> list[MetricResponse]:
        """
        Create multiple metrics in a single batch.

        Args:
            metrics_data: List of metric creation data

        Returns:
            List of created metric responses

        Raises:
            HTTPException: If database error occurs
        """
        try:
            metrics = [Metric(**m.model_dump()) for m in metrics_data]
            self.db.add_all(metrics)
            self.db.commit()

            for metric in metrics:
                self.db.refresh(metric)

            # Invalidate cache after successful commit
            # Get unique run_ids from metrics_data
            run_ids = set(m.run_id for m in metrics_data)
            for run_id in run_ids:
                self._invalidate_metric_cache(run_id)

            return [MetricResponse.model_validate(m) for m in metrics]
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="One or more metrics already exist or foreign key constraint failed",
            ) from e
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while creating metrics: {str(e)}",
            ) from e

    def delete_metric(self, metric_id: str) -> bool:
        """
        Delete a metric.

        Args:
            metric_id: Metric UUID

        Returns:
            True if deleted, False if not found

        Raises:
            HTTPException: If database error occurs
        """
        try:
            query = select(Metric).where(Metric.id == metric_id)
            metric = self.db.execute(query).scalar_one_or_none()

            if not metric:
                return False

            run_id = str(metric.run_id)

            self.db.delete(metric)
            self.db.commit()

            # Invalidate cache after successful commit
            self._invalidate_metric_cache(run_id)

            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while deleting metric: {str(e)}",
            ) from e

    def get_aggregate_metrics(
        self,
        model_id: str | None = None,
        run_type: str | None = None,
        category: str | None = None,
        metric_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict]:
        """
        Get aggregated metrics across runs.

        Args:
            model_id: Filter by model ID
            run_type: Filter by run type
            category: Filter by metric category
            metric_name: Filter by metric name
            date_from: Filter by start date
            date_to: Filter by end date

        Returns:
            List of aggregated metric dicts
        """
        # Build base query for runs
        run_query = select(Run.id)
        if model_id:
            run_query = run_query.where(Run.model_id == model_id)
        if run_type:
            run_query = run_query.where(Run.run_type == run_type)
        if date_from:
            run_query = run_query.where(Run.created_at >= date_from)
        if date_to:
            run_query = run_query.where(Run.created_at <= date_to)

        run_ids = self.db.execute(run_query).scalars().all()

        if not run_ids:
            return []

        # SQLite doesn't support stddev, so we calculate it manually
        if _is_sqlite_mode():
            # For SQLite, use subqueries to calculate stddev
            results = []

            # Get distinct metric names and units
            base_query = select(Metric.name, Metric.unit).where(
                Metric.run_id.in_(run_ids),
                Metric.value_numeric.isnot(None),
            )
            if category:
                base_query = base_query.where(Metric.category == category)
            if metric_name:
                base_query = base_query.where(Metric.name == metric_name)
            base_query = base_query.group_by(Metric.name, Metric.unit)

            metric_keys = self.db.execute(base_query).all()

            for name, unit in metric_keys:
                # Get all values for this metric
                values_query = select(Metric.value_numeric).where(
                    Metric.run_id.in_(run_ids),
                    Metric.name == name,
                    Metric.unit == unit,
                    Metric.value_numeric.isnot(None),
                )
                if category:
                    values_query = values_query.where(Metric.category == category)
                values = self.db.execute(values_query).scalars().all()

                if values:
                    n = len(values)
                    # Convert to float for calculations (SQLAlchemy Numeric returns Decimal)
                    float_values = [float(v) if v is not None else 0.0 for v in values]
                    mean_val = sum(float_values) / n
                    variance = sum((x - mean_val) ** 2 for x in float_values) / n if n > 0 else 0
                    std_dev = float(variance) ** 0.5

                    results.append({
                        "name": name,
                        "unit": unit,
                        "mean": float(mean_val),
                        "std_dev": float(std_dev),
                        "min": float(min(values)),
                        "max": float(max(values)),
                        "count": n,
                    })

            return results
        else:
            # PostgreSQL query with stddev
            query = select(
                Metric.name,
                Metric.unit,
                func.avg(Metric.value_numeric).label("mean"),
                func.stddev(Metric.value_numeric).label("std_dev"),
                func.min(Metric.value_numeric).label("min"),
                func.max(Metric.value_numeric).label("max"),
                func.count().label("count"),
            ).where(
                Metric.run_id.in_(run_ids),
                Metric.value_numeric.isnot(None),
            )

            if category:
                query = query.where(Metric.category == category)
            if metric_name:
                query = query.where(Metric.name == metric_name)

            query = query.group_by(Metric.name, Metric.unit)

            results = self.db.execute(query).all()

            return [
                {
                    "name": r.name,
                    "unit": r.unit,
                    "mean": float(r.mean) if r.mean else None,
                    "std_dev": float(r.std_dev) if r.std_dev else None,
                    "min": float(r.min) if r.min else None,
                    "max": float(r.max) if r.max else None,
                    "count": r.count,
                }
                for r in results
            ]

    def get_metric_trends(
        self,
        model_id: str,
        metric_name: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get metric trends over time for a model.

        Args:
            model_id: Model UUID
            metric_name: Name of the metric
            limit: Maximum number of data points

        Returns:
            List of trend data points with timestamps
        """
        query = (
            select(
                Run.created_at,
                Metric.value_numeric,
                Metric.unit,
            )
            .join(Run, Metric.run_id == Run.id)
            .where(
                Run.model_id == model_id,
                Metric.name == metric_name,
                Metric.value_numeric.isnot(None),
            )
            .order_by(Run.created_at.desc())
            .limit(limit)
        )

        results = self.db.execute(query).all()

        return [
            {
                "timestamp": r.created_at.isoformat(),
                "value": float(r.value_numeric),
                "unit": r.unit,
            }
            for r in results
        ]

    def compare_metrics(
        self,
        run_ids: list[str],
        categories: list[str] | None = None,
    ) -> dict:
        """
        Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare
            categories: Optional list of categories to include

        Returns:
            Dict with comparison data keyed by run_id
        """
        query = select(Metric).where(Metric.run_id.in_(run_ids))

        if categories:
            query = query.where(Metric.category.in_(categories))

        metrics = self.db.execute(query).scalars().all()

        # Group by run_id
        comparison = {}
        for metric in metrics:
            if metric.run_id not in comparison:
                comparison[metric.run_id] = []
            comparison[metric.run_id].append(
                MetricResponse.model_validate(metric).model_dump()
            )

        return comparison

    def get_performance_metrics(
        self, run_id: str
    ) -> dict[str, MetricResponse | None]:
        """
        Get standard performance metrics for a run.

        Args:
            run_id: Run UUID

        Returns:
            Dict of performance metrics
        """
        perf_metrics = {
            "seconds_to_first_token": None,
            "prefill_tokens_per_second": None,
            "token_generation_tokens_per_second": None,
            "max_memory_used_gbyte": None,
        }

        query = select(Metric).where(
            Metric.run_id == run_id,
            Metric.category == "performance",
        )
        metrics = self.db.execute(query).scalars().all()

        for metric in metrics:
            if metric.name in perf_metrics:
                perf_metrics[metric.name] = MetricResponse.model_validate(metric)

        return perf_metrics
