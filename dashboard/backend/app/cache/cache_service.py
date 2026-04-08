"""
Cache service providing high-level caching operations for specific domains.

Services:
- Model caching
- Metrics caching
- Run caching
- Comparison caching
"""

import asyncio
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache.cache_manager import CacheManager, get_cache_manager
from app.models import Model, Run, Metric


class CacheService:
    """High-level cache service for dashboard operations."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize cache service.

        Args:
            cache_manager: CacheManager instance (uses global if not provided)
        """
        self.cache = cache_manager or get_cache_manager()

    # ========================================================================
    # MODEL CACHING
    # ========================================================================

    async def get_model_list(self, db: Session) -> List[dict]:
        """
        Get cached list of all models.

        Args:
            db: Database session

        Returns:
            List of model dicts
        """
        if self.cache is None or not self.cache.connect():
            return self._fetch_models_from_db(db)

        cache_key = "cache:models:list:all"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Fetch from database
        models = self._fetch_models_from_db(db)

        # Cache the result
        self.cache.set(cache_key, models, prefix="models")
        return models

    def _fetch_models_from_db(self, db: Session) -> List[dict]:
        """Fetch models from database."""
        result = db.execute(select(Model)).scalars().all()
        return [
            {
                "id": model.id,
                "name": model.name,
                "checkpoint": model.checkpoint,
                "model_type": model.model_type,
                "family": model.family,
                "created_at": model.created_at.isoformat() if model.created_at else None,
            }
            for model in result
        ]

    async def invalidate_models(self) -> bool:
        """Invalidate all model cache entries."""
        if self.cache is None:
            return False
        return self.cache.invalidate_prefix("models") > 0

    # ========================================================================
    # METRICS CACHING
    # ========================================================================

    async def get_run_metrics(self, db: Session, run_id: str) -> List[dict]:
        """
        Get cached metrics for a specific run.

        Args:
            db: Database session
            run_id: Run ID

        Returns:
            List of metric dicts
        """
        if self.cache is None or not self.cache.connect():
            return self._fetch_run_metrics_from_db(db, run_id)

        cache_key = f"cache:metrics:run:{run_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Fetch from database
        metrics = self._fetch_run_metrics_from_db(db, run_id)

        # Cache the result
        self.cache.set(cache_key, metrics, prefix="metrics")
        return metrics

    def _fetch_run_metrics_from_db(self, db: Session, run_id: str) -> List[dict]:
        """Fetch metrics for a run from database."""
        result = db.execute(
            select(Metric).where(Metric.run_id == run_id)
        ).scalars().all()
        return [
            {
                "id": metric.id,
                "run_id": metric.run_id,
                "category": metric.category,
                "name": metric.name,
                "display_name": metric.display_name,
                "value_numeric": float(metric.value_numeric) if metric.value_numeric else None,
                "unit": metric.unit,
                "created_at": metric.created_at.isoformat() if metric.created_at else None,
            }
            for metric in result
        ]

    async def invalidate_run_metrics(self, run_id: str) -> bool:
        """
        Invalidate metrics cache for a specific run.

        Args:
            run_id: Run ID to invalidate

        Returns:
            True if invalidated
        """
        if self.cache is None:
            return False
        cache_key = f"cache:metrics:run:{run_id}"
        return self.cache.delete(cache_key)

    async def get_aggregated_metrics(
        self,
        db: Session,
        model_id: str,
        metric_name: str,
        days: int = 30,
    ) -> dict:
        """
        Get cached aggregated metrics.

        Args:
            db: Database session
            model_id: Model ID
            metric_name: Metric name
            days: Number of days to aggregate

        Returns:
            Aggregated metrics dict
        """
        if self.cache is None or not self.cache.connect():
            return self._compute_aggregated_metrics(db, model_id, metric_name, days)

        cache_key = f"cache:metrics:aggregated:{model_id}:{metric_name}:days:{days}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Compute aggregation
        aggregated = self._compute_aggregated_metrics(db, model_id, metric_name, days)

        # Cache the result
        self.cache.set(cache_key, aggregated, prefix="metrics")
        return aggregated

    def _compute_aggregated_metrics(
        self,
        db: Session,
        model_id: str,
        metric_name: str,
        days: int,
    ) -> dict:
        """Compute aggregated metrics from database."""
        from datetime import datetime, timedelta
        from sqlalchemy import func

        cutoff = datetime.utcnow() - timedelta(days=days)

        result = db.execute(
            select(
                func.avg(Metric.value_numeric),
                func.min(Metric.value_numeric),
                func.max(Metric.value_numeric),
                func.count(Metric.id),
            )
            .join(Run, Metric.run_id == Run.id)
            .where(
                Run.model_id == model_id,
                Metric.name == metric_name,
                Run.created_at >= cutoff,
            )
        ).first()

        if result and result[0] is not None:
            return {
                "model_id": model_id,
                "metric_name": metric_name,
                "days": days,
                "avg": float(result[0]),
                "min": float(result[1]) if result[1] else None,
                "max": float(result[2]) if result[2] else None,
                "count": result[3],
            }
        return {
            "model_id": model_id,
            "metric_name": metric_name,
            "days": days,
            "avg": None,
            "min": None,
            "max": None,
            "count": 0,
        }

    # ========================================================================
    # RUN CACHING
    # ========================================================================

    async def get_run_summary(self, db: Session, run_id: str) -> Optional[dict]:
        """
        Get cached run summary.

        Args:
            db: Database session
            run_id: Run ID

        Returns:
            Run summary dict or None
        """
        if self.cache is None or not self.cache.connect():
            return self._fetch_run_summary_from_db(db, run_id)

        cache_key = f"cache:runs:summary:{run_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Fetch from database
        summary = self._fetch_run_summary_from_db(db, run_id)

        if summary:
            # Cache the result
            self.cache.set(cache_key, summary, prefix="runs")

        return summary

    def _fetch_run_summary_from_db(self, db: Session, run_id: str) -> Optional[dict]:
        """Fetch run summary from database."""
        run = db.get(Run, run_id)
        if not run:
            return None

        return {
            "id": run.id,
            "model_id": run.model_id,
            "build_name": run.build_name,
            "run_type": run.run_type,
            "status": run.status,
            "device": run.device,
            "backend": run.backend,
            "dtype": run.dtype,
            "duration_seconds": float(run.duration_seconds) if run.duration_seconds else None,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

    async def invalidate_run(self, run_id: str) -> bool:
        """
        Invalidate all cache entries for a run.

        Args:
            run_id: Run ID to invalidate

        Returns:
            True if invalidated
        """
        if self.cache is None:
            return False

        # Invalidate run summary
        self.cache.delete(f"cache:runs:summary:{run_id}")

        # Invalidate run metrics
        self.cache.delete(f"cache:metrics:run:{run_id}")

        return True

    async def get_runs_list(
        self,
        db: Session,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """
        Get cached list of runs with optional filters.

        Args:
            db: Database session
            model_id: Optional model ID filter
            status: Optional status filter
            limit: Maximum number of runs

        Returns:
            List of run dicts
        """
        if self.cache is None or not self.cache.connect():
            return self._fetch_runs_from_db(db, model_id, status, limit)

        cache_key = f"cache:runs:list:model:{model_id or 'all'}:status:{status or 'all'}:limit:{limit}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Fetch from database
        runs = self._fetch_runs_from_db(db, model_id, status, limit)

        # Cache the result
        self.cache.set(cache_key, runs, prefix="runs")
        return runs

    def _fetch_runs_from_db(
        self,
        db: Session,
        model_id: Optional[str],
        status: Optional[str],
        limit: int,
    ) -> List[dict]:
        """Fetch runs from database with filters."""
        query = select(Run)

        if model_id:
            query = query.where(Run.model_id == model_id)
        if status:
            query = query.where(Run.status == status)

        query = query.order_by(Run.created_at.desc()).limit(limit)
        result = db.execute(query).scalars().all()

        return [
            {
                "id": run.id,
                "model_id": run.model_id,
                "build_name": run.build_name,
                "run_type": run.run_type,
                "status": run.status,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
            for run in result
        ]

    async def invalidate_runs_list(self) -> bool:
        """Invalidate all runs list cache entries."""
        if self.cache is None:
            return False
        return self.cache.invalidate_prefix("runs") > 0

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(self) -> dict:
        """
        Check cache service health.

        Returns:
            Health status dict
        """
        if self.cache is None:
            return {"status": "unavailable", "cache": None}
        return self.cache.health_check()
