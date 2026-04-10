"""
Run service for business logic related to evaluation runs.

Includes automatic cache invalidation after mutations to ensure
cache consistency with database state.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from fastapi import HTTPException, status

from app.models import Run, Metric
from app.schemas import RunCreate, RunUpdate, RunResponse, PaginationMeta


class RunService:
    """Service class for run operations."""

    def __init__(self, db: Session, cache_service=None):
        """
        Initialize run service.

        Args:
            db: Database session
            cache_service: Optional CacheService instance for auto-invalidation
        """
        self.db = db
        self.cache_service = cache_service

    def get_runs(
        self,
        page: int = 1,
        per_page: int = 20,
        model_id: str | None = None,
        status: str | None = None,
        run_type: str | None = None,
        device: str | None = None,
        backend: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[list[RunResponse], PaginationMeta]:
        """
        Get paginated list of runs with optional filtering.

        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            model_id: Filter by model ID
            status: Filter by run status
            run_type: Filter by run type
            device: Filter by device type
            backend: Filter by backend
            date_from: Filter by start date
            date_to: Filter by end date

        Returns:
            Tuple of (list of runs, pagination metadata)
        """
        query = select(Run).options(joinedload(Run.model))

        # Apply filters
        if model_id:
            query = query.where(Run.model_id == model_id)
        if status:
            query = query.where(Run.status == status)
        if run_type:
            query = query.where(Run.run_type == run_type)
        if device:
            query = query.where(Run.device == device)
        if backend:
            query = query.where(Run.backend == backend)
        if date_from:
            query = query.where(Run.created_at >= date_from)
        if date_to:
            query = query.where(Run.created_at <= date_to)

        # Get total count
        total_query = select(func.count()).select_from(query.subquery())
        total = self.db.execute(total_query).scalar()

        # Apply pagination
        offset = (page - 1) * per_page
        query = query.order_by(Run.created_at.desc()).offset(offset).limit(per_page)

        # Execute query
        results = self.db.execute(query).unique().scalars().all()

        # Calculate total pages
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0

        meta = PaginationMeta(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
        )

        return [RunResponse.model_validate(r) for r in results], meta

    def get_run(self, run_id: str, include_metrics: bool = False) -> dict | None:
        """
        Get a single run by ID.

        Args:
            run_id: Run UUID
            include_metrics: Whether to include metrics in response

        Returns:
            Run dict or None if not found
        """
        query = select(Run).where(Run.id == run_id)

        if include_metrics:
            from sqlalchemy.orm import selectinload
            query = query.options(selectinload(Run.metrics))

        run = self.db.execute(query).unique().scalar_one_or_none()

        if not run:
            return None

        run_dict = RunResponse.model_validate(run).model_dump()

        if include_metrics:
            from app.schemas import MetricResponse
            run_dict["metrics"] = [
                MetricResponse.model_validate(m).model_dump() for m in run.metrics
            ]

        return run_dict

    def create_run(self, run_data: RunCreate) -> RunResponse:
        """
        Create a new evaluation run.

        Args:
            run_data: Run creation data

        Returns:
            Created run response

        Raises:
            HTTPException: If database error occurs
        """
        try:
            run = Run(**run_data.model_dump())
            self.db.add(run)
            self.db.commit()
            self.db.refresh(run)

            # Invalidate cache after successful commit
            self._invalidate_run_cache(str(run.id), run.model_id)

            return RunResponse.model_validate(run)
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Run already exists or foreign key constraint failed",
            ) from e
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while creating run: {str(e)}",
            ) from e

    def _invalidate_run_cache(self, run_id: str, model_id: Optional[str] = None) -> None:
        """
        Invalidate cache entries related to a run.

        Args:
            run_id: Run ID to invalidate
            model_id: Optional model ID for additional invalidation
        """
        if self.cache_service is None:
            return

        try:
            import asyncio
            from app.cache import get_cache_manager

            cache = get_cache_manager()
            if cache and cache.connect():
                # Invalidate run summary
                cache.delete(f"cache:runs:summary:{run_id}")

                # Invalidate runs list caches
                cache.invalidate_prefix("cache:runs:list")

                # Invalidate model-related caches if model_id provided
                if model_id:
                    cache.invalidate_prefix("cache:models")
        except Exception as e:
            # Log but don't fail the operation if cache invalidation fails
            pass

    def update_run(self, run_id: str, run_data: RunUpdate) -> RunResponse | None:
        """
        Update an existing run.

        Args:
            run_id: Run UUID
            run_data: Run update data

        Returns:
            Updated run response or None if not found

        Raises:
            HTTPException: If database error occurs
        """
        try:
            query = select(Run).where(Run.id == run_id)
            run = self.db.execute(query).scalar_one_or_none()

            if not run:
                return None

            # Update only provided fields
            update_data = run_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(run, field, value)

            self.db.commit()
            self.db.refresh(run)

            # Invalidate cache after successful commit
            self._invalidate_run_cache(str(run.id), str(run.model_id) if run.model_id else None)

            return RunResponse.model_validate(run)
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while updating run: {str(e)}",
            ) from e

    def update_status(
        self, run_id: str, status: str, message: str | None = None
    ) -> RunResponse | None:
        """
        Update run status.

        Args:
            run_id: Run UUID
            status: New status
            message: Optional status message

        Returns:
            Updated run response or None if not found

        Raises:
            HTTPException: If database error occurs
        """
        try:
            query = select(Run).where(Run.id == run_id)
            run = self.db.execute(query).scalar_one_or_none()

            if not run:
                return None

            run.status = status
            if message:
                run.status_message = message

            if status == "running" and not run.started_at:
                run.started_at = datetime.now(timezone.utc)
            elif status in ("completed", "failed", "cancelled"):
                run.completed_at = datetime.now(timezone.utc)
                if run.started_at:
                    run.duration_seconds = (run.completed_at - run.started_at).total_seconds()

            self.db.commit()
            self.db.refresh(run)

            # Invalidate cache after successful commit
            self._invalidate_run_cache(str(run.id), str(run.model_id) if run.model_id else None)

            return RunResponse.model_validate(run)
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while updating run status: {str(e)}",
            ) from e

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run.

        Args:
            run_id: Run UUID

        Returns:
            True if deleted, False if not found

        Raises:
            HTTPException: If database error occurs
        """
        try:
            query = select(Run).where(Run.id == run_id)
            run = self.db.execute(query).scalar_one_or_none()

            if not run:
                return False

            model_id = str(run.model_id) if run.model_id else None

            self.db.delete(run)
            self.db.commit()

            # Invalidate cache after successful commit
            self._invalidate_run_cache(run_id, model_id)

            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while deleting run: {str(e)}",
            ) from e

    def get_run_metrics(self, run_id: str) -> list[dict]:
        """
        Get all metrics for a run.

        Args:
            run_id: Run UUID

        Returns:
            List of metric dicts
        """
        from app.schemas import MetricResponse

        query = select(Metric).where(Metric.run_id == run_id)
        metrics = self.db.execute(query).scalars().all()
        return [MetricResponse.model_validate(m).model_dump() for m in metrics]

    def get_recent_runs(self, limit: int = 10) -> list[RunResponse]:
        """
        Get recent runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of recent runs
        """
        query = (
            select(Run)
            .order_by(Run.created_at.desc())
            .limit(limit)
        )
        results = self.db.execute(query).scalars().all()
        return [RunResponse.model_validate(r) for r in results]

    def get_run_stats(self) -> dict:
        """
        Get overall run statistics.

        Returns:
            Dict with run statistics
        """
        # Total runs
        total_query = select(func.count()).select_from(Run)
        total = self.db.execute(total_query).scalar()

        # Runs by status
        status_query = select(Run.status, func.count()).group_by(Run.status)
        status_counts = dict(self.db.execute(status_query).all())

        # Runs by type
        type_query = select(Run.run_type, func.count()).group_by(Run.run_type)
        type_counts = dict(self.db.execute(type_query).all())

        return {
            "total_runs": total,
            "by_status": status_counts,
            "by_type": type_counts,
        }

    async def get_by_build_name(self, build_name: str) -> Optional[Run]:
        """
        Get run by build name.

        Args:
            build_name: Build name to search for

        Returns:
            Run instance or None if not found
        """
        query = select(Run).where(Run.build_name == build_name)
        return self.db.execute(query).scalar_one_or_none()

    async def create_scheduled_run(
        self,
        model_id: str,
        run_type: str,
        schedule_id: str = None,
    ) -> Run:
        """
        Create a run from scheduled evaluation.

        Args:
            model_id: Model ID
            run_type: Evaluation type
            schedule_id: Optional schedule ID

        Returns:
            Created Run instance
        """
        from datetime import datetime, timezone

        build_name = f"scheduled-{run_type}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        run = Run(
            model_id=model_id,
            build_name=build_name,
            run_type=run_type,
            status="running",
            config={"schedule_id": schedule_id} if schedule_id else {},
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)

        # Invalidate cache after successful commit
        self._invalidate_run_cache(str(run.id), str(model_id))

        return run

    def complete_run(self, run_id: str, output: str = None) -> Run:
        """
        Mark a run as completed.

        Args:
            run_id: Run ID
            output: Optional output data

        Returns:
            Updated Run instance
        """
        from datetime import datetime, timezone

        run = self.db.get(Run, run_id)
        if run:
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            self.db.commit()
            self.db.refresh(run)

            # Invalidate cache after successful commit
            self._invalidate_run_cache(run_id, str(run.model_id) if run.model_id else None)

        return run
