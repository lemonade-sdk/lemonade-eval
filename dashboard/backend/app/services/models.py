"""
Model service for business logic related to ML models.
"""

from typing import Optional
from uuid import uuid4

from sqlalchemy import select, func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from fastapi import HTTPException, status

from app.models import Model, ModelVersion
from app.schemas import ModelCreate, ModelUpdate, ModelResponse, PaginationMeta


class ModelService:
    """Service class for model operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_models(
        self,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        family: str | None = None,
        model_type: str | None = None,
    ) -> tuple[list[ModelResponse], PaginationMeta]:
        """
        Get paginated list of models with optional filtering.

        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            search: Search term for name/checkpoint
            family: Filter by model family
            model_type: Filter by model type

        Returns:
            Tuple of (list of models, pagination metadata)
        """
        query = select(Model)

        # Apply filters
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                (Model.name.ilike(search_pattern)) |
                (Model.checkpoint.ilike(search_pattern))
            )
        if family:
            query = query.where(Model.family == family)
        if model_type:
            query = query.where(Model.model_type == model_type)

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

        return [ModelResponse.model_validate(m) for m in results], meta

    def get_model(self, model_id: str) -> ModelResponse | None:
        """
        Get a single model by ID.

        Args:
            model_id: Model UUID

        Returns:
            Model response or None if not found
        """
        query = select(Model).where(Model.id == model_id)
        model = self.db.execute(query).scalar_one_or_none()

        if model:
            return ModelResponse.model_validate(model)
        return None

    def get_model_by_checkpoint(self, checkpoint: str) -> ModelResponse | None:
        """
        Get a model by its checkpoint identifier.

        Args:
            checkpoint: Model checkpoint string

        Returns:
            Model response or None if not found
        """
        query = select(Model).where(Model.checkpoint == checkpoint)
        model = self.db.execute(query).scalar_one_or_none()

        if model:
            return ModelResponse.model_validate(model)
        return None

    def create_model(
        self, model_data: ModelCreate, created_by: str | None = None
    ) -> ModelResponse:
        """
        Create a new model.

        Args:
            model_data: Model creation data
            created_by: Optional user ID of creator

        Returns:
            Created model response

        Raises:
            HTTPException: If checkpoint already exists or database error occurs
        """
        try:
            model = Model(
                **model_data.model_dump(),
                created_by=created_by,
            )
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)

            return ModelResponse.model_validate(model)
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with checkpoint already exists: {model_data.checkpoint}",
            ) from e
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while creating model: {str(e)}",
            ) from e

    def update_model(
        self, model_id: str, model_data: ModelUpdate
    ) -> ModelResponse | None:
        """
        Update an existing model.

        Args:
            model_id: Model UUID
            model_data: Model update data

        Returns:
            Updated model response or None if not found
        """
        try:
            query = select(Model).where(Model.id == model_id)
            model = self.db.execute(query).scalar_one_or_none()

            if not model:
                return None

            # Update only provided fields
            update_data = model_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(model, field, value)

            self.db.commit()
            self.db.refresh(model)

            return ModelResponse.model_validate(model)
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while updating model: {str(e)}",
            ) from e

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.

        Args:
            model_id: Model UUID

        Returns:
            True if deleted, False if not found
        """
        try:
            query = select(Model).where(Model.id == model_id)
            model = self.db.execute(query).scalar_one_or_none()

            if not model:
                return False

            self.db.delete(model)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error while deleting model: {str(e)}",
            ) from e

    def get_model_versions(self, model_id: str) -> list[dict]:
        """
        Get all versions of a model.

        Args:
            model_id: Model UUID

        Returns:
            List of model version dicts
        """
        query = select(ModelVersion).where(ModelVersion.model_id == model_id)
        versions = self.db.execute(query).scalars().all()
        return [{"id": v.id, "version": v.version, "quantization": v.quantization}
                for v in versions]

    def get_model_runs(self, model_id: str, limit: int = 100) -> list[dict]:
        """
        Get recent runs for a model.

        Args:
            model_id: Model UUID
            limit: Maximum number of runs to return

        Returns:
            List of run summaries
        """
        from app.models import Run

        query = (
            select(Run)
            .where(Run.model_id == model_id)
            .order_by(Run.created_at.desc())
            .limit(limit)
        )
        runs = self.db.execute(query).scalars().all()
        return [
            {
                "id": r.id,
                "build_name": r.build_name,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ]

    def search_families(self) -> list[str]:
        """Get list of unique model families."""
        query = select(Model.family).distinct().where(Model.family.isnot(None))
        results = self.db.execute(query).scalars().all()
        return [f for f in results if f]
