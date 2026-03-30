"""
API routes for model operations.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    ModelListResponse,
    APIResponse,
    PaginationMeta,
)
from app.services.models import ModelService

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search term for name/checkpoint"),
    family: str | None = Query(None, description="Filter by model family"),
    model_type: str | None = Query(None, description="Filter by model type"),
    db: Session = Depends(get_db_session),
):
    """
    List models with pagination and filtering.

    **Filters:**
    - `search`: Search in model name and checkpoint
    - `family`: Filter by model family (e.g., Llama, Qwen)
    - `model_type`: Filter by type (llm, vlm, embedding)
    """
    service = ModelService(db)
    models, meta = service.get_models(
        page=page,
        per_page=per_page,
        search=search,
        family=family,
        model_type=model_type,
    )

    return ModelListResponse(
        success=True,
        data=models,
        meta=meta,
    )


@router.post("", response_model=APIResponse, status_code=201)
async def create_model(
    model_data: ModelCreate,
    db: Session = Depends(get_db_session),
):
    """
    Create a new model.

    **Required fields:**
    - `name`: Human-readable model name
    - `checkpoint`: Model checkpoint identifier (must be unique)
    """
    service = ModelService(db)

    # Check for duplicate checkpoint
    existing = service.get_model_by_checkpoint(model_data.checkpoint)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Model with checkpoint '{model_data.checkpoint}' already exists",
        )

    model = service.create_model(model_data)

    return APIResponse(
        success=True,
        data=model.model_dump(),
    )


@router.get("/{model_id}", response_model=APIResponse)
async def get_model(
    model_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get a specific model by ID.

    Returns model details including metadata.
    """
    service = ModelService(db)
    model = service.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return APIResponse(
        success=True,
        data=model.model_dump(),
    )


@router.put("/{model_id}", response_model=APIResponse)
async def update_model(
    model_id: str,
    model_data: ModelUpdate,
    db: Session = Depends(get_db_session),
):
    """
    Update an existing model.

    Only provided fields will be updated.
    """
    service = ModelService(db)
    model = service.update_model(model_id, model_data)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return APIResponse(
        success=True,
        data=model.model_dump(),
    )


@router.delete("/{model_id}", response_model=APIResponse)
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Delete a model.

    This will cascade delete all associated runs and metrics.
    """
    service = ModelService(db)
    deleted = service.delete_model(model_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")

    return APIResponse(
        success=True,
        data={"message": f"Model {model_id} deleted successfully"},
    )


@router.get("/{model_id}/versions", response_model=APIResponse)
async def get_model_versions(
    model_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get all versions of a model.
    """
    service = ModelService(db)

    # Verify model exists
    if not service.get_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")

    versions = service.get_model_versions(model_id)

    return APIResponse(
        success=True,
        data=versions,
    )


@router.get("/{model_id}/runs", response_model=APIResponse)
async def get_model_runs(
    model_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum runs to return"),
    db: Session = Depends(get_db_session),
):
    """
    Get recent runs for a model.
    """
    service = ModelService(db)

    # Verify model exists
    if not service.get_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")

    runs = service.get_model_runs(model_id, limit=limit)

    return APIResponse(
        success=True,
        data=runs,
    )


@router.get("/families/list", response_model=APIResponse)
async def list_families(
    db: Session = Depends(get_db_session),
):
    """
    Get list of unique model families.
    """
    service = ModelService(db)
    families = service.search_families()

    return APIResponse(
        success=True,
        data=families,
    )
