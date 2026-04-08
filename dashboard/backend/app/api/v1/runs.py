"""
API routes for run operations.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas import (
    RunCreate,
    RunUpdate,
    RunResponse,
    RunListResponse,
    APIResponse,
)
from app.services.runs import RunService

router = APIRouter(prefix="/runs", tags=["Runs"])


@router.get("", response_model=RunListResponse)
async def list_runs(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    model_id: str | None = Query(None, description="Filter by model ID"),
    status: str | None = Query(None, description="Filter by status"),
    run_type: str | None = Query(None, description="Filter by run type"),
    device: str | None = Query(None, description="Filter by device"),
    backend: str | None = Query(None, description="Filter by backend"),
    db: Session = Depends(get_db_session),
):
    """
    List evaluation runs with pagination and filtering.

    **Filters:**
    - `model_id`: Filter by specific model
    - `status`: Filter by status (pending, running, completed, failed)
    - `run_type`: Filter by type (benchmark, accuracy-mmlu, etc.)
    - `device`: Filter by device (cpu, gpu, npu)
    - `backend`: Filter by backend (llamacpp, ort, flm)
    """
    service = RunService(db)
    runs, meta = service.get_runs(
        page=page,
        per_page=per_page,
        model_id=model_id,
        status=status,
        run_type=run_type,
        device=device,
        backend=backend,
    )

    return RunListResponse(
        success=True,
        data=runs,
        meta=meta,
    )


@router.post("", response_model=APIResponse, status_code=201)
async def create_run(
    run_data: RunCreate,
    db: Session = Depends(get_db_session),
):
    """
    Create a new evaluation run.

    **Required fields:**
    - `model_id`: ID of model to evaluate
    - `build_name`: Unique build identifier
    - `run_type`: Type of evaluation
    """
    service = RunService(db)
    run = service.create_run(run_data)

    return APIResponse(
        success=True,
        data=run.model_dump(),
    )


@router.get("/recent/list", response_model=APIResponse)
async def get_recent_runs(
    limit: int = Query(10, ge=1, le=100, description="Number of runs to return"),
    db: Session = Depends(get_db_session),
):
    """
    Get recent runs.
    """
    service = RunService(db)
    runs = service.get_recent_runs(limit)

    return APIResponse(
        success=True,
        data=[r.model_dump() for r in runs],
    )


@router.get("/stats", response_model=APIResponse)
async def get_run_stats(
    db: Session = Depends(get_db_session),
):
    """
    Get overall run statistics.
    """
    service = RunService(db)
    stats = service.get_run_stats()

    return APIResponse(
        success=True,
        data=stats,
    )


@router.get("/{run_id}", response_model=APIResponse)
async def get_run(
    run_id: str,
    include_metrics: bool = Query(False, description="Include metrics in response"),
    db: Session = Depends(get_db_session),
):
    """
    Get a specific run by ID.

    Use `include_metrics=true` to get metrics data.
    """
    service = RunService(db)
    run = service.get_run(run_id, include_metrics=include_metrics)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return APIResponse(
        success=True,
        data=run,
    )


@router.put("/{run_id}", response_model=APIResponse)
async def update_run(
    run_id: str,
    run_data: RunUpdate,
    db: Session = Depends(get_db_session),
):
    """
    Update an existing run.

    Common updates:
    - Change `status` (running, completed, failed)
    - Add `status_message`
    - Update timing fields
    """
    service = RunService(db)
    run = service.update_run(run_id, run_data)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return APIResponse(
        success=True,
        data=run.model_dump(),
    )


@router.post("/{run_id}/status", response_model=APIResponse)
async def update_run_status(
    run_id: str,
    status: str = Query(..., description="New status"),
    message: str | None = Query(None, description="Status message"),
    db: Session = Depends(get_db_session),
):
    """
    Update run status.

    Valid statuses: pending, running, completed, failed, cancelled

    Automatically sets started_at/completed_at timestamps.
    """
    valid_statuses = {"pending", "running", "completed", "failed", "cancelled"}
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}",
        )

    service = RunService(db)
    run = service.update_status(run_id, status, message)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return APIResponse(
        success=True,
        data=run.model_dump(),
    )


@router.delete("/{run_id}", response_model=APIResponse)
async def delete_run(
    run_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Delete a run.

    This will cascade delete all associated metrics.
    """
    service = RunService(db)
    deleted = service.delete_run(run_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")

    return APIResponse(
        success=True,
        data={"message": f"Run {run_id} deleted successfully"},
    )


@router.get("/{run_id}/metrics", response_model=APIResponse)
async def get_run_metrics(
    run_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get all metrics for a run.
    """
    service = RunService(db)

    # Verify run exists
    if not service.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")

    metrics = service.get_run_metrics(run_id)

    return APIResponse(
        success=True,
        data=metrics,
    )


@router.get("/benchmark/results", response_model=APIResponse)
async def get_benchmark_results(
    db: Session = Depends(get_db_session),
):
    """
    Get benchmark results grouped by model.

    Returns benchmark runs with their metrics, organized for comparison.
    """
    service = RunService(db)

    # Get all benchmark runs
    runs, _ = service.get_runs(run_type="benchmark", per_page=100)

    # Get metrics for each run
    results = []
    for run in runs:
        metrics = service.get_run_metrics(run.id)
        results.append({
            "run": run.model_dump(),
            "metrics": [m.model_dump() for m in metrics],
        })

    return APIResponse(
        success=True,
        data=results,
    )
