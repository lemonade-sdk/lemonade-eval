"""
API routes for metric operations.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas import (
    MetricCreate,
    MetricBulkCreate,
    MetricResponse,
    MetricListResponse,
    APIResponse,
)
from app.services.metrics import MetricService

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=MetricListResponse)
async def list_metrics(
    run_id: str | None = Query(None, description="Filter by run ID"),
    category: str | None = Query(None, description="Filter by category"),
    name: str | None = Query(None, description="Filter by metric name"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(100, ge=1, le=500, description="Items per page"),
    db: Session = Depends(get_db_session),
):
    """
    List metrics with pagination and filtering.

    **Filters:**
    - `run_id`: Filter by specific run
    - `category`: Filter by category (performance, accuracy)
    - `name`: Search metric name
    """
    service = MetricService(db)
    metrics, meta = service.get_metrics(
        run_id=run_id,
        category=category,
        name=name,
        page=page,
        per_page=per_page,
    )

    return MetricListResponse(
        success=True,
        data=metrics,
        meta=meta,
    )


@router.post("", response_model=APIResponse, status_code=201)
async def create_metric(
    metric_data: MetricCreate,
    db: Session = Depends(get_db_session),
):
    """
    Create a new metric.

    **Required fields:**
    - `run_id`: ID of associated run
    - `category`: Metric category (performance, accuracy)
    - `name`: Metric name
    - `value_numeric` or `value_text`: Metric value
    """
    service = MetricService(db)
    metric = service.create_metric(metric_data)

    return APIResponse(
        success=True,
        data=metric.model_dump(),
    )


@router.post("/bulk", response_model=APIResponse, status_code=201)
async def create_metrics_bulk(
    bulk_data: MetricBulkCreate,
    db: Session = Depends(get_db_session),
):
    """
    Create multiple metrics in a single batch.

    Useful for importing evaluation results.
    """
    service = MetricService(db)
    metrics = service.create_metrics_bulk(bulk_data.metrics)

    return APIResponse(
        success=True,
        data=[m.model_dump() for m in metrics],
    )


@router.get("/aggregate", response_model=APIResponse)
async def get_aggregate_metrics(
    model_id: str | None = Query(None, description="Filter by model ID"),
    run_type: str | None = Query(None, description="Filter by run type"),
    category: str | None = Query(None, description="Filter by category"),
    metric_name: str | None = Query(None, description="Filter by metric name"),
    db: Session = Depends(get_db_session),
):
    """
    Get aggregated metrics across runs.

    Returns mean, std_dev, min, max for each metric.
    """
    service = MetricService(db)
    aggregates = service.get_aggregate_metrics(
        model_id=model_id,
        run_type=run_type,
        category=category,
        metric_name=metric_name,
    )

    return APIResponse(
        success=True,
        data=aggregates,
    )


@router.get("/trends", response_model=APIResponse)
async def get_metric_trends(
    model_id: str = Query(..., description="Model ID"),
    metric_name: str = Query(..., description="Metric name"),
    limit: int = Query(100, ge=1, le=1000, description="Max data points"),
    db: Session = Depends(get_db_session),
):
    """
    Get metric trends over time for a model.

    Useful for plotting performance improvements over time.
    """
    service = MetricService(db)
    trends = service.get_metric_trends(
        model_id=model_id,
        metric_name=metric_name,
        limit=limit,
    )

    return APIResponse(
        success=True,
        data=trends,
    )


@router.get("/compare", response_model=APIResponse)
async def compare_metrics(
    run_ids: str = Query(..., description="Comma-separated run IDs"),
    categories: str | None = Query(None, description="Comma-separated categories"),
    db: Session = Depends(get_db_session),
):
    """
    Compare metrics across multiple runs.

    **Example:** `?run_ids=uuid1,uuid2,uuid3&categories=performance,accuracy`
    """
    run_id_list = run_ids.split(",")

    category_list = None
    if categories:
        category_list = categories.split(",")

    service = MetricService(db)
    comparison = service.compare_metrics(run_id_list, category_list)

    return APIResponse(
        success=True,
        data=comparison,
    )


@router.get("/performance/{run_id}", response_model=APIResponse)
async def get_performance_metrics(
    run_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get standard performance metrics for a run.

    Returns: TTFT, prefill TPS, generation TPS, memory usage
    """
    service = MetricService(db)

    # Verify run exists
    from app.services.runs import RunService
    run_service = RunService(db)
    if not run_service.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")

    metrics = service.get_performance_metrics(run_id)

    return APIResponse(
        success=True,
        data={k: v.model_dump() if v else None for k, v in metrics.items()},
    )


@router.get("/{metric_id}", response_model=APIResponse)
async def get_metric(
    metric_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get a specific metric by ID.
    """
    service = MetricService(db)
    metric = service.get_metric(metric_id)

    if not metric:
        raise HTTPException(status_code=404, detail="Metric not found")

    return APIResponse(
        success=True,
        data=metric.model_dump(),
    )


@router.delete("/{metric_id}", response_model=APIResponse)
async def delete_metric(
    metric_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Delete a metric.
    """
    service = MetricService(db)
    deleted = service.delete_metric(metric_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Metric not found")

    return APIResponse(
        success=True,
        data={"message": f"Metric {metric_id} deleted successfully"},
    )
