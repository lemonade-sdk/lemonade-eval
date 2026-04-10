"""
Client for receiving data from lemonade-eval CLI.

Features:
- Receive evaluation results via HTTP
- Validate incoming data
- Queue results for processing
- Send acknowledgments
"""

import hashlib
import hmac
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class EvaluationRunCreate(BaseModel):
    """Schema for creating an evaluation run from CLI."""

    model_id: str = Field(..., description="Model ID or checkpoint path")
    run_type: str = Field(..., description="Type of evaluation (benchmark, accuracy-mmlu, etc.)")
    build_name: str = Field(..., description="Unique build identifier")
    config: Dict[str, Any] = Field(default_factory=dict, description="Run configuration")
    device: Optional[str] = Field(None, description="Device type (cpu, gpu, npu)")
    backend: Optional[str] = Field(None, description="Backend runtime")
    dtype: Optional[str] = Field(None, description="Data type")
    started_at: Optional[str] = Field(None, description="Evaluation start time (ISO format)")


class MetricData(BaseModel):
    """Schema for a single metric from CLI."""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    category: str = Field(default="performance", description="Metric category")
    unit: Optional[str] = Field(None, description="Measurement unit")
    display_name: Optional[str] = Field(None, description="Human-readable name")


class EvaluationMetricsSubmit(BaseModel):
    """Schema for submitting evaluation metrics from CLI."""

    run_id: str = Field(..., description="Run ID")
    metrics: List[MetricData] = Field(..., description="List of metrics")


class EvaluationComplete(BaseModel):
    """Schema for marking evaluation as complete."""

    run_id: str = Field(..., description="Run ID")
    status: str = Field(..., description="Final status (completed, failed)")
    message: Optional[str] = Field(None, description="Status message")
    duration_seconds: Optional[float] = Field(None, description="Total duration")
    completed_at: Optional[str] = Field(None, description="Completion time (ISO format)")


class BulkEvaluationImport(BaseModel):
    """Schema for bulk importing multiple evaluations."""

    evaluations: List["BulkEvaluationEntry"] = Field(
        ...,
        description="List of evaluations to import",
    )
    skip_duplicates: bool = Field(
        default=True,
        description="Skip runs that already exist",
    )


class BulkEvaluationEntry(BaseModel):
    """Single evaluation entry for bulk import."""

    model_checkpoint: str = Field(..., description="Model checkpoint path")
    run_type: str = Field(..., description="Evaluation type")
    build_name: str = Field(..., description="Build identifier")
    metrics: List[MetricData] = Field(..., description="Evaluation metrics")
    config: Dict[str, Any] = Field(default_factory=dict, description="Run config")
    device: Optional[str] = None
    backend: Optional[str] = None
    dtype: Optional[str] = None
    status: str = Field(default="completed", description="Run status")
    duration_seconds: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# Update forward reference
BulkEvaluationImport.model_rebuild()


class ProgressUpdate(BaseModel):
    """Schema for progress updates via WebSocket."""

    run_id: str = Field(..., description="Run ID")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    message: Optional[str] = Field(None, description="Progress message")
    stage: Optional[str] = Field(None, description="Current evaluation stage")


# ============================================================================
# CLI CLIENT
# ============================================================================

class CLIClient:
    """
    Client for handling CLI integration.

    Provides methods for:
    - Creating runs from CLI
    - Receiving metrics
    - Handling progress updates
    """

    def __init__(self, db_session_factory=None):
        """
        Initialize CLI client.

        Args:
            db_session_factory: Database session factory
        """
        self.db_factory = db_session_factory

    async def create_run(
        self,
        data: EvaluationRunCreate,
    ) -> Dict[str, Any]:
        """
        Create a new evaluation run from CLI request.

        Args:
            data: Run creation data

        Returns:
            Run ID and metadata
        """
        from app.database import get_db
        from app.services.runs import RunService
        from app.services.models import ModelService

        db = next(get_db()) if self.db_factory is None else self.db_factory()

        try:
            # Find or create model
            model_service = ModelService(db)
            model = await model_service.get_or_create_by_checkpoint(
                checkpoint=data.model_id,
                model_type=self._detect_model_type(data.model_id),
            )

            # Create run using synchronous service method
            run_service = RunService(db)
            from app.schemas import RunCreate
            run_data = RunCreate(
                model_id=model.id,
                run_type=data.run_type,
                build_name=data.build_name,
                config=data.config,
                device=data.device,
                backend=data.backend,
                dtype=data.dtype,
                status="running",
            )
            run = run_service.create_run(run_data)

            # Broadcast run created via WebSocket
            await self._broadcast_run_status(run.id, "running")

            return {
                "run_id": run.id,
                "model_id": model.id,
                "status": "running",
            }

        finally:
            if self.db_factory is None:
                db.close()

    async def submit_metrics(
        self,
        data: EvaluationMetricsSubmit,
    ) -> Dict[str, Any]:
        """
        Submit metrics for a run.

        Args:
            data: Metrics submission data

        Returns:
            Number of metrics created
        """
        from app.database import get_db
        from app.services.metrics import MetricService

        db = next(get_db()) if self.db_factory is None else self.db_factory()

        try:
            metric_service = MetricService(db)

            # Convert and create metrics
            from app.schemas import MetricCreate
            metrics_data = [
                MetricCreate(
                    run_id=data.run_id,
                    category=m.category,
                    name=m.name,
                    display_name=m.display_name or self._format_display_name(m.name),
                    value_numeric=m.value,
                    unit=m.unit,
                )
                for m in data.metrics
            ]

            created = metric_service.create_metrics_bulk(metrics_data)

            # Broadcast metrics update via WebSocket
            await self._broadcast_metrics_update(
                data.run_id,
                [{"name": m.name, "value": m.value} for m in data.metrics],
            )

            return {
                "metrics_created": len(created),
                "run_id": data.run_id,
            }

        finally:
            if self.db_factory is None:
                db.close()

    async def complete_run(
        self,
        data: EvaluationComplete,
    ) -> Dict[str, Any]:
        """
        Mark a run as complete or failed.

        Args:
            data: Completion data

        Returns:
            Updated run status
        """
        from app.database import get_db
        from app.services.runs import RunService

        db = next(get_db()) if self.db_factory is None else self.db_factory()

        try:
            run_service = RunService(db)
            run = run_service.update_status(
                run_id=data.run_id,
                status=data.status,
                message=data.message,
            )

            # Broadcast completion via WebSocket
            await self._broadcast_run_status(
                data.run_id,
                data.status,
                message=data.message,
            )

            return {
                "run_id": data.run_id,
                "status": data.status,
                "completed_at": run.completed_at.isoformat() if run and run.completed_at else None,
            }

        finally:
            if self.db_factory is None:
                db.close()

    async def import_bulk(
        self,
        data: BulkEvaluationImport,
    ) -> Dict[str, Any]:
        """
        Import multiple evaluations in bulk.

        Args:
            data: Bulk import data

        Returns:
            Import summary
        """
        from app.database import get_db
        from app.services.runs import RunService
        from app.services.metrics import MetricService
        from app.services.models import ModelService

        db = next(get_db()) if self.db_factory is None else self.db_factory()

        try:
            run_service = RunService(db)
            metric_service = MetricService(db)
            model_service = ModelService(db)

            imported = 0
            skipped = 0
            failed = 0

            for entry in data.evaluations:
                try:
                    # Check for duplicates
                    if data.skip_duplicates:
                        existing = await run_service.get_by_build_name(entry.build_name)
                        if existing:
                            skipped += 1
                            continue

                    # Find or create model
                    model = await model_service.get_or_create_by_checkpoint(
                        checkpoint=entry.model_checkpoint,
                        model_type=self._detect_model_type(entry.model_checkpoint),
                    )

                    # Create run
                    run = await run_service.create(
                        model_id=model.id,
                        run_type=entry.run_type,
                        build_name=entry.build_name,
                        config=entry.config,
                        device=entry.device,
                        backend=entry.backend,
                        dtype=entry.dtype,
                        status=entry.status,
                        duration_seconds=entry.duration_seconds,
                    )

                    # Create metrics
                    if entry.metrics:
                        metrics_data = [
                            {
                                "run_id": run.id,
                                "category": m.category,
                                "name": m.name,
                                "display_name": m.display_name or self._format_display_name(m.name),
                                "value_numeric": m.value,
                                "unit": m.unit,
                            }
                            for m in entry.metrics
                        ]
                        await metric_service.create_batch(metrics_data)

                    imported += 1

                except Exception as e:
                    logger.error(f"Failed to import evaluation {entry.build_name}: {e}")
                    failed += 1

            return {
                "imported": imported,
                "skipped": skipped,
                "failed": failed,
                "total": len(data.evaluations),
            }

        finally:
            if self.db_factory is None:
                db.close()

    def _detect_model_type(self, checkpoint: str) -> str:
        """Detect model type from checkpoint path."""
        checkpoint_lower = checkpoint.lower()
        if any(kw in checkpoint_lower for kw in ["vision", "vlm", "clip"]):
            return "vlm"
        elif any(kw in checkpoint_lower for kw in ["embedding", "embed"]):
            return "embedding"
        return "llm"

    def _format_display_name(self, name: str) -> str:
        """Convert snake_case metric name to Title Case display name."""
        return name.replace("_", " ").title()

    async def _broadcast_run_status(
        self,
        run_id: str,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        """Broadcast run status update via WebSocket."""
        from app.websocket import manager, emit_run_status

        try:
            await emit_run_status(run_id, status, message)
        except Exception as e:
            logger.debug(f"Failed to broadcast run status: {e}")

    async def _broadcast_metrics_update(
        self,
        run_id: str,
        metrics: List[dict],
    ) -> None:
        """Broadcast metrics update via WebSocket."""
        from app.websocket import manager, emit_metric_update

        try:
            await emit_metric_update(run_id, metrics)
        except Exception as e:
            logger.debug(f"Failed to broadcast metrics update: {e}")


# ============================================================================
# AUTHENTICATION FOR CLI
# ============================================================================

def verify_cli_signature(payload: str, signature: str, secret: str) -> bool:
    """
    Verify CLI request signature.

    Args:
        payload: Request payload string
        signature: Signature from header
        secret: Shared secret

    Returns:
        True if signature is valid
    """
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def generate_cli_signature(payload: str, secret: str) -> str:
    """
    Generate signature for CLI request.

    Args:
        payload: Request payload string
        secret: Shared secret

    Returns:
        Signature string
    """
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
