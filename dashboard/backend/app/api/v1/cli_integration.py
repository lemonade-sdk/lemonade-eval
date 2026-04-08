"""
API routes for CLI integration.

Endpoints:
- POST /api/v1/import/evaluation - Receive evaluation results from CLI
- POST /api/v1/import/bulk - Bulk import multiple evaluations
- POST /api/v1/import/yaml - Import evaluation from YAML data
- WebSocket /ws/v1/evaluation-progress - Real-time evaluation progress

Security:
- All import endpoints support optional HMAC-SHA256 signature verification
- Signature sent via X-CLI-Signature header
- Verification can be disabled via CLI_SIGNATURE_ENABLED=false config
"""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas import APIResponse
from app.config import settings
from app.integration.cli_client import (
    CLIClient,
    EvaluationRunCreate,
    EvaluationMetricsSubmit,
    EvaluationComplete,
    BulkEvaluationImport,
    ProgressUpdate,
    verify_cli_signature,
)
from app.integration.import_pipeline import EvaluationImporter
from app.websocket import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/import", tags=["CLI Integration"])


# ============================================================================
# SIGNATURE VERIFICATION HELPER
# ============================================================================

def validate_cli_signature(request_body: bytes, signature: Optional[str]) -> bool:
    """
    Validate CLI request signature.

    Args:
        request_body: Raw request body bytes
        signature: Signature from X-CLI-Signature header

    Returns:
        True if signature is valid or verification is disabled

    Raises:
        HTTPException: If signature is missing or invalid
    """
    # Skip verification if disabled (for development)
    if not settings.cli_signature_enabled:
        return True

    # Check if signature is provided
    if not signature:
        raise HTTPException(
            status_code=401,
            detail="Missing X-CLI-Signature header. Signature verification is enabled.",
        )

    # Verify signature
    try:
        is_valid = verify_cli_signature(
            payload=request_body.decode('utf-8'),
            signature=signature,
            secret=settings.cli_secret,
        )
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail="Invalid signature. Request authentication failed.",
            )
        return True
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Signature verification error: {str(e)}",
        )


async def get_request_body(request: Request) -> bytes:
    """Get raw request body for signature verification."""
    return await request.body()


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@router.post("/evaluation", response_model=APIResponse)
async def import_evaluation(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session),
):
    """
    Receive evaluation results from lemonade-eval CLI.

    This endpoint accepts evaluation data from the CLI and stores it in the database.

    **Security:**
    - Requires `X-CLI-Signature` header with HMAC-SHA256 signature
    - Signature computed over raw JSON body using shared secret
    - Can be disabled for development via CLI_SIGNATURE_ENABLED=false

    **Request Body:**
    - `model_id`: Model checkpoint or ID
    - `run_type`: Type of evaluation (benchmark, accuracy-mmlu, etc.)
    - `build_name`: Unique build identifier
    - `metrics`: List of metric objects with name, value, category, unit
    - `config`: Optional run configuration
    - `device`: Optional device type
    - `backend`: Optional backend runtime
    - `dtype`: Optional data type
    - `status`: Run status (running, completed, failed)
    - `duration_seconds`: Optional total duration
    - `started_at`: Optional start time (ISO format)
    - `completed_at`: Optional completion time (ISO format)

    **Example:**
    ```json
    {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "run_type": "benchmark",
        "build_name": "llama-3.2-1b-benchmark-20240101",
        "metrics": [
            {"name": "seconds_to_first_token", "value": 0.025, "unit": "seconds"},
            {"name": "token_generation_tokens_per_second", "value": 45.5, "unit": "tokens/s"}
        ],
        "config": {"iterations": 10},
        "device": "gpu",
        "backend": "ort",
        "status": "completed",
        "duration_seconds": 120.5
    }
    ```
    """
    # Get raw body for signature verification
    raw_body = await get_request_body(request)
    signature = request.headers.get("X-CLI-Signature")

    # Validate signature
    validate_cli_signature(raw_body, signature)

    # Parse JSON from raw body
    try:
        request_data = json.loads(raw_body.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {str(e)}",
        )

    try:
        cli_client = CLIClient()

        # Extract data
        run_data = {
            "model_id": request_data.get("model_id"),
            "run_type": request_data.get("run_type"),
            "build_name": request_data.get("build_name"),
            "config": request_data.get("config", {}),
            "device": request_data.get("device"),
            "backend": request_data.get("backend"),
            "dtype": request_data.get("dtype"),
            "started_at": request_data.get("started_at"),
        }

        # Validate required fields
        for field in ["model_id", "run_type", "build_name"]:
            if not run_data.get(field):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}",
                )

        # Create or get run
        run_result = await cli_client.create_run(EvaluationRunCreate(**run_data))
        run_id = run_result["run_id"]

        # Submit metrics if provided
        metrics = request_data.get("metrics", [])
        if metrics:
            metrics_data = [
                {
                    "name": m.get("name"),
                    "value": m.get("value"),
                    "category": m.get("category", "performance"),
                    "unit": m.get("unit"),
                    "display_name": m.get("display_name"),
                }
                for m in metrics
            ]
            await cli_client.submit_metrics(
                EvaluationMetricsSubmit(
                    run_id=run_id,
                    metrics=metrics_data,
                )
            )

        # Complete run if status is provided
        status = request_data.get("status")
        if status in ["completed", "failed"]:
            await cli_client.complete_run(
                EvaluationComplete(
                    run_id=run_id,
                    status=status,
                    message=request_data.get("message"),
                    duration_seconds=request_data.get("duration_seconds"),
                    completed_at=request_data.get("completed_at"),
                )
            )

        return APIResponse(
            success=True,
            data={
                "run_id": run_id,
                "status": status or "running",
                "metrics_imported": len(metrics),
            },
            message="Evaluation imported successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import evaluation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import evaluation: {str(e)}",
        )


@router.post("/bulk", response_model=APIResponse)
async def import_bulk_evaluations(
    request: Request,
    db: Session = Depends(get_db_session),
):
    """
    Bulk import multiple evaluations.

    This endpoint is optimized for importing multiple evaluations at once,
    such as when migrating historical data or syncing from another system.

    **Security:**
    - Requires `X-CLI-Signature` header with HMAC-SHA256 signature
    - Signature computed over raw JSON body using shared secret
    - Can be disabled for development via CLI_SIGNATURE_ENABLED=false

    **Request Body:**
    - `evaluations`: List of evaluation entries
    - `skip_duplicates`: Whether to skip existing runs (default: true)

    **Evaluation Entry:**
    - `model_checkpoint`: Model checkpoint path
    - `run_type`: Evaluation type
    - `build_name`: Unique build identifier
    - `metrics`: List of metrics
    - `config`: Optional run configuration
    - `device`: Optional device type
    - `backend`: Optional backend runtime
    - `dtype`: Optional data type
    - `status`: Run status (default: completed)
    - `duration_seconds`: Optional duration
    - `started_at`: Optional start time
    - `completed_at`: Optional completion time
    """
    # Get raw body for signature verification
    raw_body = await get_request_body(request)
    signature = request.headers.get("X-CLI-Signature")

    # Validate signature
    validate_cli_signature(raw_body, signature)

    # Parse JSON from raw body
    try:
        request_data = json.loads(raw_body.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {str(e)}",
        )

    try:
        # Create BulkEvaluationImport from parsed data
        bulk_data = BulkEvaluationImport(**request_data)
        cli_client = CLIClient()

        result = await cli_client.import_bulk(bulk_data)

        return APIResponse(
            success=True,
            data=result,
            message=f"Imported {result['imported']} evaluations, skipped {result['skipped']}, failed {result['failed']}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import bulk evaluations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import evaluations: {str(e)}",
        )


@router.post("/yaml", response_model=APIResponse)
async def import_yaml_data(
    request: Request,
    db: Session = Depends(get_db_session),
):
    """
    Import evaluation from YAML data format.

    Accepts raw YAML data (as a dict) and imports it directly.
    Useful for migrating from file-based storage.

    **Security:**
    - Requires `X-CLI-Signature` header with HMAC-SHA256 signature
    - Signature computed over raw JSON body using shared secret
    - Can be disabled for development via CLI_SIGNATURE_ENABLED=false

    **Request Body:**
    - `yaml_data`: Parsed YAML data as dict
    - `build_name`: Build name (optional, extracted from path if not provided)
    - `skip_duplicates`: Whether to skip existing runs (default: true)
    """
    # Get raw body for signature verification
    raw_body = await get_request_body(request)
    signature = request.headers.get("X-CLI-Signature")

    # Validate signature
    validate_cli_signature(raw_body, signature)

    # Parse JSON from raw body
    try:
        request_data = json.loads(raw_body.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {str(e)}",
        )

    try:
        yaml_data = request_data.get("yaml_data")
        build_name = request_data.get("build_name")
        skip_duplicates = request_data.get("skip_duplicates", True)

        if not yaml_data:
            raise HTTPException(
                status_code=400,
                detail="yaml_data is required",
            )

        # Extract build name from YAML if not provided
        if not build_name:
            build_name = yaml_data.get("build_name")

        if not build_name:
            raise HTTPException(
                status_code=400,
                detail="build_name is required (either in request or YAML data)",
            )

        importer = EvaluationImporter()
        result = await importer.import_from_data(
            yaml_data=yaml_data,
            build_name=build_name,
            skip_duplicates=skip_duplicates,
        )

        return APIResponse(
            success=result.get("success", False),
            data=result,
            message=result.get("message", "Import completed"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import YAML data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import YAML data: {str(e)}",
        )


@router.get("/status/{run_id}", response_model=APIResponse)
async def get_import_status(
    run_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get the status of an imported evaluation run.

    **Path Parameters:**
    - `run_id`: The run ID to check

    **Returns:**
    - Run status and details
    """
    from sqlalchemy import select
    from app.models import Run

    try:
        run = db.execute(
            select(Run).where(Run.id == run_id)
        ).scalar_one_or_none()

        if not run:
            raise HTTPException(
                status_code=404,
                detail=f"Run not found: {run_id}",
            )

        return APIResponse(
            success=True,
            data={
                "run_id": run.id,
                "build_name": run.build_name,
                "status": run.status,
                "run_type": run.run_type,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "duration_seconds": float(run.duration_seconds) if run.duration_seconds else None,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get import status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}",
        )


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@router.websocket("/progress")
async def evaluation_progress_websocket(
    websocket: WebSocket,
    run_id: Optional[str] = None,
):
    """
    WebSocket endpoint for real-time evaluation progress updates.

    Connect to receive progress updates during evaluation execution.

    **Query Parameters:**
    - `run_id`: Optional run ID to subscribe to specific updates

    **Client Messages:**
    - `{"type": "subscribe", "run_id": "..."}` - Subscribe to specific run
    - `{"type": "unsubscribe"}` - Unsubscribe from current run
    - `{"type": "ping"}` - Health check

    **Server Messages:**
    - `{"type": "progress", "run_id": "...", "progress": 50.0, "message": "..."}`
    - `{"type": "run_status", "run_id": "...", "status": "running"}`
    - `type": "subscribed", "run_id": "..."}` - Subscription confirmation

    **Example:**
    ```python
    import asyncio
    import websockets

    async def watch_progress(run_id):
        uri = f"ws://localhost:8000/ws/v1/evaluation-progress?run_id={run_id}"
        async with websockets.connect(uri) as ws:
            while True:
                msg = await ws.recv()
                print(f"Progress: {msg}")
    ```
    """
    await manager.connect(websocket, run_id)

    try:
        while True:
            try:
                data = await websocket.receive_text()
                import json

                message = json.loads(data)
                await handle_progress_message(websocket, message)

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected: run_id={run_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


async def handle_progress_message(
    websocket: WebSocket,
    message: Dict[str, Any],
):
    """
    Handle incoming WebSocket messages.

    Args:
        websocket: Client connection
        message: Received message
    """
    msg_type = message.get("type")

    if msg_type == "subscribe":
        run_id = message.get("run_id")
        if run_id:
            manager.active_connections[websocket] = run_id
            if run_id not in manager.run_subscribers:
                manager.run_subscribers[run_id] = set()
            manager.run_subscribers[run_id].add(websocket)
            await websocket.send_json({
                "type": "subscribed",
                "run_id": run_id,
            })

    elif msg_type == "unsubscribe":
        run_id = manager.active_connections.get(websocket)
        if run_id and run_id in manager.run_subscribers:
            manager.run_subscribers[run_id].discard(websocket)
        manager.active_connections[websocket] = None
        await websocket.send_json({"type": "unsubscribed"})

    elif msg_type == "ping":
        await websocket.send_json({"type": "pong"})

    else:
        await websocket.send_json({
            "error": f"Unknown message type: {msg_type}",
        })
