"""
API routes for YAML import operations.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas import APIResponse, ImportRequest, ImportJobStatus
from app.services.import_service import ImportService

router = APIRouter(prefix="/import", tags=["Import"])

# In-memory job status storage (use Redis in production)
import_jobs: dict[str, dict] = {}


def process_import_task(
    job_id: str,
    cache_dir: str,
    skip_duplicates: bool,
    db=None,
):
    """
    Background task to process YAML import.

    Args:
        job_id: Unique job identifier
        cache_dir: Cache directory to import from
        skip_duplicates: Whether to skip existing runs
        db: Database session (provided by caller for test mode)
    """
    try:
        import_jobs[job_id]["status"] = "running"
        import_jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

        # Use provided db session or create new session
        # In test mode, db will be provided directly
        # In production, we create a new session
        if db is not None:
            # Use the provided session (test mode)
            service = ImportService(db)
        else:
            # Production mode - create new session
            from app.database import get_db
            db_session = next(get_db())
            service = ImportService(db_session)

        try:
            result = service.import_directory(
                cache_dir=cache_dir,
                skip_duplicates=skip_duplicates,
            )

            import_jobs[job_id]["status"] = "completed"
            import_jobs[job_id]["result"] = result
            import_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            import_jobs[job_id]["total_files"] = result.get("total_files", 0)
            import_jobs[job_id]["imported_runs"] = result.get("imported_runs", 0)
            import_jobs[job_id]["skipped_duplicates"] = result.get("skipped_duplicates", 0)
            import_jobs[job_id]["errors"] = result.get("errors", [])
        finally:
            if db is not None:
                # Only close if we created it ourselves (production mode)
                pass  # In test mode, don't close - the test fixture handles it
    except Exception as e:
        import_jobs[job_id]["status"] = "failed"
        import_jobs[job_id]["error"] = str(e)
        import_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/yaml", response_model=APIResponse)
async def import_yaml(
    request: ImportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session),
):
    """
    Import YAML files from a cache directory.

    This is an async operation. Use the returned job_id to check status.

    **Parameters:**
    - `cache_dir`: Path to lemonade cache directory
    - `skip_duplicates`: Skip runs that already exist (default: true)
    - `dry_run`: Only scan, don't import (default: false)
    """
    job_id = str(uuid.uuid4())

    # Initialize job status
    import_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "cache_dir": request.cache_dir,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if request.dry_run:
        # Do dry run synchronously (no database writes)
        service = ImportService(db)
        result = service.import_directory(
            cache_dir=request.cache_dir,
            skip_duplicates=request.skip_duplicates,
            dry_run=True,
        )
        import_jobs[job_id]["status"] = "completed"
        import_jobs[job_id]["result"] = result
        import_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        import_jobs[job_id]["total_files"] = result.get("total_files", 0)
        import_jobs[job_id]["discovered_files"] = result.get("discovered_files", [])
    elif os.environ.get("TESTING", "false").lower() == "true":
        # Run synchronously in test mode to use overridden database session
        service = ImportService(db)
        result = service.import_directory(
            cache_dir=request.cache_dir,
            skip_duplicates=request.skip_duplicates,
        )
        import_jobs[job_id]["status"] = "completed"
        import_jobs[job_id]["result"] = result
        import_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        import_jobs[job_id]["total_files"] = result.get("total_files", 0)
        import_jobs[job_id]["imported_runs"] = result.get("imported_runs", 0)
        import_jobs[job_id]["skipped_duplicates"] = result.get("skipped_duplicates", 0)
        import_jobs[job_id]["errors"] = result.get("errors", [])
    else:
        # Start background task for async execution in production
        background_tasks.add_task(
            process_import_task,
            job_id,
            request.cache_dir,
            request.skip_duplicates,
            None,  # No db provided - production mode creates its own session
        )

    return APIResponse(
        success=True,
        data={"job_id": job_id},
    )


@router.get("/status/{job_id}", response_model=APIResponse)
async def get_import_status(
    job_id: str,
    db: Session = Depends(get_db_session),
):
    """
    Get status of an import job.
    """
    if job_id not in import_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = import_jobs[job_id]

    response_data = {
        "job_id": job["job_id"],
        "status": job["status"],
        "total_files": job.get("total_files", 0),
        "processed_files": job.get("processed_files", 0),
        "imported_runs": job.get("imported_runs", 0),
        "skipped_duplicates": job.get("skipped_duplicates", 0),
    }

    if job["status"] == "failed":
        response_data["error"] = job.get("error", "Unknown error")

    if job["status"] == "completed" and "result" in job:
        response_data["errors"] = job.get("errors", [])

    return APIResponse(
        success=True,
        data=response_data,
    )


@router.post("/scan", response_model=APIResponse)
async def scan_cache_directory(
    cache_dir: str = Query(..., description="Cache directory to scan"),
    db: Session = Depends(get_db_session),
):
    """
    Scan a cache directory and report found YAML files.

    Does not import, just discovers files.
    """
    service = ImportService(db)
    discovered = service.scan_cache_dir(cache_dir)

    return APIResponse(
        success=True,
        data={
            "cache_dir": cache_dir,
            "files_found": len(discovered),
            "files": discovered,
        },
    )


@router.get("/jobs", response_model=APIResponse)
async def list_import_jobs(
    db: Session = Depends(get_db_session),
):
    """
    List all import jobs.
    """
    jobs = list(import_jobs.values())
    return APIResponse(
        success=True,
        data=jobs,
    )
