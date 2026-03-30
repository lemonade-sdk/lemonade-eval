"""
Health check endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import APIResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
@router.get("/")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.

    Returns:
        Health status with database connectivity
    """
    try:
        # Check database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "version": "1.0.0",
    }


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check endpoint.

    Returns:
        Readiness status
    """
    try:
        db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        return {"ready": False}
