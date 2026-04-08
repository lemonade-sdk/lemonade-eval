"""
API routes version 1.
"""

from app.api.v1.models import router as models_router
from app.api.v1.runs import router as runs_router
from app.api.v1.metrics import router as metrics_router
from app.api.v1.import_routes import router as import_router
from app.api.v1.health import router as health_router
from app.api.v1.auth import router as auth_router
from app.api.v1.cli_integration import router as cli_router

__all__ = [
    "models_router",
    "runs_router",
    "metrics_router",
    "import_router",
    "health_router",
    "auth_router",
    "cli_router",
]
