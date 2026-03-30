"""
FastAPI application entry point.

Sets up the main application with:
- CORS middleware
- API routers
- WebSocket handlers
- Event handlers
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import async_init_db
from app.websocket import websocket_endpoint, manager
from app.api.v1 import (
    models_router,
    runs_router,
    metrics_router,
    import_router,
    health_router,
    auth_router,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up Lemonade Eval Dashboard...")

    # Initialize database (in development)
    if settings.debug:
        try:
            await async_init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Lemonade Eval Dashboard...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for Lemonade Eval Dashboard - Store, visualize, and compare LLM/VLM evaluation results",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
# Note: allow_credentials=True should NOT be used with allow_origins=["*"]
# We use specific origins from settings (defaults to localhost URLs in development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,  # Safe because cors_origins is restricted to specific URLs
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(auth_router, prefix=f"{settings.api_v1_prefix}/auth")
app.include_router(models_router, prefix=settings.api_v1_prefix)
app.include_router(runs_router, prefix=settings.api_v1_prefix)
app.include_router(metrics_router, prefix=settings.api_v1_prefix)
app.include_router(import_router, prefix=settings.api_v1_prefix)


# WebSocket endpoint
@app.websocket(f"{settings.ws_v1_prefix}/evaluations")
async def websocket_evaluations(websocket: WebSocket, run_id: str | None = None):
    """
    WebSocket endpoint for real-time evaluation updates.

    Query parameters:
    - `run_id`: Optional run ID to subscribe to specific updates
    """
    await websocket_endpoint(websocket, run_id)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "websocket": f"{settings.ws_v1_prefix}/evaluations",
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "prefix": settings.api_v1_prefix,
        "endpoints": {
            "health": f"{settings.api_v1_prefix}/health",
            "models": f"{settings.api_v1_prefix}/models",
            "runs": f"{settings.api_v1_prefix}/runs",
            "metrics": f"{settings.api_v1_prefix}/metrics",
            "import": f"{settings.api_v1_prefix}/import",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
