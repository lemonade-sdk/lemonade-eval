"""
FastAPI application entry point.

Sets up the main application with:
- CORS middleware
- Rate limiting middleware
- API routers
- WebSocket handlers
- Event handlers
- Prometheus monitoring
- Redis caching
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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
    cli_router,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up Lemonade Eval Dashboard...")

    # Initialize Redis connections
    await init_redis_connections(app)

    # Initialize database (in development)
    if settings.debug:
        try:
            await async_init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}")

    # Setup monitoring
    await init_monitoring(app)

    yield

    # Shutdown
    logger.info("Shutting down Lemonade Eval Dashboard...")
    await cleanup_redis_connections(app)
    logger.info("Shutdown complete")


async def init_redis_connections(app: FastAPI) -> None:
    """Initialize Redis connections for caching and rate limiting."""
    try:
        # Initialize rate limiter
        from app.middleware.rate_limiter import init_rate_limiter
        limiter = init_rate_limiter(settings.redis_url)
        app.state.rate_limiter = limiter
        logger.info(f"Rate limiter initialized: {settings.redis_url}")

        # Initialize cache manager
        from app.cache.cache_manager import init_cache_manager
        cache = init_cache_manager(settings.redis_url)
        app.state.cache_manager = cache
        logger.info(f"Cache manager initialized: {settings.redis_url}")

    except Exception as e:
        logger.warning(f"Redis initialization skipped: {e}")
        app.state.rate_limiter = None
        app.state.cache_manager = None


async def cleanup_redis_connections(app: FastAPI) -> None:
    """Cleanup Redis connections."""
    try:
        if hasattr(app.state, "cache_manager") and app.state.cache_manager:
            app.state.cache_manager.disconnect()
            logger.info("Cache manager disconnected")
    except Exception as e:
        logger.warning(f"Error during Redis cleanup: {e}")


async def init_monitoring(app: FastAPI) -> None:
    """Initialize Prometheus monitoring."""
    try:
        from app.monitoring.metrics import setup_monitoring
        setup_monitoring(app)
        logger.info("Prometheus monitoring initialized")
    except Exception as e:
        logger.warning(f"Monitoring initialization skipped: {e}")


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

# Add rate limiting middleware (only in production mode)
if settings.rate_limit_enabled and not settings.debug:
    try:
        from app.middleware.rate_limiter import RateLimitMiddleware, get_rate_limiter
        limiter = get_rate_limiter()
        if limiter:
            app.add_middleware(RateLimitMiddleware, limiter=limiter)
            logger.info("Rate limiting middleware added")
        else:
            logger.warning("Rate limiter not available, middleware not added")
    except Exception as e:
        logger.warning(f"Failed to add rate limiting middleware: {e}")


# Include API routers
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(auth_router, prefix=f"{settings.api_v1_prefix}/auth")
app.include_router(models_router, prefix=settings.api_v1_prefix)
app.include_router(runs_router, prefix=settings.api_v1_prefix)
app.include_router(metrics_router, prefix=settings.api_v1_prefix)
app.include_router(import_router, prefix=settings.api_v1_prefix)
app.include_router(cli_router, prefix=settings.api_v1_prefix)


# WebSocket endpoint for evaluations
@app.websocket(f"{settings.ws_v1_prefix}/evaluations")
async def websocket_evaluations(websocket: WebSocket, run_id: str | None = None):
    """
    WebSocket endpoint for real-time evaluation updates.

    Query parameters:
    - `run_id`: Optional run ID to subscribe to specific updates
    """
    await websocket_endpoint(websocket, run_id)


# WebSocket endpoint for CLI progress reporting
@app.websocket(f"{settings.ws_v1_prefix}/evaluation-progress")
async def websocket_evaluation_progress(
    websocket: WebSocket,
    run_id: str | None = None,
):
    """
    WebSocket endpoint for CLI progress reporting.

    Query parameters:
    - `run_id`: Optional run ID to subscribe to specific updates
    """
    # Import the handler from cli_integration
    from app.api.v1.cli_integration import evaluation_progress_websocket
    await evaluation_progress_websocket(websocket, run_id)


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
        "metrics": "/metrics",
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
            "cli": f"{settings.api_v1_prefix}/import/evaluation",
        },
        "websocket": {
            "evaluations": f"{settings.ws_v1_prefix}/evaluations",
            "progress": f"{settings.ws_v1_prefix}/evaluation-progress",
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
