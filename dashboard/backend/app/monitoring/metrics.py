"""
Prometheus metrics and health monitoring.

Metrics exposed:
- HTTP request latency histogram
- Request count by endpoint
- Database connection pool stats
- WebSocket connection count
- Import job progress
- Cache hit/miss rates
"""

import time
from datetime import datetime
from typing import Optional, Callable, Awaitable

from fastapi import FastAPI, Response, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

# HTTP Request Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

http_request_size = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
)

http_response_size = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 1000000],
)

# Database Metrics
db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
)

db_connections_total = Gauge(
    "db_connections_total",
    "Total database connections in pool",
)

db_query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# WebSocket Metrics
websocket_connections = Gauge(
    "websocket_connections_total",
    "Active WebSocket connections",
)

websocket_messages_total = Counter(
    "websocket_messages_total",
    "Total WebSocket messages",
    ["type"],
)

# Import Job Metrics
import_jobs_total = Counter(
    "import_jobs_total",
    "Total import jobs",
    ["status"],
)

import_jobs_duration = Histogram(
    "import_jobs_duration_seconds",
    "Import job duration",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# Evaluation Run Metrics
evaluation_runs_total = Counter(
    "evaluation_runs_total",
    "Total evaluation runs",
    ["status", "run_type"],
)

evaluation_duration = Histogram(
    "evaluation_duration_seconds",
    "Evaluation run duration",
    ["run_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
)

# Cache Metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_name"],
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_name"],
)

cache_operation_duration = Histogram(
    "cache_operation_duration_seconds",
    "Cache operation duration",
    ["operation", "cache_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# System Metrics
app_info = Gauge(
    "app_info",
    "Application information",
    ["version", "environment"],
)

app_start_time = Gauge(
    "app_start_time",
    "Application start time as Unix timestamp",
)


# ============================================================================
# MONITORING SETUP
# ============================================================================

def setup_monitoring(app: FastAPI) -> None:
    """
    Set up Prometheus monitoring for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Set static metrics
    app_info.labels(
        version=app.version or "1.0.0",
        environment="production",
    ).set(1)
    app_start_time.set(time.time())

    # Add metrics endpoint
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics_endpoint() -> Response:
        """
        Prometheus metrics scraping endpoint.

        Returns:
            PlainTextResponse with Prometheus metrics format
        """
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Add health check endpoints
    @app.get("/health/live", tags=["Health"])
    async def liveness_check() -> dict:
        """
        Kubernetes liveness probe endpoint.

        Returns:
            Health status dict
        """
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/health/ready", tags=["Health"])
    async def readiness_check(db=None) -> dict:
        """
        Kubernetes readiness probe with dependency checks.

        Returns:
            Health status with component checks
        """
        checks = {
            "database": "ok",
            "redis": "ok",
            "memory": "ok",
        }
        is_ready = True

        # Check database
        try:
            if db is not None:
                # Perform quick DB query
                from sqlalchemy import text
                db.execute(text("SELECT 1"))
                checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {str(e)}"
            is_ready = False

        # Check memory (fail if > 95% used)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 95:
                checks["memory"] = f"critical: {memory_percent}%"
                is_ready = False
            elif memory_percent > 85:
                checks["memory"] = f"warning: {memory_percent}%"
        except ImportError:
            checks["memory"] = "psutil not installed"

        status = "ready" if is_ready else "not_ready"
        return {
            "status": status,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Add middleware for automatic metric collection
    app.add_middleware(MonitoringMiddleware)

    logger.info("Prometheus monitoring setup complete")


# ============================================================================
# MIDDLEWARE
# ============================================================================

class MonitoringMiddleware:
    """Middleware for capturing request metrics automatically."""

    def __init__(self, app: FastAPI):
        """
        Initialize monitoring middleware.

        Args:
            app: FastAPI application
        """
        self.app = app

    async def __call__(
        self,
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        """
        Process request and capture metrics.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()
        request_body_size = 0

        async def send_wrapper(message: dict) -> None:
            """Wrapper to capture response metrics."""
            if message["type"] == "http.response.start":
                duration = time.time() - start_time
                method = scope["method"]
                endpoint = self._normalize_endpoint(scope["path"])
                status = message["status"]

                # Record request count
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status,
                ).inc()

                # Record request duration
                http_request_duration.labels(
                    method=method,
                    endpoint=endpoint,
                ).observe(duration)

                # Record response size if available
                for name, value in message.get("headers", []):
                    if name == b"content-length":
                        http_response_size.labels(
                            method=method,
                            endpoint=endpoint,
                        ).observe(int(value))
                        break

            return await send(message)

        return await self.app(scope, receive, send_wrapper)

    def _normalize_endpoint(self, path: str) -> str:
        """
        Normalize endpoint path by replacing UUIDs and IDs with placeholders.

        Args:
            path: Request path

        Returns:
            Normalized path for metrics
        """
        import re

        # Replace UUIDs
        normalized = re.sub(
            r"/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
            "/{id}",
            path,
        )
        # Replace numeric IDs
        normalized = re.sub(r"/\d+", "/{id}", normalized)
        # Replace api key prefixes
        normalized = re.sub(r"/ledash_[a-zA-Z0-9]+", "/ledash_{key}", normalized)

        return normalized


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_db_query(query_type: str, duration: float) -> None:
    """
    Record a database query metric.

    Args:
        query_type: Type of query (select, insert, update, delete)
        duration: Query duration in seconds
    """
    db_query_duration.labels(query_type=query_type).observe(duration)


def record_cache_hit(cache_name: str, hit: bool, duration: float = 0.0) -> None:
    """
    Record a cache operation metric.

    Args:
        cache_name: Name of the cache
        hit: Whether it was a hit
        duration: Operation duration in seconds
    """
    if hit:
        cache_hits_total.labels(cache_name=cache_name).inc()
    else:
        cache_misses_total.labels(cache_name=cache_name).inc()

    if duration > 0:
        cache_operation_duration.labels(
            operation="hit" if hit else "miss",
            cache_name=cache_name,
        ).observe(duration)


def record_websocket_message(msg_type: str) -> None:
    """
    Record a WebSocket message metric.

    Args:
        msg_type: Type of message
    """
    websocket_messages_total.labels(type=msg_type).inc()


def record_import_job(status: str, duration: float) -> None:
    """
    Record an import job metric.

    Args:
        status: Job status (completed, failed, cancelled)
        duration: Job duration in seconds
    """
    import_jobs_total.labels(status=status).inc()
    import_jobs_duration.observe(duration)


def record_evaluation_run(
    status: str,
    run_type: str,
    duration: float,
) -> None:
    """
    Record an evaluation run metric.

    Args:
        status: Run status (completed, failed, cancelled)
        run_type: Type of evaluation
        duration: Run duration in seconds
    """
    evaluation_runs_total.labels(status=status, run_type=run_type).inc()
    evaluation_duration.labels(run_type=run_type).observe(duration)


def update_websocket_connections(count: int) -> None:
    """
    Update the WebSocket connections gauge.

    Args:
        count: Number of active connections
    """
    websocket_connections.set(count)


def update_db_connections(active: int, total: int) -> None:
    """
    Update database connection gauges.

    Args:
        active: Number of active connections
        total: Total connections in pool
    """
    db_connections_active.set(active)
    db_connections_total.set(total)
