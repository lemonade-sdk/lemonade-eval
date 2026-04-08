"""
Monitoring module for the Dashboard API.

Provides:
- Prometheus metrics
- Health check endpoints
- Performance monitoring
"""

from app.monitoring.metrics import (
    setup_monitoring,
    MonitoringMiddleware,
    http_requests_total,
    http_request_duration,
    db_connections_active,
    websocket_connections,
    import_jobs_total,
    evaluation_runs_total,
)

__all__ = [
    "setup_monitoring",
    "MonitoringMiddleware",
    "http_requests_total",
    "http_request_duration",
    "db_connections_active",
    "websocket_connections",
    "import_jobs_total",
    "evaluation_runs_total",
]
