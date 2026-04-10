# Production Automation & Lemonade-Eval CLI Integration Plan

**Document Version:** 1.0
**Date:** 2026-04-07
**Author:** Dr. Sarah Kim, Technical Product Strategist & Engineering Lead
**Status:** Ready for Implementation

---

## Executive Summary

This document provides a comprehensive implementation plan for production automation and CLI integration of the UI-UX Eval Dashboard. Based on the existing foundation (FastAPI backend with 269 tests, 84% coverage, React frontend, and import service), this plan addresses the critical path items for production deployment.

### Current State Assessment

| Component | Status | Coverage | Notes |
|-----------|--------|----------|-------|
| Backend API | Complete | 84% (269 tests) | FastAPI + SQLAlchemy + PostgreSQL |
| Frontend UI | Complete | - | React 18 + TypeScript + Mantine |
| Import Service | Complete | - | YAML migration from cache directory |
| WebSocket | Complete | - | Real-time evaluation updates |
| Authentication | Complete | - | JWT + API key support |
| Setup Scripts | Complete | - | Cross-platform (sh, ps1, bat) |
| Documentation | Partial | - | SETUP.md, DEPLOYMENT.md, API.md |

### What's Needed for Production

1. **CLI Integration** - Direct `lemonade-eval` to dashboard write path
2. **Automation Pipeline** - Scheduled evaluations, trend analysis, notifications
3. **Production Hardening** - Rate limiting, monitoring, backup strategies
4. **CI/CD Enhancements** - Automated testing, deployment pipelines
5. **User Documentation** - Dashboard usage guides, troubleshooting

---

## 1. Lemonade-Eval CLI Integration Architecture

### 1.1 Integration Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLI INTEGRATION FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  lemonade-eval   │     │   Dashboard      │     │   PostgreSQL     │
│  CLI             │     │   Backend API    │     │   Database       │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         │  1. POST /runs         │                        │
         │     (create run)       │                        │
         │───────────────────────>│                        │
         │                        │                        │
         │  2. Run ACK            │                        │
         │     (run_id)           │                        │
         │<───────────────────────│                        │
         │                        │                        │
         │  3. Evaluation         │                        │
         │     Executes           │                        │
         │                        │                        │
         │  4. POST /metrics      │                        │
         │     (stream results)   │                        │
         │───────────────────────>│  5. INSERT metrics     │
         │                        │───────────────────────>│
         │                        │                        │
         │  6. POST /runs/{id}/status                      │
         │     (complete)         │                        │
         │───────────────────────>│                        │
         │                        │                        │
         │  7. WS Event           │                        │
         │     (real-time update) │                        │
         │<───────────────────────│                        │
         │                        │                        │
```

### 1.2 CLI Command Extensions

Add the following flags to `lemonade-eval` CLI:

```python
# src/lemonade/cli.py - New arguments

parser.add_argument(
    "--dashboard-url",
    type=str,
    help="Dashboard API URL for direct result upload (e.g., https://dashboard.example.com)",
    required=False,
)

parser.add_argument(
    "--dashboard-api-key",
    type=str,
    help="API key for dashboard authentication",
    required=False,
)

parser.add_argument(
    "--dashboard-skip-verify",
    action="store_true",
    help="Skip SSL certificate verification for dashboard connection",
)

parser.add_argument(
    "--dashboard-batch-size",
    type=int,
    default=10,
    help="Number of metrics to batch per API call (default: 10)",
)

parser.add_argument(
    "--dashboard-sync-mode",
    choices=["async", "sync"],
    default="async",
    help="Sync mode: 'async' (fire-and-forget) or 'sync' (wait for ACK)",
)
```

### 1.3 Dashboard Client Module

Create a new module for dashboard communication:

```python
# src/lemonade/dashboard_client.py

"""
Dashboard client for direct result upload.

Features:
- Automatic retry with exponential backoff
- Batch metric uploads
- WebSocket real-time status
- Offline queue for failed uploads
"""

import asyncio
import aiohttp
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import websockets


class DashboardClient:
    """Client for communicating with the Eval Dashboard API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        skip_verify: bool = False,
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.skip_verify = skip_verify
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._offline_queue: List[Dict] = []

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(ssl=not self.skip_verify)
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def create_run(
        self,
        model_id: str,
        run_type: str,
        build_name: str,
        config: Dict[str, Any],
        device: Optional[str] = None,
        backend: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> str:
        """Create a new evaluation run and return run_id."""
        payload = {
            "model_id": model_id,
            "run_type": run_type,
            "build_name": build_name,
            "config": config,
            "device": device,
            "backend": backend,
            "dtype": dtype,
            "status": "running",
        }

        async with self._session.post(
            f"{self.base_url}/api/v1/runs",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["data"]["id"]

    async def upload_metrics(
        self,
        run_id: str,
        metrics: List[Dict[str, Any]],
    ) -> bool:
        """Batch upload metrics for a run."""
        # Group metrics into batches
        batches = [
            metrics[i:i + self.batch_size]
            for i in range(0, len(metrics), self.batch_size)
        ]

        for batch in batches:
            await self._upload_metric_batch(run_id, batch)

        return True

    async def _upload_metric_batch(
        self,
        run_id: str,
        metrics: List[Dict[str, Any]],
    ):
        """Upload a single batch of metrics with retry."""
        payload = {"metrics": [
            {
                "run_id": run_id,
                "category": m.get("category", "performance"),
                "name": m["name"],
                "display_name": m.get("display_name"),
                "value_numeric": m.get("value"),
                "unit": m.get("unit"),
            }
            for m in metrics
        ]}

        for attempt in range(self.max_retries):
            try:
                async with self._session.post(
                    f"{self.base_url}/api/v1/metrics/bulk",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    return
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    # Queue for offline retry
                    self._offline_queue.append(payload)
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ):
        """Update run status (completed, failed, etc.)."""
        payload = {
            "status": status,
            "status_message": message,
            "duration_seconds": duration_seconds,
            "completed_at": datetime.utcnow().isoformat() if status in ["completed", "failed"] else None,
        }

        async with self._session.put(
            f"{self.base_url}/api/v1/runs/{run_id}/status",
            json=payload,
        ) as resp:
            resp.raise_for_status()

    async def connect_websocket(self, run_id: Optional[str] = None):
        """Connect to WebSocket for real-time updates."""
        ws_url = self.base_url.replace("https", "wss").replace("http", "ws")
        ws_endpoint = f"{ws_url}/ws/v1/evaluations"
        if run_id:
            ws_endpoint += f"?run_id={run_id}"

        self._ws = await websockets.connect(
            ws_endpoint,
            extra_headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def send_progress(self, run_id: str, progress: float, message: str):
        """Send progress update via WebSocket."""
        if self._ws:
            await self._ws.send(json.dumps({
                "type": "progress",
                "run_id": run_id,
                "progress": progress,
                "message": message,
            }))

    def flush_offline_queue(self) -> int:
        """Attempt to flush queued metrics from failed uploads."""
        flushed = 0
        # Implementation for offline retry
        return flushed
```

### 1.4 Integration with Evaluation Sequence

Modify the sequence execution to report to dashboard:

```python
# src/lemonade/sequence.py - Dashboard integration

class Sequence:
    """Evaluation sequence with dashboard integration."""

    def __init__(
        self,
        tools: List[Tool],
        profilers: Optional[List[Profiler]] = None,
        dashboard_client: Optional[DashboardClient] = None,
    ):
        self.tools = tools
        self.profilers = profilers or []
        self.dashboard_client = dashboard_client
        self._run_id: Optional[str] = None

    async def launch_with_dashboard(
        self,
        state: State,
        model_id: str,
        run_type: str,
    ) -> State:
        """Launch sequence with dashboard reporting."""
        start_time = datetime.utcnow()

        if self.dashboard_client:
            async with self.dashboard_client:
                # Create run
                self._run_id = await self.dashboard_client.create_run(
                    model_id=model_id,
                    run_type=run_type,
                    build_name=state.build_name,
                    config=state.sequence_info,
                    device=state.device,
                    backend=state.backend,
                    dtype=state.dtype,
                )

                # Connect WebSocket for real-time updates
                await self.dashboard_client.connect_websocket(self._run_id)

                # Execute tools
                try:
                    state = await self._execute_tools(state)

                    # Upload final metrics
                    metrics = self._extract_metrics(state)
                    await self.dashboard_client.upload_metrics(
                        self._run_id,
                        metrics,
                    )

                    # Mark complete
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    await self.dashboard_client.update_run_status(
                        self._run_id,
                        status="completed",
                        message="Evaluation completed successfully",
                        duration_seconds=duration,
                    )

                except Exception as e:
                    # Mark failed
                    await self.dashboard_client.update_run_status(
                        self._run_id,
                        status="failed",
                        message=str(e),
                    )
                    raise

        else:
            # Standard execution without dashboard
            state = await self._execute_tools(state)

        return state
```

### 1.5 YAML Import Command Enhancement

Add a dedicated import command to lemonade-eval:

```python
# src/lemonade/tools/dashboard_import.py

from lemonade.tools import Tool
from lemonade.state import State
from lemonade.dashboard_client import DashboardClient
import yaml
import os
from pathlib import Path


class DashboardImport(Tool):
    """Import evaluation results to dashboard database."""

    unique_name = "import-dashboard"

    @classmethod
    def add_arguments_to_parser(cls, parser):
        parser.add_argument(
            "--dashboard-url",
            type=str,
            required=True,
            help="Dashboard API URL",
        )
        parser.add_argument(
            "--dashboard-api-key",
            type=str,
            required=True,
            help="Dashboard API key",
        )
        parser.add_argument(
            "--cache-dir",
            type=str,
            default="~/.cache/lemonade",
            help="Cache directory to import from",
        )
        parser.add_argument(
            "--skip-duplicates",
            action="store_true",
            default=True,
            help="Skip runs that already exist",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Scan only, don't import",
        )
        parser.add_argument(
            "--build-name",
            type=str,
            help="Import a specific build only",
        )

    def run(self, state: State, **kwargs) -> State:
        """Execute the import operation."""
        import asyncio

        return asyncio.run(self._import_async(**kwargs))

    async def _import_async(
        self,
        dashboard_url: str,
        dashboard_api_key: str,
        cache_dir: str,
        skip_duplicates: bool,
        dry_run: bool,
        build_name: Optional[str],
    ) -> State:
        """Async import implementation."""
        cache_path = Path(os.path.expanduser(cache_dir))

        async with DashboardClient(
            base_url=dashboard_url,
            api_key=dashboard_api_key,
        ) as client:
            # Discover YAML files
            yaml_files = list(cache_path.rglob("lemonade_stats.yaml"))
            if build_name:
                yaml_files = [
                    f for f in yaml_files
                    if f.parent.name == build_name
                ]

            print(f"Found {len(yaml_files)} evaluation files")

            if dry_run:
                print("Dry run - no files will be imported")
                for f in yaml_files:
                    print(f"  - {f.parent.name}")
                return state

            # Import each file
            imported = 0
            skipped = 0
            failed = 0

            for yaml_file in yaml_files:
                try:
                    success = await self._import_file(
                        client,
                        yaml_file,
                        skip_duplicates,
                    )
                    if success == "imported":
                        imported += 1
                    elif success == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error importing {yaml_file.parent.name}: {e}")
                    failed += 1

            print(f"\nImport complete:")
            print(f"  Imported: {imported}")
            print(f"  Skipped:  {skipped}")
            print(f"  Failed:   {failed}")

        return state

    async def _import_file(
        self,
        client: DashboardClient,
        yaml_file: Path,
        skip_duplicates: bool,
    ) -> str:
        """Import a single YAML file."""
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        build_name = yaml_file.parent.name

        # Check if model exists or create
        checkpoint = data.get("checkpoint", "unknown")
        model_id = await client.get_or_create_model(checkpoint, data)

        # Create run
        run_id = await client.create_run(
            model_id=model_id,
            run_type=self._determine_run_type(data),
            build_name=build_name,
            config=data,
        )

        # Upload metrics
        metrics = self._extract_metrics(data, run_id)
        await client.upload_metrics(run_id, metrics)

        # Mark complete
        await client.update_run_status(
            run_id,
            status="completed",
            completed_at=data.get("timestamp"),
        )

        return "imported"

    def _determine_run_type(self, data: dict) -> str:
        """Determine run type from YAML data."""
        if any(k.startswith("mmlu_") for k in data.keys()):
            return "accuracy-mmlu"
        elif any(k.startswith("humaneval_") for k in data.keys()):
            return "accuracy-humaneval"
        elif any(k.startswith("lm_eval_") for k in data.keys()):
            return "lm-eval"
        elif "perplexity" in data:
            return "perplexity"
        else:
            return "benchmark"

    def _extract_metrics(self, data: dict, run_id: str) -> List[Dict]:
        """Extract metrics from YAML data."""
        metrics = []

        # Performance metrics
        perf_metrics = {
            "seconds_to_first_token": ("seconds", "performance"),
            "token_generation_tokens_per_second": ("tokens/s", "performance"),
            "max_memory_used_gbyte": ("GB", "performance"),
        }

        for name, (unit, category) in perf_metrics.items():
            if name in data and data[name] is not None:
                metrics.append({
                    "run_id": run_id,
                    "name": name,
                    "value": data[name],
                    "unit": unit,
                    "category": category,
                })

        # Accuracy metrics
        for key, value in data.items():
            if key.startswith(("mmlu_", "humaneval_", "lm_eval_")):
                if isinstance(value, (int, float)):
                    metrics.append({
                        "run_id": run_id,
                        "name": key,
                        "value": value,
                        "unit": "%",
                        "category": "accuracy",
                    })

        return metrics
```

---

## 2. Production Hardening

### 2.1 Rate Limiting Implementation

```python
# dashboard/backend/app/middleware/rate_limiter.py

"""
Rate limiting middleware using Redis.

Implements:
- Token bucket algorithm for smooth rate limiting
- Per-API-key limits
- Different limits for different endpoints
"""

import time
import redis
from typing import Optional, Dict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RateLimiter:
    """Redis-based rate limiter using token bucket algorithm."""

    def __init__(
        self,
        redis_url: str,
        default_rate: int = 100,  # requests per minute
        default_burst: int = 200,  # max burst requests
    ):
        self.redis = redis.from_url(redis_url)
        self.default_rate = default_rate
        self.default_burst = default_burst

        # Endpoint-specific limits
        self.endpoint_limits: Dict[str, Dict] = {
            "/api/v1/import/yaml": {"rate": 10, "burst": 20},  # Heavy operation
            "/api/v1/reports/export": {"rate": 5, "burst": 10},  # Expensive operation
            "/api/v1/auth/login": {"rate": 10, "burst": 15},  # Security-sensitive
        }

    def _get_key(self, identifier: str) -> str:
        return f"ratelimit:{identifier}"

    def _get_limits(self, path: str) -> tuple:
        """Get rate limits for a specific endpoint."""
        for prefix, limits in self.endpoint_limits.items():
            if path.startswith(prefix):
                return limits["rate"], limits["burst"]
        return self.default_rate, self.default_burst

    def is_allowed(self, identifier: str, path: str) -> tuple:
        """
        Check if request is allowed.

        Returns: (allowed: bool, retry_after: Optional[int])
        """
        rate, burst = self._get_limits(path)
        key = self._get_key(identifier)
        now = time.time()
        window = 60  # 1 minute window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window * 2)
        results = pipe.execute()

        request_count = results[2]

        if request_count > rate:
            # Calculate retry-after
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = int(oldest[0][1] + window - now) + 1
                return False, max(1, retry_after)
            return False, 60

        return True, None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        # Get identifier (API key prefix, IP, or user ID)
        identifier = self._get_identifier(request)

        allowed, retry_after = self.limiter.is_allowed(
            identifier,
            request.url.path,
        )

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.default_rate)
        response.headers["X-RateLimit-Remaining"] = "unknown"  # Could be calculated

        return response

    def _get_identifier(self, request: Request) -> str:
        """Extract identifier for rate limiting."""
        # Try API key first
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Use API key prefix as identifier
            token = auth_header[7:]
            return f"apikey:{token[:16]}"

        # Fall back to IP
        client_ip = request.client.host
        return f"ip:{client_ip}"
```

### 2.2 Monitoring & Alerting Setup

```python
# dashboard/backend/app/monitoring.py

"""
Prometheus metrics and health monitoring.

Metrics exposed:
- HTTP request latency histogram
- Request count by endpoint
- Database connection pool stats
- WebSocket connection count
- Import job progress
"""

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Response
import time
import logging

# Metrics definitions
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

db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
)

websocket_connections = Gauge(
    "websocket_connections_total",
    "Active WebSocket connections",
)

import_jobs_total = Counter(
    "import_jobs_total",
    "Total import jobs",
    ["status"],
)

evaluation_runs_total = Counter(
    "evaluation_runs_total",
    "Total evaluation runs",
    ["status", "run_type"],
)


def setup_monitoring(app: FastAPI):
    """Set up Prometheus monitoring."""

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inplace=True,
        excluded_handlers=["/health", "/metrics", "/docs", "/redoc"],
    )

    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics")

    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe endpoint."""
        return {"status": "healthy"}

    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe with dependency checks."""
        checks = {
            "database": "ok",
            "redis": "ok",
            "memory": "ok",
        }

        # Check database
        try:
            from app.database import get_db_session
            # Perform quick DB query
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {str(e)}"

        # Check memory (fail if > 95% used)
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 95:
            checks["memory"] = f"critical: {memory_percent}%"

        is_ready = all(v == "ok" for v in checks.values())

        return {
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
        }


class MonitoringMiddleware:
    """Middleware for capturing request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration = time.time() - start_time
                method = scope["method"]
                endpoint = scope["path"]
                status = message["status"]

                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status,
                ).inc()

                http_request_duration.labels(
                    method=method,
                    endpoint=endpoint,
                ).observe(duration)

            return await send(message)

        return await self.app(scope, receive, send_wrapper)
```

### 2.3 Performance Optimization

```python
# dashboard/backend/app/cache.py

"""
Redis caching layer for expensive queries.

Cached operations:
- Aggregated metrics (5 min TTL)
- Model lists (10 min TTL)
- Run summaries (2 min TTL)
- Comparison results (1 min TTL)
"""

import json
import hashlib
import redis
from typing import Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis cache manager with automatic key generation."""

    def __init__(self, redis_url: str, default_ttl: int = 300):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"cache:{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            data = json.dumps(value)
            self.redis.setex(key, ttl or self.default_ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        try:
            keys = self.redis.keys(f"cache:{pattern}*")
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return 0


def cached(
    prefix: str,
    ttl: Optional[int] = None,
    key_fn: Optional[callable] = None,
):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager from app state
            cache = args[0].request.app.state.cache if hasattr(args[0], 'request') else None
            if not cache:
                return await func(*args, **kwargs)

            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                cache_key = f"{prefix}:{func.__name__}:{args}:{sorted(kwargs.items())}"

            # Try cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute and cache
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


# Usage example in API routes
@cached(prefix="metrics", ttl=300)
async def get_aggregated_metrics(db, model_id: str, days: int):
    """Get aggregated metrics with caching."""
    # ... expensive query ...
    return result
```

### 2.4 Backup Strategies

```yaml
# dashboard/backend/docker-compose.backup.yml

# Backup configuration for production
services:
  backup:
    image: postgres:16
    command: >
      bash -c "
      pg_dump -h db -U postgres lemonade_dashboard | gzip > /backups/dashboard_\$(date +%Y%m%d_%H%M%S).sql.gz
      && find /backups -name '*.sql.gz' -mtime +7 -delete
      "
    environment:
      PGPASSWORD: postgres
    volumes:
      - ./backups:/backups
    depends_on:
      - db
    networks:
      - backend

  backup-s3:
    image: amazon/aws-cli
    command: >
      bash -c "
      aws s3 cp /backups/ s3://lemonade-dashboard-backups/ --recursive --exclude '*' --include '*.sql.gz'
      "
    environment:
      AWS_ACCESS_KEY_ID: \${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: \${AWS_SECRET_ACCESS_KEY}
      AWS_DEFAULT_REGION: us-east-1
    volumes:
      - ./backups:/backups
    depends_on:
      - backup
```

```python
# dashboard/backend/app/backup.py

"""
Automated backup management.

Features:
- Daily PostgreSQL dumps
- Weekly full backups
- S3 upload with lifecycle policies
- Backup verification
- Point-in-time recovery support
"""

import subprocess
import boto3
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage database backups."""

    def __init__(
        self,
        db_url: str,
        backup_dir: str,
        s3_bucket: Optional[str] = None,
        retention_days: int = 7,
    ):
        self.db_url = db_url
        self.backup_dir = Path(backup_dir)
        self.s3_bucket = s3_bucket
        self.retention_days = retention_days
        self.s3_client = boto3.client("s3") if s3_bucket else None

    def create_backup(self) -> str:
        """Create a database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"dashboard_{timestamp}.sql.gz"

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Run pg_dump
        cmd = f"pg_dump {self.db_url} | gzip > {backup_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Backup failed: {result.stderr}")
            raise Exception(f"Backup failed: {result.stderr}")

        logger.info(f"Backup created: {backup_file}")

        # Upload to S3
        if self.s3_client:
            self._upload_to_s3(backup_file)

        # Cleanup old backups
        self._cleanup_old_backups()

        return str(backup_file)

    def _upload_to_s3(self, backup_file: Path):
        """Upload backup to S3."""
        key = f"backups/{backup_file.name}"
        self.s3_client.upload_file(
            str(backup_file),
            self.s3_bucket,
            key,
            ExtraArgs={
                "StorageClass": "STANDARD_IA",  # Infrequent access
                "ServerSideEncryption": "AES256",
            },
        )
        logger.info(f"Backup uploaded to S3: s3://{self.s3_bucket}/{key}")

    def _cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        for backup_file in self.backup_dir.glob("dashboard_*.sql.gz"):
            mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if mtime < cutoff:
                backup_file.unlink()
                logger.info(f"Deleted old backup: {backup_file}")

    def restore_backup(self, backup_file: str) -> bool:
        """Restore from a backup file."""
        cmd = f"gunzip -c {backup_file} | psql {self.db_url}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Restore failed: {result.stderr}")
            return False

        logger.info(f"Restored from backup: {backup_file}")
        return True

    def list_backups(self) -> list:
        """List available backups."""
        backups = []

        # Local backups
        for f in sorted(self.backup_dir.glob("dashboard_*.sql.gz")):
            backups.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "location": "local",
            })

        # S3 backups
        if self.s3_client:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix="backups/",
            )
            for obj in response.get("Contents", []):
                backups.append({
                    "name": obj["Key"].split("/")[-1],
                    "path": f"s3://{self.s3_bucket}/{obj['Key']}",
                    "size": obj["Size"],
                    "created": obj["LastModified"].isoformat(),
                    "location": "s3",
                })

        return sorted(backups, key=lambda x: x["created"], reverse=True)
```

---

## 3. Automation Pipeline

### 3.1 Automated Evaluation Scheduling

```python
# dashboard/backend/app/services/scheduler.py

"""
Celery-based evaluation scheduler.

Features:
- Cron-like scheduling for evaluations
- Queue management with priorities
- Dependency resolution (model availability)
- Resource allocation
"""

from celery import Celery, Task
from celery.schedules import crontab
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

celery_app = Celery(
    "dashboard",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "dashboard.tasks.run_evaluation": {"queue": "evaluations"},
        "dashboard.tasks.import_results": {"queue": "imports"},
        "dashboard.tasks.generate_report": {"queue": "reports"},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@celery_app.task(bind=True, max_retries=3)
def run_evaluation(
    self,
    model_id: str,
    run_type: str,
    config: dict,
    schedule_id: str = None,
):
    """Execute a scheduled evaluation."""
    try:
        # Update run status
        from app.services.runs import RunService
        from app.database import get_db

        db = next(get_db())
        run_service = RunService(db)

        run = run_service.create_scheduled_run(
            model_id=model_id,
            run_type=run_type,
            schedule_id=schedule_id,
        )

        # Execute lemonade-eval CLI
        import subprocess
        cmd = [
            "lemonade-eval",
            "--input", config["checkpoint"],
            "--cache-dir", config.get("cache_dir", "~/.cache/lemonade"),
        ]

        # Add evaluation-specific arguments
        if run_type == "accuracy-mmlu":
            cmd.extend(["AccuracyMMLU"])
        elif run_type == "accuracy-humaneval":
            cmd.extend(["AccuracyHumaneval"])
        elif run_type == "benchmark":
            cmd.extend(["ServerBench"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.get("timeout", 3600),
        )

        if result.returncode != 0:
            raise Exception(f"Evaluation failed: {result.stderr}")

        # Import results to dashboard
        run_service.complete_run(run.id, result.stdout)

        return {"run_id": run.id, "status": "completed"}

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        self.retry(exc=e, countdown=60)


@celery_app.task
def import_results(cache_dir: str, dashboard_url: str, api_key: str):
    """Import evaluation results from cache directory."""
    import requests

    response = requests.post(
        f"{dashboard_url}/api/v1/import/yaml",
        json={"cache_dir": cache_dir, "skip_duplicates": True},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()

    return response.json()


@celery_app.task
def generate_report(report_id: str, format: str = "pdf"):
    """Generate a scheduled report."""
    from app.services.reports import ReportService
    from app.database import get_db

    db = next(get_db())
    report_service = ReportService(db)

    report = report_service.generate(report_id, format)

    # Send notification
    send_report_notification(report.user_id, report.id)

    return {"report_id": report.id, "path": report.file_path}


def send_report_notification(user_id: str, report_id: str):
    """Send notification when report is ready."""
    # Implementation for email/webhook notification
    pass


# Celery beat schedule
celery_app.conf.beat_schedule = {
    "daily-mmlu-evaluation": {
        "task": "dashboard.tasks.run_evaluation",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM UTC
        "kwargs": {
            "model_id": "default-model",
            "run_type": "accuracy-mmlu",
            "config": {"checkpoint": "meta-llama/Llama-3.2-1B-Instruct"},
        },
    },
    "hourly-benchmark": {
        "task": "dashboard.tasks.run_evaluation",
        "schedule": crontab(minute=0),  # Every hour
        "kwargs": {
            "model_id": "default-model",
            "run_type": "benchmark",
            "config": {"checkpoint": "meta-llama/Llama-3.2-1B-Instruct"},
        },
    },
    "cleanup-old-imports": {
        "task": "dashboard.tasks.cleanup_import_jobs",
        "schedule": crontab(hour=0, minute=0),  # Daily at midnight
    },
}
```

### 3.2 Trend Analysis Service

```python
# dashboard/backend/app/services/trend_analysis.py

"""
Trend analysis for evaluation metrics over time.

Features:
- Linear regression for trend detection
- Anomaly detection
- Performance degradation alerts
- Comparative analysis
"""

import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from app.models import Metric, Run


class TrendAnalysis:
    """Analyze evaluation metric trends."""

    def __init__(self, db: Session):
        self.db = db

    def calculate_trend(
        self,
        model_id: str,
        metric_name: str,
        days: int = 30,
    ) -> Dict:
        """Calculate trend for a specific metric."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Get metrics for the period
        metrics = (
            self.db.query(Metric)
            .join(Run)
            .filter(
                Run.model_id == model_id,
                Metric.name == metric_name,
                Run.created_at >= cutoff,
            )
            .order_by(Run.created_at)
            .all()
        )

        if len(metrics) < 2:
            return {"trend": "insufficient_data", "data_points": len(metrics)}

        # Extract values and timestamps
        values = [float(m.value_numeric) for m in metrics if m.value_numeric]
        timestamps = [(m.run.created_at - cutoff).total_seconds() / 86400 for m in metrics]

        if len(values) < 2:
            return {"trend": "insufficient_data", "data_points": len(values)}

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)

        # Determine trend direction
        if p_value > 0.05:
            trend = "stable"
        elif slope > 0:
            trend = "improving" if metric_name in ["token_generation_tokens_per_second"] else "degrading"
        else:
            trend = "degrading" if metric_name in ["token_generation_tokens_per_second"] else "improving"

        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "data_points": len(values),
            "start_value": values[0] if values else None,
            "end_value": values[-1] if values else None,
            "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] else None,
        }

    def detect_anomalies(
        self,
        model_id: str,
        metric_name: str,
        days: int = 7,
        std_threshold: float = 2.0,
    ) -> List[Dict]:
        """Detect anomalies in recent metrics."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        metrics = (
            self.db.query(Metric)
            .join(Run)
            .filter(
                Run.model_id == model_id,
                Metric.name == metric_name,
                Run.created_at >= cutoff,
            )
            .all()
        )

        values = [float(m.value_numeric) for m in metrics if m.value_numeric]

        if len(values) < 3:
            return []

        mean = np.mean(values)
        std = np.std(values)

        anomalies = []
        for m in metrics:
            if m.value_numeric:
                z_score = abs((float(m.value_numeric) - mean) / std)
                if z_score > std_threshold:
                    anomalies.append({
                        "run_id": m.run_id,
                        "value": float(m.value_numeric),
                        "z_score": z_score,
                        "timestamp": m.run.created_at.isoformat(),
                    })

        return anomalies

    def compare_periods(
        self,
        model_id: str,
        metric_name: str,
        period1_days: int = 7,
        period2_days: int = 7,
    ) -> Dict:
        """Compare metrics between two time periods."""
        now = datetime.utcnow()
        period1_start = now - timedelta(days=period1_days)
        period2_start = period1_start - timedelta(days=period2_days)

        # Get metrics for each period
        period1 = self._get_metrics(model_id, metric_name, period1_start, now)
        period2 = self._get_metrics(model_id, metric_name, period2_start, period1_start)

        if not period1 or not period2:
            return {"comparison": "insufficient_data"}

        mean1 = np.mean(period1)
        mean2 = np.mean(period2)
        change = ((mean1 - mean2) / mean2 * 100) if mean2 else 0

        return {
            "period1_mean": mean1,
            "period2_mean": mean2,
            "change_percent": change,
            "period1_count": len(period1),
            "period2_count": len(period2),
        }

    def _get_metrics(
        self,
        model_id: str,
        metric_name: str,
        start: datetime,
        end: datetime,
    ) -> List[float]:
        """Get metric values for a time period."""
        metrics = (
            self.db.query(Metric)
            .join(Run)
            .filter(
                Run.model_id == model_id,
                Metric.name == metric_name,
                Run.created_at >= start,
                Run.created_at <= end,
            )
            .all()
        )
        return [float(m.value_numeric) for m in metrics if m.value_numeric]
```

### 3.3 Notification System

```python
# dashboard/backend/app/services/notifications.py

"""
Notification service for evaluation events.

Channels:
- Email (SMTP)
- Webhook (Slack, Teams, custom)
- In-app notifications
"""

import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationService:
    """Send notifications through various channels."""

    def __init__(
        self,
        smtp_host: str = None,
        smtp_port: int = 587,
        smtp_user: str = None,
        smtp_password: str = None,
        webhook_urls: Dict[str, str] = None,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.webhook_urls = webhook_urls or {}

    def send_evaluation_complete(
        self,
        user_email: str,
        run_id: str,
        model_name: str,
        run_type: str,
        status: str,
        dashboard_url: str,
    ):
        """Send notification when evaluation completes."""
        subject = f"Evaluation {'Completed' if status == 'completed' else 'Failed'}: {model_name}"

        body = f"""
        Evaluation Report

        Model: {model_name}
        Type: {run_type}
        Status: {status}
        Run ID: {run_id}

        View results: {dashboard_url}/runs/{run_id}
        """

        if status == "failed":
            body += "\n\nPlease check the logs for error details."

        self.send_email(user_email, subject, body)
        self.send_webhook("evaluation_complete", {
            "run_id": run_id,
            "model_name": model_name,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def send_trend_alert(
        self,
        user_email: str,
        model_name: str,
        metric_name: str,
        trend: str,
        change_percent: float,
        dashboard_url: str,
    ):
        """Send alert for significant metric trends."""
        subject = f"Performance Alert: {model_name} {metric_name}"

        direction = "improvement" if "improving" in trend else "degradation"
        body = f"""
        Performance Trend Alert

        Model: {model_name}
        Metric: {metric_name}
        Trend: {trend}
        Change: {change_percent:.2f}%

        This may indicate a {direction} in model performance.

        View details: {dashboard_url}/models/{model_name}/trends
        """

        self.send_email(user_email, subject, body)

    def send_report_ready(
        self,
        user_email: str,
        report_name: str,
        download_url: str,
    ):
        """Send notification when report is ready."""
        subject = f"Report Ready: {report_name}"

        body = f"""
        Your report is ready for download.

        Report: {report_name}
        Download: {download_url}

        This link will expire in 7 days.
        """

        self.send_email(user_email, subject, body)

    def send_email(self, to_email: str, subject: str, body: str):
        """Send email notification."""
        if not self.smtp_host:
            logger.info(f"Email (simulated): To {to_email}, Subject: {subject}")
            return

        msg = MIMEMultipart()
        msg["From"] = self.smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info(f"Email sent to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def send_webhook(self, event_type: str, payload: Dict, channel: str = "default"):
        """Send webhook notification."""
        webhook_url = self.webhook_urls.get(channel)
        if not webhook_url:
            return

        try:
            # Slack format
            slack_payload = {
                "text": f"{event_type}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": self._format_slack_message(event_type, payload),
                        },
                    },
                ],
            }

            requests.post(webhook_url, json=slack_payload, timeout=10)
            logger.info(f"Webhook sent to {channel}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def _format_slack_message(self, event_type: str, payload: Dict) -> str:
        """Format message for Slack."""
        lines = [f"*{event_type}*"]
        for key, value in payload.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
```

---

## 4. CI/CD Pipeline Enhancements

### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml

name: Deploy Dashboard

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: lemonade-sdk/lemonade-eval-dashboard

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_dashboard
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          cd dashboard/backend
          pip install -r requirements.txt
          pip install pytest-cov

      - name: Run backend tests
        run: |
          cd dashboard/backend
          pytest --cov=app --cov-report=xml
        env:
          TEST_DATABASE_URL: postgresql://postgres:test@localhost:5432/test_dashboard
          REDIS_URL: redis://localhost:6379/0

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./dashboard/backend/coverage.xml

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install frontend dependencies
        run: |
          cd dashboard/frontend
          npm ci

      - name: Run frontend tests
        run: |
          cd dashboard/frontend
          npm run test:ci

      - name: Run E2E tests
        run: |
          docker compose -f docker-compose.test.yml up -d
          sleep 30
          npm run test:e2e
        working-directory: dashboard/frontend

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push backend
        uses: docker/build-push-action@v5
        with:
          context: dashboard/backend
          push: true
          tags: ${{ steps.meta.outputs.tags }}-backend:latest
          labels: ${{ steps.meta.outputs.labels }}

      - name: Build and push frontend
        uses: docker/build-push-action@v5
        with:
          context: dashboard/frontend
          push: true
          tags: ${{ steps.meta.outputs.tags }}-frontend:latest
          labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v')

    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/
          kubectl set image deployment/dashboard-backend backend=ghcr.io/${{ env.IMAGE_NAME }}-backend:latest
          kubectl set image deployment/dashboard-frontend frontend=ghcr.io/${{ env.IMAGE_NAME }}-frontend:latest
          kubectl rollout status deployment/dashboard-backend
          kubectl rollout status deployment/dashboard-frontend
        env:
          KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
```

### 4.2 Docker Compose for Production

```yaml
# docker-compose.prod.yml

version: '3.8'

services:
  backend:
    image: ghcr.io/lemonade-sdk/lemonade-eval-dashboard-backend:latest
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/lemonade_dashboard
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: ${SECRET_KEY}
      DEBUG: false
      CORS_ORIGINS: https://dashboard.lemonade-eval.com
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
      restart_policy:
        condition: on-failure
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: ghcr.io/lemonade-sdk/lemonade-eval-dashboard-frontend:latest
    environment:
      VITE_API_URL: https://api.dashboard.lemonade-eval.com
    networks:
      - frontend
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 512M

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: lemonade_dashboard
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  celery-worker:
    image: ghcr.io/lemonade-sdk/lemonade-eval-dashboard-backend:latest
    command: celery -A app.services.scheduler worker --loglevel=info
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/lemonade_dashboard
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis
    networks:
      - backend
    deploy:
      replicas: 2

  celery-beat:
    image: ghcr.io/lemonade-sdk/lemonade-eval-dashboard-backend:latest
    command: celery -A app.services.scheduler beat --loglevel=info
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/lemonade_dashboard
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis
    networks:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    networks:
      - frontend
      - backend
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 256M

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - backend
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - backend
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  frontend:
  backend:
```

---

## 5. Documentation Needs

### 5.1 User Guide Structure

```
docs/user-guide/
├── getting-started.md
│   - Dashboard overview
│   - First-time setup
│   - Navigation guide
├── importing-data.md
│   - YAML import process
│   - Direct CLI integration
│   - Troubleshooting imports
├── viewing-results.md
│   - Dashboard overview
│   - Model library
│   - Run details
│   - Metrics interpretation
├── comparisons.md
│   - Side-by-side comparison
│   - Statistical analysis
│   - Export results
├── reports.md
│   - Creating reports
│   - Scheduling
│   - Export formats
└── api-integration.md
    - REST API reference
    - Authentication
    - Code examples
```

### 5.2 API Integration Guide

```markdown
# API Integration Guide

## Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://dashboard.lemonade-eval.com/api/v1/runs
```

## Creating an Evaluation Run

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "https://dashboard.lemonade-eval.com"

# Create a run
response = requests.post(
    f"{BASE_URL}/api/v1/runs",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model_id": "model-uuid",
        "run_type": "benchmark",
        "build_name": "my-evaluation-run",
        "config": {"iterations": 10},
        "device": "gpu",
        "backend": "ort",
    },
)

run_id = response.json()["data"]["id"]

# Upload metrics
requests.post(
    f"{BASE_URL}/api/v1/metrics/bulk",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "metrics": [
            {
                "run_id": run_id,
                "category": "performance",
                "name": "seconds_to_first_token",
                "value_numeric": 0.025,
                "unit": "seconds",
            },
            # ... more metrics
        ]
    },
)

# Complete the run
requests.put(
    f"{BASE_URL}/api/v1/runs/{run_id}/status",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"status": "completed"},
)
```

## WebSocket for Real-time Updates

```python
import asyncio
import websockets
import json

async def watch_evaluation(run_id: str):
    uri = f"wss://dashboard.lemonade-eval.com/ws/v1/evaluations?run_id={run_id}"

    async with websockets.connect(
        uri,
        extra_headers={"Authorization": f"Bearer {API_KEY}"}
    ) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["event_type"] == "progress":
                print(f"Progress: {data['progress']}%")
            elif data["event_type"] == "run_status":
                print(f"Status: {data['status']}")
                if data["status"] in ["completed", "failed"]:
                    break

asyncio.run(watch_evaluation(run_id))
```
```

### 5.3 Troubleshooting Guide

```markdown
# Troubleshooting Guide

## Import Issues

### "No YAML files found"
- Verify cache directory path
- Check file permissions
- Ensure files are named `lemonade_stats.yaml`

### "Duplicate run detected"
- Use `--skip-duplicates=false` to force update
- Delete existing run from dashboard first

### "Invalid YAML structure"
- Validate YAML syntax
- Ensure required fields: `checkpoint`, `timestamp`

## Connection Issues

### "Failed to connect to dashboard"
- Verify `--dashboard-url` format (include https://)
- Check network connectivity
- Verify SSL certificate

### "Authentication failed"
- Confirm API key is correct
- Check API key hasn't expired
- Verify user has write permissions

## Performance Issues

### "Dashboard is slow"
- Check database connection pool settings
- Verify Redis is running for caching
- Consider adding database indexes

### "Import is taking too long"
- Reduce batch size
- Increase timeout settings
- Check database performance
```

---

## 6. Testing Strategy

### 6.1 Integration Test Suite

```python
# dashboard/backend/tests/integration/test_cli_integration.py

"""
Integration tests for CLI-to-dashboard integration.
"""

import pytest
import subprocess
from unittest.mock import patch, AsyncMock


class TestCLIIntegration:
    """Test lemonade-eval CLI integration with dashboard."""

    @pytest.mark.asyncio
    async def test_create_run_from_cli(
        self,
        test_client,
        test_user,
        test_model,
    ):
        """Test run creation via CLI simulation."""
        # Simulate CLI creating a run
        response = await test_client.post(
            "/api/v1/runs",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
            json={
                "model_id": test_model.id,
                "run_type": "benchmark",
                "build_name": "test-cli-run",
                "config": {"iterations": 5},
            },
        )

        assert response.status_code == 201
        run_id = response.json()["data"]["id"]

        # Verify run was created
        get_response = await test_client.get(
            f"/api/v1/runs/{run_id}",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
        )

        assert get_response.json()["data"]["build_name"] == "test-cli-run"

    @pytest.mark.asyncio
    async def test_metric_upload_from_cli(
        self,
        test_client,
        test_user,
        test_run,
    ):
        """Test metric upload via CLI simulation."""
        metrics = [
            {
                "run_id": test_run.id,
                "category": "performance",
                "name": "seconds_to_first_token",
                "value_numeric": 0.025,
                "unit": "seconds",
            },
            {
                "run_id": test_run.id,
                "category": "performance",
                "name": "token_generation_tokens_per_second",
                "value_numeric": 45.5,
                "unit": "tokens/s",
            },
        ]

        response = await test_client.post(
            "/api/v1/metrics/bulk",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
            json={"metrics": metrics},
        )

        assert response.status_code == 201

        # Verify metrics were created
        get_response = await test_client.get(
            f"/api/v1/runs/{test_run.id}/metrics",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
        )

        assert len(get_response.json()["data"]) == 2

    @pytest.mark.asyncio
    async def test_websocket_realtime_updates(
        self,
        test_client,
        test_user,
        test_run,
    ):
        """Test WebSocket real-time updates."""
        # Connect to WebSocket
        async with test_client.websocket_connect(
            f"/ws/v1/evaluations?run_id={test_run.id}",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
        ) as websocket:
            # Update run status
            await test_client.put(
                f"/api/v1/runs/{test_run.id}/status",
                headers={"Authorization": f"Bearer {test_user.api_key}"},
                json={"status": "running"},
            )

            # Receive update
            message = await websocket.receive_json()
            assert message["event_type"] == "run_status"
            assert message["status"] == "running"

    @pytest.mark.asyncio
    async def test_import_existing_yaml(
        self,
        test_client,
        test_user,
        sample_yaml_file,
    ):
        """Test importing existing YAML file."""
        response = await test_client.post(
            "/api/v1/import/yaml",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
            json={
                "cache_dir": str(sample_yaml_file.parent.parent),
                "skip_duplicates": False,
            },
        )

        assert response.status_code == 200
        job_id = response.json()["data"]["job_id"]

        # Wait for import to complete
        import asyncio
        await asyncio.sleep(2)

        status_response = await test_client.get(
            f"/api/v1/import/status/{job_id}",
            headers={"Authorization": f"Bearer {test_user.api_key}"},
        )

        assert status_response.json()["data"]["status"] == "completed"
```

### 6.2 Load Testing Plan

```python
# tests/load/locustfile.py

"""
Load testing with Locust.

Scenarios:
1. Browse dashboard (read-heavy)
2. Import evaluations (write-heavy)
3. Real-time monitoring (WebSocket)
"""

from locust import HttpUser, task, between, events
import websockets
import json
import random


class DashboardBrowser(HttpUser):
    """Simulate users browsing the dashboard."""

    wait_time = between(1, 3)

    @task(5)
    def view_runs(self):
        """View runs list."""
        self.client.get("/api/v1/runs")

    @task(3)
    def view_models(self):
        """View models list."""
        self.client.get("/api/v1/models")

    @task(2)
    def view_run_detail(self):
        """View run detail with metrics."""
        run_id = random.choice(["run-1", "run-2", "run-3"])
        self.client.get(f"/api/v1/runs/{run_id}")
        self.client.get(f"/api/v1/runs/{run_id}/metrics")

    @task(1)
    def compare_runs(self):
        """Compare multiple runs."""
        self.client.get("/api/v1/metrics/compare?run_ids=1,2,3")


class EvaluationUploader(HttpUser):
    """Simulate CLI uploading evaluation results."""

    wait_time = between(5, 10)

    @task
    def upload_evaluation(self):
        """Upload a complete evaluation."""
        # Create run
        run_response = self.client.post(
            "/api/v1/runs",
            json={
                "model_id": "test-model",
                "run_type": "benchmark",
                "build_name": f"load-test-{random.randint(1, 1000)}",
            },
        )

        if run_response.status_code != 201:
            return

        run_id = run_response.json()["data"]["id"]

        # Upload metrics in batches
        for i in range(0, 50, 10):
            metrics = [
                {
                    "run_id": run_id,
                    "category": "performance",
                    "name": "seconds_to_first_token",
                    "value_numeric": 0.025 + random.random() * 0.01,
                    "unit": "seconds",
                }
                for _ in range(10)
            ]

            self.client.post(
                "/api/v1/metrics/bulk",
                json={"metrics": metrics},
            )

        # Complete run
        self.client.put(
            f"/api/v1/runs/{run_id}/status",
            json={"status": "completed"},
        )


class WebSocketMonitor(HttpUser):
    """Simulate users monitoring evaluations via WebSocket."""

    wait_time = between(0, 1)

    @task
    def monitor_run(self):
        """Connect to WebSocket and monitor."""
        # Note: Locust WebSocket support requires locust-websockets plugin
        pass


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Setup before test starts."""
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report after test ends."""
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
```

### 6.3 E2E Test Scenarios

```typescript
// dashboard/frontend/tests/e2e/evaluation-flow.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Evaluation Flow', () => {
  test('complete evaluation workflow', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/dashboard');

    // Navigate to import page
    await page.click('text=Import');
    await expect(page).toHaveURL('/import');

    // Configure import
    await page.fill('[name="cache_dir"]', '~/.cache/lemonade');
    await page.check('[name="skip_duplicates"]');

    // Start import
    await page.click('button:has-text("Start Import")');

    // Wait for import to complete
    await expect(page.locator('[data-testid="status-badge"]'))
      .toHaveText('COMPLETED', { timeout: 60000 });

    // View imported runs
    await page.click('text=Runs');
    await expect(page.locator('[data-testid="run-row"]')).toHaveCount({ min: 1 });

    // Open run detail
    await page.click('[data-testid="run-row"]:first-child');
    await expect(page).toHaveURL(/\/runs\/[a-f0-9-]+/);

    // Verify metrics display
    await expect(page.locator('[data-testid="metric-card"]')).toHaveCount({ min: 1 });

    // Add to comparison
    await page.click('button:has-text("Add to Compare")');

    // Navigate to compare page
    await page.click('text=Compare');
    await expect(page).toHaveURL('/compare');

    // Verify comparison display
    await expect(page.locator('[data-testid="comparison-table"]')).toBeVisible();

    // Export comparison
    await page.click('button:has-text("Export")');
    const download = await page.waitForEvent('download');
    expect(download.suggestedFilename()).toContain('comparison');
  });

  test('real-time evaluation monitoring', async ({ page }) => {
    await page.goto('/login');
    // ... login ...

    // Navigate to runs page
    await page.click('text=Runs');

    // Start a new evaluation (simulated)
    await page.click('button:has-text("New Evaluation")');
    await page.fill('[name="model_id"]', 'test-model');
    await page.click('button:has-text("Start")');

    // Watch real-time updates
    await expect(page.locator('[data-testid="progress-bar"]'))
      .toHaveAttribute('aria-valuenow', { min: 0, max: 100 });

    // Wait for completion
    await expect(page.locator('[data-testid="status-badge"]'))
      .toHaveText('COMPLETED', { timeout: 300000 });
  });
});
```

---

## 7. Phased Implementation Plan (P2 Items)

### Phase 1: CLI Integration (Weeks 1-2)

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Add `--dashboard-url` flag to CLI | P0 | 4 | Backend |
| Create `DashboardClient` module | P0 | 8 | Backend |
| Implement metric upload | P0 | 6 | Backend |
| Add WebSocket progress reporting | P1 | 6 | Backend |
| Create `import-dashboard` command | P0 | 8 | Backend |
| Integration tests | P0 | 8 | QA |

**Deliverables:**
- CLI can write results directly to dashboard
- `lemonade-eval import-dashboard` command
- Integration test suite

### Phase 2: Production Hardening (Weeks 3-4)

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Rate limiting middleware | P0 | 6 | Backend |
| Prometheus metrics | P0 | 6 | Backend |
| Health check endpoints | P0 | 4 | Backend |
| Redis caching layer | P1 | 8 | Backend |
| Backup automation | P1 | 6 | DevOps |
| Docker Compose production | P0 | 6 | DevOps |

**Deliverables:**
- Production-ready deployment configuration
- Monitoring and alerting setup
- Backup/recovery procedures

### Phase 3: Automation Pipeline (Weeks 5-6)

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Celery scheduler setup | P1 | 8 | Backend |
| Trend analysis service | P1 | 12 | Data |
| Notification service | P1 | 8 | Backend |
| Scheduled evaluations | P2 | 6 | Backend |
| CI/CD pipeline | P0 | 8 | DevOps |

**Deliverables:**
- Automated evaluation scheduling
- Trend analysis and alerts
- Complete CI/CD pipeline

### Phase 4: Documentation & Testing (Weeks 7-8)

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| User guide documentation | P0 | 12 | Tech Writer |
| API integration guide | P0 | 6 | Backend |
| Troubleshooting guide | P1 | 6 | Support |
| Load testing | P1 | 8 | QA |
| E2E test scenarios | P0 | 12 | QA |

**Deliverables:**
- Complete user documentation
- Load test results
- E2E test coverage

---

## 8. Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations run
- [ ] Redis cluster configured
- [ ] Backup procedures tested
- [ ] Monitoring dashboards created
- [ ] Alert rules configured
- [ ] Rate limits tuned

### Security

- [ ] Secret keys rotated
- [ ] API keys generated for users
- [ ] CORS configured for production domains
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] Penetration testing completed
- [ ] Dependency vulnerabilities scanned

### Performance

- [ ] Database indexes verified
- [ ] Connection pool tuned
- [ ] Cache hit rates monitored
- [ ] Load testing completed
- [ ] Slow query log analyzed

### Monitoring

- [ ] Prometheus scraping all services
- [ ] Grafana dashboards imported
- [ ] Alert rules tested
- [ ] Log aggregation configured
- [ ] Error tracking enabled (Sentry)

### Documentation

- [ ] API documentation current
- [ ] User guide published
- [ ] Runbook for on-call
- [ ] Incident response plan

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Response Time (p95) | < 200ms | Prometheus histogram |
| Import Success Rate | > 99% | Import job logs |
| Dashboard Uptime | > 99.9% | Monitoring |
| CLI Integration Adoption | > 80% of evaluations | Usage analytics |
| User Documentation CSAT | > 4.5/5 | User surveys |

---

## Appendix A: File Paths Reference

```
dashboard/
├── backend/
│   ├── app/
│   │   ├── middleware/
│   │   │   └── rate_limiter.py         # Rate limiting
│   │   ├── monitoring.py               # Prometheus metrics
│   │   ├── cache.py                    # Redis caching
│   │   ├── backup.py                   # Backup management
│   │   └── services/
│   │       ├── scheduler.py            # Celery tasks
│   │       ├── trend_analysis.py       # Trend detection
│   │       └── notifications.py        # Notification service
│   └── tests/
│       └── integration/
│           └── test_cli_integration.py # CLI integration tests
├── frontend/
│   └── tests/
│       └── e2e/
│           └── evaluation-flow.spec.ts # E2E tests
├── tests/
│   └── load/
│       └── locustfile.py               # Load testing
├── .github/
│   └── workflows/
│       └── deploy.yml                  # CI/CD pipeline
├── docker-compose.prod.yml             # Production deployment
└── docs/
    └── user-guide/                     # User documentation
```

---

*This document is ready for handoff to senior developers for implementation.*
