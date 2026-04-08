"""
Rate limiting middleware using Redis.

Implements:
- Token bucket algorithm for smooth rate limiting
- Per-API-key limits
- Different limits for different endpoints
- Configurable via environment variables
- Connection pooling with max 50 connections
"""

import time
from typing import Optional, Dict, Tuple

import redis
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import settings


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""

    def __init__(
        self,
        redis_url: str,
        default_rate: int = 100,  # requests per minute
        default_burst: int = 200,  # max burst requests
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
    ):
        """
        Initialize the rate limiter with connection pooling.

        Args:
            redis_url: Redis connection URL
            default_rate: Default requests per minute limit
            default_burst: Maximum burst requests allowed
            max_connections: Maximum number of connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
        """
        self.redis: Optional[redis.Redis] = None
        self.pool: Optional[redis.ConnectionPool] = None
        self.redis_url = redis_url
        self.default_rate = default_rate
        self.default_burst = default_burst

        # Connection pool settings
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout

        # Endpoint-specific limits (path prefix -> {rate, burst})
        self.endpoint_limits: Dict[str, Dict[str, int]] = {
            "/api/v1/import/yaml": {"rate": 10, "burst": 20},  # Heavy operation
            "/api/v1/import/bulk": {"rate": 1000, "burst": 2000},  # High throughput import
            "/api/v1/reports/export": {"rate": 5, "burst": 10},  # Expensive operation
            "/api/v1/auth/login": {"rate": 10, "burst": 15},  # Security-sensitive
            "/api/v1/metrics/bulk": {"rate": 500, "burst": 1000},  # High throughput
            "/ws/v1/evaluation-progress": {"rate": 60, "burst": 120},  # WebSocket messages
        }

    def connect(self) -> None:
        """Connect to Redis using connection pool."""
        if self.redis is None:
            try:
                # Create connection pool
                self.pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=self.socket_connect_timeout,
                    socket_timeout=self.socket_timeout,
                    retry_on_timeout=self.retry_on_timeout,
                    max_connections=self.max_connections,
                )

                # Create Redis instance with pool
                self.redis = redis.Redis(connection_pool=self.pool)
            except redis.ConnectionError as e:
                print(f"Warning: Could not connect to Redis: {e}. Rate limiting disabled.")
                self.redis = None
                self.pool = None

    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for rate limiting."""
        return f"ratelimit:{identifier}"

    def _get_limits(self, path: str) -> Tuple[int, int]:
        """Get rate limits for a specific endpoint."""
        for prefix, limits in self.endpoint_limits.items():
            if path.startswith(prefix):
                return limits["rate"], limits["burst"]
        return self.default_rate, self.default_burst

    def is_allowed(self, identifier: str, path: str) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed using sliding window algorithm.

        Args:
            identifier: Unique identifier (API key prefix or IP)
            path: Request path

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        # If Redis not available, allow all requests
        if self.redis is None:
            return True, None

        try:
            rate, burst = self._get_limits(path)
            key = self._get_key(identifier)
            now = time.time()
            window = 60  # 1 minute window

            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, window * 2)
            results = pipe.execute()

            request_count = int(results[2])

            if request_count > rate:
                # Calculate retry-after based on oldest request in window
                oldest = self.redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1] + window - now) + 1
                    return False, max(1, retry_after)
                return False, 60

            return True, None

        except redis.RedisError as e:
            # Log error but don't block requests
            print(f"Rate limiter Redis error: {e}")
            return True, None

    def get_remaining(self, identifier: str, path: str) -> int:
        """Get remaining requests for an identifier."""
        if self.redis is None:
            return -1  # Unknown

        try:
            rate, _ = self._get_limits(path)
            key = self._get_key(identifier)
            now = time.time()
            window = 60

            # Count requests in current window
            count = self.redis.zcount(key, now - window, now)
            return max(0, rate - count)

        except redis.RedisError:
            return -1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        limiter: RateLimiter,
        enabled: bool = True,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            limiter: RateLimiter instance
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.limiter = limiter
        self.enabled = enabled and settings.debug is False  # Disable in debug mode

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process incoming request with rate limiting."""
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)

        # Connect to Redis if not already connected
        if self.limiter.redis is None:
            self.limiter.connect()

        # Get identifier for rate limiting
        identifier = self._get_identifier(request)

        # Check rate limit
        allowed, retry_after = self.limiter.is_allowed(
            identifier,
            request.url.path,
        )

        if not allowed:
            # Ensure Redis is available for rate limiting
            if self.limiter.redis is not None:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        remaining = self.limiter.get_remaining(identifier, request.url.path)
        rate, _ = self.limiter._get_limits(request.url.path)

        response.headers["X-RateLimit-Limit"] = str(rate)
        response.headers["X-RateLimit-Remaining"] = str(remaining) if remaining >= 0 else "unknown"

        return response

    def _get_identifier(self, request: Request) -> str:
        """
        Extract identifier for rate limiting.

        Priority:
        1. API key from Authorization header
        2. Client IP address
        """
        # Try API key first
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Use API key prefix as identifier for granular limiting
            token = auth_header[7:]
            # Use first 16 chars as identifier
            return f"apikey:{token[:16]}"

        # Fall back to IP address
        # Check for X-Forwarded-For header (for proxied requests)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"


# Global rate limiter instance (initialized when app starts)
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> Optional[RateLimiter]:
    """Get the global rate limiter instance."""
    return _rate_limiter


def init_rate_limiter(
    redis_url: Optional[str] = None,
    max_connections: int = 50,
    socket_timeout: float = 5.0,
) -> RateLimiter:
    """
    Initialize the global rate limiter.

    Args:
        redis_url: Redis URL (optional, uses settings if not provided)
        max_connections: Maximum number of connections in pool
        socket_timeout: Socket timeout in seconds

    Returns:
        Initialized RateLimiter instance
    """
    global _rate_limiter

    url = redis_url or getattr(settings, "redis_url", "redis://localhost:6379/0")

    # Get rate limit settings
    default_rate = int(getattr(settings, "rate_limit_default", 100))
    default_burst = int(getattr(settings, "rate_limit_burst", 200))

    _rate_limiter = RateLimiter(
        redis_url=url,
        default_rate=default_rate,
        default_burst=default_burst,
        max_connections=max_connections,
        socket_timeout=socket_timeout,
    )
    _rate_limiter.connect()

    return _rate_limiter
