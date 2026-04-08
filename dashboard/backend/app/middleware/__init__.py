"""
Middleware module for the Dashboard API.

Provides:
- Rate limiting middleware
- Request logging
- Error handling
"""

from app.middleware.rate_limiter import RateLimitMiddleware, RateLimiter

__all__ = ["RateLimitMiddleware", "RateLimiter"]
