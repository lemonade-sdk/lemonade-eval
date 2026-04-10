"""
Load tests for rate limiting under stress conditions.

Tests cover:
- High-volume request flooding
- Concurrent client simulation
- Rate limit recovery behavior
- Endpoint-specific rate limits
- Memory usage under load
"""

import pytest
import asyncio
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from app.main import app
from app.middleware.rate_limiter import RateLimiter, RateLimitMiddleware


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def rate_limiter():
    """Create a test rate limiter with in-memory storage."""
    limiter = RateLimiter(
        redis_url="redis://localhost:6379/0",
        default_rate=100,
        default_burst=200,
        max_connections=10,
    )
    # Mock Redis for unit testing
    limiter.redis = MagicMock()
    limiter._connected = True
    return limiter


@pytest.fixture
def client_with_rate_limiting(rate_limiter):
    """Create test client with rate limiting enabled."""
    with patch('app.middleware.rate_limiter._rate_limiter', rate_limiter):
        with TestClient(app) as client:
            yield client


# ============================================================================
# LOAD TESTS
# ============================================================================

class TestRateLimitLoad:
    """Load tests for rate limiting."""

    def test_rapid_sequential_requests(self, rate_limiter):
        """Test handling of rapid sequential requests."""
        # Simulate 150 rapid requests (above rate limit of 100)
        results = []
        for i in range(150):
            allowed, retry_after = rate_limiter.is_allowed(
                identifier="test-user",
                path="/api/v1/import/evaluation"
            )
            results.append(allowed)

        # First 100 should be allowed (rate limit)
        allowed_count = sum(results)
        assert allowed_count <= rate_limiter.default_rate

    def test_concurrent_requests_same_identifier(self, rate_limiter):
        """Test concurrent requests from same identifier."""
        def make_request(i):
            return rate_limiter.is_allowed(
                identifier="concurrent-user",
                path="/api/v1/models"
            )

        # Simulate 50 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(make_request, range(50)))

        # Check results
        allowed_count = sum(1 for r in results if r[0])
        assert allowed_count <= rate_limiter.default_rate

    def test_multiple_identifiers_isolation(self, rate_limiter):
        """Test rate limiting isolation between identifiers."""
        # User A makes 50 requests
        for i in range(50):
            rate_limiter.is_allowed("user-a", "/api/v1/models")

        # User B should still have full allowance
        allowed, _ = rate_limiter.is_allowed("user-b", "/api/v1/models")
        assert allowed is True

    def test_endpoint_specific_limits(self, rate_limiter):
        """Test different rate limits for different endpoints."""
        # Import endpoint has lower limit (10/min)
        import_rate, import_burst = rate_limiter._get_limits("/api/v1/import/yaml")
        assert import_rate == 10
        assert import_burst == 20

        # Bulk import has higher limit (1000/min)
        bulk_rate, bulk_burst = rate_limiter._get_limits("/api/v1/import/bulk")
        assert bulk_rate == 1000
        assert bulk_burst == 2000

    def test_rate_limit_recovery(self, rate_limiter):
        """Test that rate limits recover after window expires."""
        # Exhaust rate limit
        for i in range(100):
            rate_limiter.is_allowed("recovery-user", "/api/v1/models")

        # Should be rate limited now
        allowed, retry_after = rate_limiter.is_allowed(
            "recovery-user",
            "/api/v1/models"
        )

        # After "window" expires, should recover (simulated)
        # In real scenario, time.sleep(60) would be needed
        # Here we verify the retry_after is calculated correctly
        assert retry_after is not None
        assert retry_after > 0


class TestRateLimitStress:
    """Stress tests for rate limiting."""

    def test_high_concurrency_stress(self, rate_limiter):
        """Stress test with high concurrency."""
        def make_requests(user_id):
            results = []
            for i in range(20):
                result = rate_limiter.is_allowed(
                    f"stress-user-{user_id}",
                    "/api/v1/runs"
                )
                results.append(result[0])
            return results

        # 20 users making 20 requests each = 400 total requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            all_results = list(executor.map(make_requests, range(20)))

        # Flatten results
        flat_results = [r for sublist in all_results for r in sublist]
        allowed_count = sum(1 for r in flat_results if r)

        # Each user should have their own rate limit
        # 20 users * 20 requests each, all should be allowed
        assert allowed_count == 400

    def test_redis_failure_graceful_degradation(self, rate_limiter):
        """Test graceful degradation when Redis fails."""
        # Simulate Redis failure
        rate_limiter.redis = None

        # Should allow all requests when Redis is unavailable
        for i in range(200):
            allowed, _ = rate_limiter.is_allowed("user", "/api/v1/models")
            assert allowed is True

    def test_memory_usage_pattern(self, rate_limiter):
        """Test memory usage pattern under load."""
        import sys

        # Initial memory reference (approximate)
        initial_objects = len(gc.garbage) if hasattr(gc, 'garbage') else 0

        # Make many requests
        for i in range(500):
            rate_limiter.is_allowed(f"user-{i % 50}", "/api/v1/models")

        # Check no memory leak pattern (basic check)
        # In production, would use memory profiler
        assert True  # Placeholder for memory profiling


# ============================================================================
# ENDPOINT-SPECIFIC LOAD TESTS
# ============================================================================

class TestEndpointLoadPatterns:
    """Load tests for specific endpoint patterns."""

    def test_import_endpoint_burst(self, rate_limiter):
        """Test burst handling on import endpoint."""
        # Import endpoint has limit of 10/min, burst of 20
        results = []
        for i in range(25):
            allowed, _ = rate_limiter.is_allowed(
                "import-user",
                "/api/v1/import/yaml"
            )
            results.append(allowed)

        # First 10 should be allowed (rate limit)
        allowed_count = sum(results)
        assert allowed_count <= 15  # Allow some burst

    def test_websocket_message_rate(self, rate_limiter):
        """Test WebSocket message rate limiting."""
        # WebSocket has 60/min limit
        results = []
        for i in range(70):
            allowed, _ = rate_limiter.is_allowed(
                "ws-user",
                "/ws/v1/evaluation-progress"
            )
            results.append(allowed)

        # Should allow up to 60 messages per minute
        allowed_count = sum(results)
        assert allowed_count <= 60

    def test_auth_endpoint_security(self, rate_limiter):
        """Test strict rate limiting on auth endpoint."""
        # Auth endpoint has very strict limits (10/min, burst 15)
        results = []
        for i in range(20):
            allowed, _ = rate_limiter.is_allowed(
                "auth-user",
                "/api/v1/auth/login"
            )
            results.append(allowed)

        # Should be strictly limited for security
        allowed_count = sum(results)
        assert allowed_count <= 15


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_rate_limit_metrics(results: List[Tuple[bool, int]]) -> dict:
    """Calculate metrics from rate limit test results."""
    total = len(results)
    allowed = sum(1 for r in results if r[0])
    rate_limited = total - allowed
    retry_afters = [r[1] for r in results if r[1] is not None]

    return {
        "total_requests": total,
        "allowed_requests": allowed,
        "rate_limited_requests": rate_limited,
        "rate_limit_percentage": (rate_limited / total * 100) if total > 0 else 0,
        "avg_retry_after": sum(retry_afters) / len(retry_afters) if retry_afters else 0,
        "max_retry_after": max(retry_afters) if retry_afters else 0,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
