"""
Redis cache manager with automatic key generation.

Cached operations:
- Aggregated metrics (5 min TTL)
- Model lists (10 min TTL)
- Run summaries (2 min TTL)
- Comparison results (1 min TTL)

Cache Stampede Prevention:
- Redis-based distributed locking
- get_or_set with atomic lock acquisition
- Configurable lock timeout (default 10 seconds)
- Fallback for lock acquisition failure
"""

import hashlib
import json
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import redis
from redis.lock import Lock
from fastapi import Request

from app.config import settings


class CacheManager:
    """Redis cache manager with automatic key generation and TTL management."""

    # Default TTLs in seconds
    DEFAULT_TTL = 300  # 5 minutes
    TTL_CONFIG = {
        "models": 600,  # 10 minutes
        "metrics": 300,  # 5 minutes
        "runs": 120,  # 2 minutes
        "comparison": 60,  # 1 minute
        "aggregated": 300,  # 5 minutes
        "health": 30,  # 30 seconds
    }

    # Lock configuration for cache stampede prevention
    DEFAULT_LOCK_TIMEOUT = 10  # seconds
    DEFAULT_LOCK_RETRY_INTERVAL = 0.1  # seconds
    DEFAULT_LOCK_RETRY_COUNT = 50  # max retries

    def __init__(
        self,
        redis_url: str,
        default_ttl: int = 300,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
    ):
        """
        Initialize cache manager with connection pooling.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            max_connections: Maximum number of connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
        """
        self.redis: Optional[redis.Redis] = None
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._connected = False

        # Connection pool settings
        self.pool: Optional[redis.ConnectionPool] = None
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout

    def connect(self) -> bool:
        """
        Connect to Redis using connection pool.

        Returns:
            True if connection successful, False otherwise
        """
        if self._connected:
            return True

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

            # Test connection
            self.redis.ping()
            self._connected = True
            return True
        except redis.ConnectionError as e:
            print(f"Warning: Could not connect to Redis: {e}. Caching disabled.")
            self.redis = None
            self.pool = None
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis and close pool."""
        if self.pool:
            self.pool.disconnect()
            self.pool = None
        if self.redis:
            self.redis.close()
            self._connected = False

    def _get_lock(
        self,
        name: str,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        blocking: bool = True,
        blocking_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> Lock:
        """
        Get a distributed lock from Redis.

        Args:
            name: Lock name (will be prefixed with 'lock:')
            timeout: Lock expiration time in seconds
            blocking: Whether to block waiting for lock
            blocking_timeout: How long to wait to acquire lock

        Returns:
            Redis Lock object
        """
        if not self._connected or self.redis is None:
            raise RuntimeError("Redis not connected")

        lock_name = f"lock:{name}"
        return self.redis.lock(
            lock_name,
            timeout=timeout,
            blocking=blocking,
            blocking_timeout=blocking_timeout,
        )

    def _generate_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """
        Generate cache key from arguments.

        Args:
            prefix: Key prefix (e.g., "models", "metrics")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Generated cache key
        """
        # Create a unique string from arguments
        key_parts = [prefix]

        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))

        # Add sorted keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(str(sorted_kwargs))

        key_data = ":".join(key_parts)

        # Hash long keys to keep them manageable
        if len(key_data) > 100:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
            return f"cache:{prefix}:{key_hash}"

        # Use readable key for short data
        key_string = key_data.replace(" ", "_").replace("(", "_").replace(")", "_")
        return f"cache:{key_string}"

    def _get_ttl(self, prefix: str) -> int:
        """
        Get TTL for a specific cache prefix.

        Args:
            prefix: Cache prefix

        Returns:
            TTL in seconds
        """
        for key_prefix, ttl in self.TTL_CONFIG.items():
            if prefix.startswith(key_prefix):
                return ttl
        return self.default_ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._connected or self.redis is None:
            return None

        try:
            start_time = time.time()
            data = self.redis.get(key)
            duration = time.time() - start_time

            if data:
                from app.monitoring.metrics import record_cache_hit
                record_cache_hit("default", True, duration)
                return json.loads(data)

            from app.monitoring.metrics import record_cache_hit
            record_cache_hit("default", False, duration)
            return None

        except redis.RedisError as e:
            print(f"Cache get error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Cache JSON decode error: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            prefix: Key prefix for TTL lookup (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self._connected or self.redis is None:
            return False

        try:
            # Determine TTL
            if ttl is None:
                ttl = self._get_ttl(prefix) if prefix else self.default_ttl

            start_time = time.time()
            data = json.dumps(value, default=str)
            self.redis.setex(key, ttl, data)
            duration = time.time() - start_time

            return True

        except redis.RedisError as e:
            print(f"Cache set error: {e}")
            return False
        except (TypeError, ValueError) as e:
            print(f"Cache JSON encode error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False otherwise
        """
        if not self._connected or self.redis is None:
            return False

        try:
            self.redis.delete(key)
            return True
        except redis.RedisError as e:
            print(f"Cache delete error: {e}")
            return False

    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (e.g., "cache:models:*")

        Returns:
            Number of keys deleted
        """
        if not self._connected or self.redis is None:
            return 0

        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except redis.RedisError as e:
            print(f"Cache invalidate error: {e}")
            return 0

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all cache keys with a specific prefix.

        Args:
            prefix: Cache prefix to invalidate (e.g., "models")

        Returns:
            Number of keys deleted
        """
        return self.invalidate(f"cache:{prefix}:*")

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self._connected or self.redis is None:
            return False

        try:
            return bool(self.redis.exists(key))
        except redis.RedisError as e:
            print(f"Cache exists error: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        if not self._connected or self.redis is None:
            return -2

        try:
            return self.redis.ttl(key)
        except redis.RedisError as e:
            print(f"Cache TTL error: {e}")
            return -2

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        prefix: Optional[str] = None,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
        blocking_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> Any:
        """
        Get value from cache or set it using factory function with distributed locking.

        This method prevents cache stampede (thundering herd) by using Redis-based
        distributed locking. Only one client will execute the factory function
        while others wait for the result to be cached.

        Args:
            key: Cache key
            factory: Function to generate value if not in cache
            ttl: Time-to-live in seconds (optional)
            prefix: Key prefix for TTL lookup (optional)
            lock_timeout: Lock expiration time in seconds
            blocking_timeout: How long to wait to acquire lock

        Returns:
            Cached or newly generated value

        Raises:
            RuntimeError: If lock acquisition fails or Redis not connected
        """
        if not self._connected or self.redis is None:
            # Fallback: execute factory without caching
            print("Warning: Redis not connected, executing factory without caching")
            return factory()

        # Try to get from cache first
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        # Try to acquire lock
        lock = self._get_lock(
            name=f"compute:{key}",
            timeout=lock_timeout,
            blocking=True,
            blocking_timeout=blocking_timeout,
        )

        acquired = False
        try:
            acquired = lock.acquire(blocking=True, blocking_timeout=blocking_timeout)

            if not acquired:
                # Lock acquisition failed - fallback to executing factory
                print(f"Warning: Failed to acquire lock for key {key}, executing factory without locking")
                return factory()

            # Double-check pattern: another client may have populated cache
            # while we were waiting for the lock
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value

            # Execute factory and cache result
            result = factory()

            # Determine TTL
            if ttl is None:
                ttl = self._get_ttl(prefix) if prefix else self.default_ttl

            # Set in cache
            self.set(key, result, ttl=ttl, prefix=prefix)

            return result

        except redis.RedisError as e:
            print(f"Cache get_or_set error: {e}, executing factory without caching")
            return factory()

        finally:
            if acquired:
                try:
                    lock.release()
                except Exception as e:
                    print(f"Warning: Failed to release lock: {e}")

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        prefix: Optional[str] = None,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
        blocking_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> Any:
        """
        Async version of get_or_set with distributed locking.

        Args:
            key: Cache key
            factory: Async function to generate value if not in cache
            ttl: Time-to-live in seconds (optional)
            prefix: Key prefix for TTL lookup (optional)
            lock_timeout: Lock expiration time in seconds
            blocking_timeout: How long to wait to acquire lock

        Returns:
            Cached or newly generated value
        """
        if not self._connected or self.redis is None:
            # Fallback: execute factory without caching
            print("Warning: Redis not connected, executing factory without caching")
            return await factory()

        # Try to get from cache first
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        # Try to acquire lock
        lock = self._get_lock(
            name=f"compute:{key}",
            timeout=lock_timeout,
            blocking=True,
            blocking_timeout=blocking_timeout,
        )

        acquired = False
        try:
            acquired = lock.acquire(blocking=True, blocking_timeout=blocking_timeout)

            if not acquired:
                # Lock acquisition failed - fallback to executing factory
                print(f"Warning: Failed to acquire lock for key {key}, executing factory without locking")
                return await factory()

            # Double-check pattern: another client may have populated cache
            # while we were waiting for the lock
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value

            # Execute factory and cache result
            result = await factory()

            # Determine TTL
            if ttl is None:
                ttl = self._get_ttl(prefix) if prefix else self.default_ttl

            # Set in cache
            self.set(key, result, ttl=ttl, prefix=prefix)

            return result

        except redis.RedisError as e:
            print(f"Cache get_or_set_async error: {e}, executing factory without caching")
            return await factory()

        finally:
            if acquired:
                try:
                    lock.release()
                except Exception as e:
                    print(f"Warning: Failed to release lock: {e}")

    def health_check(self) -> dict:
        """
        Check cache health.

        Returns:
            Health status dict
        """
        if not self._connected or self.redis is None:
            return {"status": "disconnected", "connected": False}

        try:
            start_time = time.time()
            self.redis.ping()
            duration = time.time() - start_time

            info = self.redis.info("stats")
            return {
                "status": "healthy",
                "connected": True,
                "ping_latency_ms": round(duration * 1000, 2),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except redis.RedisError as e:
            return {
                "status": "unhealthy",
                "connected": True,
                "error": str(e),
            }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> Optional[CacheManager]:
    """Get the global cache manager instance."""
    return _cache_manager


def init_cache_manager(
    redis_url: Optional[str] = None,
    max_connections: int = 50,
    socket_timeout: float = 5.0,
) -> CacheManager:
    """
    Initialize the global cache manager.

    Args:
        redis_url: Redis URL (optional, uses settings if not provided)
        max_connections: Maximum number of connections in pool
        socket_timeout: Socket timeout in seconds

    Returns:
        Initialized CacheManager instance
    """
    global _cache_manager

    url = redis_url or getattr(settings, "redis_url", "redis://localhost:6379/0")
    _cache_manager = CacheManager(
        redis_url=url,
        max_connections=max_connections,
        socket_timeout=socket_timeout,
    )
    _cache_manager.connect()

    return _cache_manager


def cached(
    prefix: str,
    ttl: Optional[int] = None,
    key_fn: Optional[Callable] = None,
    cache_manager: Optional[CacheManager] = None,
):
    """
    Decorator for caching function results.

    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds (optional)
        key_fn: Optional function to generate cache key from arguments
        cache_manager: CacheManager instance (uses global if not provided)

    Returns:
        Decorated function

    Example:
        @cached(prefix="models", ttl=600)
        async def get_all_models(db):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache manager
            cm = cache_manager or get_cache_manager()
            if cm is None:
                return await func(*args, **kwargs)

            # Ensure connected
            if not cm.connect():
                return await func(*args, **kwargs)

            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                cache_key = cm._generate_key(prefix, *args, **kwargs)

            # Try cache
            cached_result = cm.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cm.set(cache_key, result, ttl=ttl, prefix=prefix)
            return result

        return wrapper
    return decorator
