"""
Caching module for the Dashboard API.

Provides:
- Redis caching layer
- Cache decorators for expensive queries
- Cache invalidation utilities
"""

from app.cache.cache_manager import CacheManager, cached, get_cache_manager, init_cache_manager
from app.cache.cache_service import CacheService

__all__ = [
    "CacheManager",
    "CacheService",
    "cached",
    "get_cache_manager",
    "init_cache_manager",
]
