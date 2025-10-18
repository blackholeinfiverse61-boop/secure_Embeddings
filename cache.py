import time
import hashlib
import json
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Simple in-memory cache manager with TTL support."""

    def __init__(self):
        self._cache = {}
        self.enabled = True

    def _make_key(self, key: str) -> str:
        """Create a consistent cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if not self.enabled:
            return None

        cache_key = self._make_key(key)
        if cache_key in self._cache:
            value, expiry = self._cache[cache_key]
            if time.time() < expiry:
                return value
            else:
                # Expired, remove it
                del self._cache[cache_key]
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (time to live in seconds)."""
        if not self.enabled:
            return False

        cache_key = self._make_key(key)
        expiry = time.time() + ttl
        self._cache[cache_key] = (value, expiry)
        return True

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False

    def flush(self):
        """Clear all cache entries."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        total_entries = len(self._cache)
        current_time = time.time()
        valid_entries = sum(1 for _, expiry in self._cache.values() if current_time < expiry)
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries
        }

# Global instance
cache_manager = CacheManager()

def cached(ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in kwargs.items())
            cache_key = "|".join(key_parts)

            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator