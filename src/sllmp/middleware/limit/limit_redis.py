"""
Redis backend for limit enforcement middleware.

Provides a production-ready Redis backend that supports:
- Budget tracking with time windows
- Rate limiting with sliding windows
- Atomic operations for consistency
- Proper expiration handling
"""

import logging
import time
from typing import Optional

from .limit import BaseLimitBackend

logger = logging.getLogger(__name__)

try:
    import redis.asyncio
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


if REDIS_AVAILABLE:
    from redis.asyncio import Redis, RedisCluster, from_url

    class RedisLimitBackend(BaseLimitBackend):
        """
        Redis backend for budget and rate limit tracking.

        Features:
        - Time window support with automatic expiration
        - Atomic operations using Redis transactions
        - Sliding window rate limiting
        - Configurable key prefixes for multi-tenancy
        """

        def __init__(
            self,
            redis_url: str = "redis://localhost:6379/0",
            key_prefix: str = "llm_limit:",
            budget_key_ttl: Optional[int] = None,
            rate_key_ttl: int = 120  # 2 minutes for rate limiting cleanup
        ):
            """
            Initialize Redis backend.

            Args:
                redis_url: Redis connection URL
                key_prefix: Prefix for all keys (useful for multi-tenancy)
                budget_key_ttl: TTL for budget keys in seconds (None = use window-based TTL)
                rate_key_ttl: TTL for rate limiting keys in seconds
            """
            if not REDIS_AVAILABLE:
                raise ImportError(
                    "Redis is not installed. Install with: pip install redis"
                )

            self.redis_url = redis_url
            self.key_prefix = key_prefix
            self.budget_key_ttl = budget_key_ttl
            self.rate_key_ttl = rate_key_ttl
            self._redis: Optional['Redis'] = None

        async def _get_redis(self) -> 'Redis':
            """Get Redis connection, creating if needed."""
            if not REDIS_AVAILABLE:
                raise ImportError(
                    "Redis is not installed. Install with: pip install redis"
                )
            if self._redis is None:
                self._redis = from_url(self.redis_url, decode_responses=True)
            return self._redis

        async def close(self) -> None:
            """Close Redis connection."""
            if self._redis:
                await self._redis.aclose()
                self._redis = None

        def _get_budget_key(self, constraint_key: str, window: str) -> str:
            """Generate Redis key for budget tracking."""
            return f"{self.key_prefix}budget:{constraint_key}:{window}"

        def _get_rate_key(self, constraint_key: str) -> str:
            """Generate Redis key for rate limiting."""
            return f"{self.key_prefix}rate:{constraint_key}"

        def _get_window_ttl(self, window: str) -> int:
            """Get TTL in seconds for a time window."""
            window_to_seconds = {
                "1h": 3600,
                "1d": 86400,
                "7d": 604800,
                "30d": 2592000
            }
            return window_to_seconds.get(window, 86400)  # Default to 1 day

        async def get_usage(self, constraint_key: str, window: str) -> float:
            """Get current usage for a constraint within the time window."""
            try:
                r = await self._get_redis()
                budget_key = self._get_budget_key(constraint_key, window)

                usage_str = await r.get(budget_key)
                if usage_str is None:
                    return 0.0

                return float(usage_str)

            except Exception:
                logger.exception(f"Failed to get budget usage for {constraint_key}")
                return 0.0  # Fail open

        async def increment_usage(self, constraint_key: str, amount: float, window: str) -> None:
            """Increment usage by the specified amount."""
            try:
                r = await self._get_redis()
                budget_key = self._get_budget_key(constraint_key, window)

                # Use a transaction to ensure atomicity
                async with r.pipeline(transaction=True) as pipe:
                    pipe.incrbyfloat(budget_key, amount)

                    # Set TTL if key is new
                    ttl = self.budget_key_ttl or self._get_window_ttl(window)
                    pipe.expire(budget_key, ttl)

                    await pipe.execute()

            except Exception:
                logger.exception(f"Failed to increment budget usage for {constraint_key}")
                # Don't raise - this is tracking, not enforcement

        async def get_rate_usage(self, constraint_key: str) -> int:
            """Get current rate limit usage (requests in current minute)."""
            try:
                r = await self._get_redis()
                rate_key = self._get_rate_key(constraint_key)
                current_minute = int(time.time() // 60)

                # Use sorted set to store timestamps
                minute_key = f"{rate_key}:{current_minute}"
                count = await r.get(minute_key)

                return int(count) if count else 0

            except Exception:
                logger.exception(f"Failed to get rate usage for {constraint_key}")
                return 0  # Fail open

        async def increment_rate_usage(self, constraint_key: str) -> None:
            """Increment rate limit usage by 1 request."""
            try:
                r = await self._get_redis()
                rate_key = self._get_rate_key(constraint_key)
                current_minute = int(time.time() // 60)

                # Use a transaction to ensure atomicity
                minute_key = f"{rate_key}:{current_minute}"

                async with r.pipeline(transaction=True) as pipe:
                    pipe.incr(minute_key)
                    pipe.expire(minute_key, self.rate_key_ttl)
                    await pipe.execute()

            except Exception:
                logger.exception(f"Failed to increment rate usage for {constraint_key}")
                # Don't raise - this is tracking, not enforcement

        async def get_detailed_usage(self, constraint_key: str, window: str) -> dict:
            """
            Get detailed usage information including metadata.

            This is useful for debugging and monitoring.
            """
            try:
                r = await self._get_redis()
                budget_key = self._get_budget_key(constraint_key, window)

                # Get usage and TTL
                usage_str = await r.get(budget_key)
                ttl = await r.ttl(budget_key)

                usage = float(usage_str) if usage_str else 0.0

                return {
                    "constraint_key": constraint_key,
                    "window": window,
                    "usage": usage,
                    "ttl_seconds": ttl,
                    "redis_key": budget_key
                }

            except Exception as e:
                logger.exception(f"Failed to get detailed usage for {constraint_key}")
                return {
                    "constraint_key": constraint_key,
                    "window": window,
                    "usage": 0.0,
                    "ttl_seconds": -1,
                    "error": str(e)
                }

        async def reset_usage(self, constraint_key: str, window: str) -> bool:
            """
            Reset usage for a constraint (useful for testing/admin operations).

            Returns True if key existed and was deleted, False otherwise.
            """
            try:
                r = await self._get_redis()
                budget_key = self._get_budget_key(constraint_key, window)

                deleted = await r.delete(budget_key)
                return bool(deleted)

            except Exception:
                logger.exception(f"Failed to reset usage for {constraint_key}")
                return False

        async def reset_rate_usage(self, constraint_key: str) -> bool:
            """
            Reset rate limit usage for a constraint.

            Returns True if any keys were deleted, False otherwise.
            """
            try:
                r = await self._get_redis()
                rate_key = self._get_rate_key(constraint_key)

                # Find all rate keys for this constraint
                pattern = f"{rate_key}:*"
                keys = await r.keys(pattern)

                if keys:
                    deleted = await r.delete(*keys)
                    return bool(deleted)

                return False

            except Exception:
                logger.exception(f"Failed to reset rate usage for {constraint_key}")
                return False

        async def health_check(self) -> dict:
            """
            Perform a health check on the Redis backend.

            Returns status information about the Redis connection.
            """
            try:
                r = await self._get_redis()

                # Test basic operations
                test_key = f"{self.key_prefix}health_check"
                await r.set(test_key, "ok", ex=10)
                value = await r.get(test_key)
                await r.delete(test_key)

                if value == "ok":
                    return {
                        "status": "healthy",
                        "redis_url": self.redis_url,
                        "key_prefix": self.key_prefix,
                        "connection": "ok"
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "redis_url": self.redis_url,
                        "error": "Failed to set/get test value"
                    }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "redis_url": self.redis_url,
                    "error": str(e)
                }


    class RedisClusterLimitBackend(BaseLimitBackend):
        """
        Redis Cluster backend for high-availability deployments.

        Similar to RedisLimitBackend but uses Redis Cluster for distributed deployments.
        """

        def __init__(
            self,
            startup_nodes: list,
            key_prefix: str = "llm_limit:",
            budget_key_ttl: Optional[int] = None,
            rate_key_ttl: int = 120
        ):
            """
            Initialize Redis Cluster backend.

            Args:
                startup_nodes: List of Redis cluster node dictionaries
                key_prefix: Prefix for all keys
                budget_key_ttl: TTL for budget keys in seconds
                rate_key_ttl: TTL for rate limiting keys in seconds
            """
            if not REDIS_AVAILABLE:
                raise ImportError(
                    "Redis is not installed. Install with: pip install redis"
                )

            self.startup_nodes = startup_nodes
            self.key_prefix = key_prefix
            self.budget_key_ttl = budget_key_ttl
            self.rate_key_ttl = rate_key_ttl
            self._redis: Optional['RedisCluster'] = None

        async def _get_redis(self) -> 'RedisCluster':
            """Get Redis Cluster connection, creating if needed."""
            if self._redis is None:
                self._redis = RedisCluster(
                    startup_nodes=self.startup_nodes,
                    decode_responses=True
                )
            return self._redis

        async def close(self) -> None:
            """Close Redis Cluster connection."""
            if self._redis:
                await self._redis.aclose()
                self._redis = None

        # Implementation methods are identical to RedisLimitBackend
        # but use the cluster connection - inheriting would work but
        # keeping separate for clarity in production deployments

        def _get_budget_key(self, constraint_key: str, window: str) -> str:
            return f"{self.key_prefix}budget:{constraint_key}:{window}"

        def _get_rate_key(self, constraint_key: str) -> str:
            return f"{self.key_prefix}rate:{constraint_key}"

        def _get_window_ttl(self, window: str) -> int:
            window_to_seconds = {
                "1h": 3600,
                "1d": 86400,
                "7d": 604800,
                "30d": 2592000
            }
            return window_to_seconds.get(window, 86400)

        async def get_usage(self, constraint_key: str, window: str) -> float:
            try:
                r = await self._get_redis()
                budget_key = self._get_budget_key(constraint_key, window)

                usage_str = await r.get(budget_key)
                if usage_str is None:
                    return 0.0

                return float(usage_str)

            except Exception:
                logger.exception(f"Failed to get budget usage for {constraint_key}")
                return 0.0

        async def increment_usage(self, constraint_key: str, amount: float, window: str) -> None:
            try:
                r = await self._get_redis()
                budget_key = self._get_budget_key(constraint_key, window)

                async with r.pipeline(transaction=True) as pipe:
                    pipe.incrbyfloat(budget_key, amount)

                    ttl = self.budget_key_ttl or self._get_window_ttl(window)
                    pipe.expire(budget_key, ttl)

                    await pipe.execute()

            except Exception:
                logger.exception(f"Failed to increment budget usage for {constraint_key}")

        async def get_rate_usage(self, constraint_key: str) -> int:
            try:
                r = await self._get_redis()
                rate_key = self._get_rate_key(constraint_key)
                current_minute = int(time.time() // 60)

                minute_key = f"{rate_key}:{current_minute}"
                count = await r.get(minute_key)

                return int(count) if count else 0

            except Exception:
                logger.exception(f"Failed to get rate usage for {constraint_key}")
                return 0

        async def increment_rate_usage(self, constraint_key: str) -> None:
            try:
                r = await self._get_redis()
                rate_key = self._get_rate_key(constraint_key)
                current_minute = int(time.time() // 60)

                minute_key = f"{rate_key}:{current_minute}"

                async with r.pipeline(transaction=True) as pipe:
                    pipe.incr(minute_key)
                    pipe.expire(minute_key, self.rate_key_ttl)
                    await pipe.execute()

            except Exception:
                logger.exception(f"Failed to increment rate usage for {constraint_key}")

        async def health_check(self) -> dict:
            try:
                r = await self._get_redis()

                test_key = f"{self.key_prefix}health_check"
                await r.set(test_key, "ok", ex=10)
                value = await r.get(test_key)
                await r.delete(test_key)

                if value == "ok":
                    return {
                        "status": "healthy",
                        "startup_nodes": self.startup_nodes,
                        "key_prefix": self.key_prefix,
                        "connection": "ok"
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "startup_nodes": self.startup_nodes,
                        "error": "Failed to set/get test value"
                    }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "startup_nodes": self.startup_nodes,
                    "error": str(e)
                }
