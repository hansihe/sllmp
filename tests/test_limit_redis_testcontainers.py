"""
Test suite for Redis limit backend using testcontainers.

This test suite automatically starts a Redis container for testing,
eliminating the need for manual Redis setup.
"""

import pytest
import asyncio
import time
import docker
from testcontainers.redis import RedisContainer

pytestmark = [pytest.mark.asyncio, pytest.mark.containers]

try:
    from simple_llm_proxy.middleware.limit.limit_redis import RedisLimitBackend
    redis_available = True
except ImportError:
    redis_available = False

# Check if Docker is available
docker_available = False
try:
    client = docker.from_env()
    client.ping()
    docker_available = True
    client.close()
except Exception:
    docker_available = False


@pytest.fixture(scope="session")
def redis_container():
    """Start a Redis container for the test session."""
    if not redis_available:
        pytest.skip("Redis backend not available")
    if not docker_available:
        pytest.skip("Docker not available for testcontainers")
    
    try:
        container = RedisContainer("redis:7-alpine")
        container.start()
        
        yield container
        
        container.stop()
    except Exception as e:
        pytest.skip(f"Failed to start Redis container: {e}")


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """Get the Redis URL from the container."""
    return redis_container.get_connection_url()


@pytest.fixture
async def redis_backend(redis_url):
    """Redis backend fixture that cleans up after tests."""
    backend = RedisLimitBackend(
        redis_url=redis_url,
        key_prefix="test_container:",
        rate_key_ttl=10
    )
    
    health = await backend.health_check()
    if health["status"] != "healthy":
        pytest.skip(f"Redis container not healthy: {health.get('error', 'Unknown error')}")
    
    yield backend
    
    try:
        r = await backend._get_redis()
        keys = await r.keys(f"{backend.key_prefix}*")
        if keys:
            await r.delete(*keys)
        await backend.close()
    except Exception:
        pass


class TestRedisLimitBackendWithContainers:
    """Test Redis limit backend with testcontainers."""
    
    async def test_container_health_check(self, redis_backend):
        """Test that the Redis container is working."""
        health = await redis_backend.health_check()
        assert health["status"] == "healthy"
        assert health["connection"] == "ok"
    
    async def test_basic_usage_tracking(self, redis_backend):
        """Test basic usage increment and retrieval."""
        constraint_key = "user:container_test"
        
        await redis_backend.increment_usage(constraint_key, 10.5, "1d")
        usage = await redis_backend.get_usage(constraint_key, "1d")
        
        assert usage == 10.5
    
    async def test_multiple_users_isolation(self, redis_backend):
        """Test that different users have isolated usage tracking."""
        await redis_backend.increment_usage("user:alice", 5.0, "1d")
        await redis_backend.increment_usage("user:bob", 7.5, "1d")
        
        alice_usage = await redis_backend.get_usage("user:alice", "1d")
        bob_usage = await redis_backend.get_usage("user:bob", "1d")
        
        assert alice_usage == 5.0
        assert bob_usage == 7.5
    
    async def test_window_isolation(self, redis_backend):
        """Test that different time windows are isolated."""
        constraint_key = "user:window_test"
        
        await redis_backend.increment_usage(constraint_key, 2.0, "1h")
        await redis_backend.increment_usage(constraint_key, 3.0, "1d")
        
        hourly_usage = await redis_backend.get_usage(constraint_key, "1h")
        daily_usage = await redis_backend.get_usage(constraint_key, "1d")
        
        assert hourly_usage == 2.0
        assert daily_usage == 3.0
    
    async def test_cumulative_usage(self, redis_backend):
        """Test that usage accumulates correctly."""
        constraint_key = "user:cumulative"
        
        await redis_backend.increment_usage(constraint_key, 1.0, "1d")
        await redis_backend.increment_usage(constraint_key, 2.5, "1d")
        await redis_backend.increment_usage(constraint_key, 1.5, "1d")
        
        total_usage = await redis_backend.get_usage(constraint_key, "1d")
        assert total_usage == 5.0
    
    async def test_rate_limiting(self, redis_backend):
        """Test rate limiting functionality."""
        constraint_key = "user:rate_test"
        
        await redis_backend.increment_rate_usage(constraint_key)
        await redis_backend.increment_rate_usage(constraint_key)
        await redis_backend.increment_rate_usage(constraint_key)
        
        rate_usage = await redis_backend.get_rate_usage(constraint_key)
        assert rate_usage == 3
    
    async def test_rate_limiting_isolation(self, redis_backend):
        """Test that rate limits are isolated between users."""
        await redis_backend.increment_rate_usage("user:rate_alice")
        await redis_backend.increment_rate_usage("user:rate_bob")
        await redis_backend.increment_rate_usage("user:rate_bob")
        
        alice_rate = await redis_backend.get_rate_usage("user:rate_alice")
        bob_rate = await redis_backend.get_rate_usage("user:rate_bob")
        
        assert alice_rate == 1
        assert bob_rate == 2
    
    async def test_detailed_usage_info(self, redis_backend):
        """Test detailed usage information retrieval."""
        constraint_key = "user:detailed"
        window = "1h"
        
        await redis_backend.increment_usage(constraint_key, 12.5, window)
        
        details = await redis_backend.get_detailed_usage(constraint_key, window)
        
        assert details["constraint_key"] == constraint_key
        assert details["window"] == window
        assert details["usage"] == 12.5
        assert details["ttl_seconds"] > 0
        assert details["redis_key"] == f"{redis_backend.key_prefix}budget:{constraint_key}:{window}"
    
    async def test_usage_reset(self, redis_backend):
        """Test resetting usage."""
        constraint_key = "user:reset_test"
        
        await redis_backend.increment_usage(constraint_key, 20.0, "1d")
        
        usage_before = await redis_backend.get_usage(constraint_key, "1d")
        assert usage_before == 20.0
        
        was_reset = await redis_backend.reset_usage(constraint_key, "1d")
        assert was_reset is True
        
        usage_after = await redis_backend.get_usage(constraint_key, "1d")
        assert usage_after == 0.0
        
        was_reset_again = await redis_backend.reset_usage(constraint_key, "1d")
        assert was_reset_again is False
    
    async def test_rate_usage_reset(self, redis_backend):
        """Test resetting rate usage."""
        constraint_key = "user:rate_reset"
        
        await redis_backend.increment_rate_usage(constraint_key)
        await redis_backend.increment_rate_usage(constraint_key)
        
        rate_before = await redis_backend.get_rate_usage(constraint_key)
        assert rate_before == 2
        
        was_reset = await redis_backend.reset_rate_usage(constraint_key)
        assert was_reset is True
        
        rate_after = await redis_backend.get_rate_usage(constraint_key)
        assert rate_after == 0
    
    async def test_key_expiration(self, redis_backend):
        """Test that keys expire correctly."""
        backend = RedisLimitBackend(
            redis_url=redis_backend.redis_url,
            key_prefix="test_expire:",
            budget_key_ttl=1,
            rate_key_ttl=1
        )
        
        try:
            constraint_key = "user:expire_test"
            
            await backend.increment_usage(constraint_key, 15.0, "1d")
            usage_before = await backend.get_usage(constraint_key, "1d")
            assert usage_before == 15.0
            
            await asyncio.sleep(2)
            
            usage_after = await backend.get_usage(constraint_key, "1d")
            assert usage_after == 0.0
            
        finally:
            await backend.close()
    
    async def test_concurrent_operations(self, redis_backend):
        """Test concurrent operations don't cause race conditions."""
        constraint_key = "user:concurrent"
        
        async def increment_usage():
            await redis_backend.increment_usage(constraint_key, 1.0, "1d")
        
        # Run 10 concurrent increments
        tasks = [increment_usage() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        final_usage = await redis_backend.get_usage(constraint_key, "1d")
        assert final_usage == 10.0
    
    async def test_large_usage_values(self, redis_backend):
        """Test handling of large usage values."""
        constraint_key = "user:large_values"
        
        large_value = 1234567.89
        await redis_backend.increment_usage(constraint_key, large_value, "1d")
        
        usage = await redis_backend.get_usage(constraint_key, "1d")
        assert abs(usage - large_value) < 0.01  # Float precision tolerance
    
    async def test_negative_usage_values(self, redis_backend):
        """Test handling of negative usage values (credits/refunds)."""
        constraint_key = "user:credits"
        
        await redis_backend.increment_usage(constraint_key, 10.0, "1d")
        await redis_backend.increment_usage(constraint_key, -3.0, "1d")  # Credit
        
        usage = await redis_backend.get_usage(constraint_key, "1d")
        assert usage == 7.0


class TestRedisBackendIntegrationWithContainers:
    """Test Redis backend integration with middleware using containers."""
    
    async def test_middleware_integration(self, redis_backend):
        """Test full integration with limit enforcement middleware."""
        from simple_llm_proxy.middleware.limit.limit import LimitEnforcementMiddleware, Constraint, BudgetLimit
        from simple_llm_proxy.pipeline import create_request_context, PipelineAction
        
        constraint = Constraint(
            name="Container Test Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=2.0, window="1d"),
            rate_limit=None,
            description="Test budget with containers"
        )
        
        middleware = LimitEnforcementMiddleware(
            constraints=[constraint],
            backend=redis_backend
        )
        
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test message"}]
        }
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'container_integration_user'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        
        ctx.response = {"usage": {"total_tokens": 1000}}
        await middleware.after_llm(ctx)
        
        usage = await redis_backend.get_usage("user:container_integration_user", "1d")
        assert usage > 0.0
    
    async def test_budget_enforcement_with_containers(self, redis_backend):
        """Test budget enforcement using containers."""
        from simple_llm_proxy.middleware.limit.limit import LimitEnforcementMiddleware, Constraint, BudgetLimit
        from simple_llm_proxy.pipeline import create_request_context, PipelineAction
        
        await redis_backend.increment_usage("user:budget_enforce", 1.95, "1d")
        
        constraint = Constraint(
            name="Strict Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=2.0, window="1d"),
            rate_limit=None,
            description="Strict budget limit"
        )
        
        middleware = LimitEnforcementMiddleware(
            constraints=[constraint],
            backend=redis_backend
        )
        
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "This should be blocked"}]
        }
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'budget_enforce'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.HALT
        assert "Budget limit exceeded" in result_ctx.response["error"]["message"]


class TestRedisContainerConfiguration:
    """Test various Redis configuration options with containers."""
    
    async def test_custom_key_prefix(self, redis_url):
        """Test custom key prefix functionality."""
        backend = RedisLimitBackend(
            redis_url=redis_url,
            key_prefix="custom_test:",
            budget_key_ttl=3600
        )
        
        try:
            constraint_key = "user:prefix_test"
            await backend.increment_usage(constraint_key, 5.0, "1d")
            
            r = await backend._get_redis()
            keys = await r.keys("custom_test:*")
            assert len(keys) > 0
            
            expected_key = "custom_test:budget:user:prefix_test:1d"
            assert expected_key in keys
            
        finally:
            await backend.close()
    
    async def test_window_ttl_calculations(self, redis_url):
        """Test TTL calculations for different windows."""
        backend = RedisLimitBackend(redis_url=redis_url)
        
        try:
            assert backend._get_window_ttl("1h") == 3600
            assert backend._get_window_ttl("1d") == 86400
            assert backend._get_window_ttl("7d") == 604800
            assert backend._get_window_ttl("30d") == 2592000
            assert backend._get_window_ttl("unknown_window") == 86400  # Default
            
        finally:
            await backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])