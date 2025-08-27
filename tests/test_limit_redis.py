"""
Test suite for Redis limit backend.

These tests require a Redis instance running on localhost:6379.
Use pytest -m redis to run only these tests, or pytest -m "not redis" to skip them.
"""

import pytest
import time
from unittest.mock import AsyncMock, patch

# Mark all tests in this file as requiring Redis
pytestmark = pytest.mark.redis

try:
    from simple_llm_proxy.middleware.limit_redis import RedisLimitBackend, RedisClusterLimitBackend
    redis_available = True
except ImportError:
    redis_available = False


@pytest.fixture(scope="session")
def redis_url():
    """Redis URL for testing."""
    return "redis://localhost:6379/15"  # Use DB 15 for tests


@pytest.fixture
async def redis_backend(redis_url):
    """Redis backend fixture that cleans up after tests."""
    if not redis_available:
        pytest.skip("Redis not available")
    
    backend = RedisLimitBackend(
        redis_url=redis_url,
        key_prefix="test_llm_limit:",
        rate_key_ttl=10  # Short TTL for tests
    )
    
    # Ensure we can connect
    health = await backend.health_check()
    if health["status"] != "healthy":
        pytest.skip(f"Redis not available: {health.get('error', 'Unknown error')}")
    
    yield backend
    
    # Cleanup: remove all test keys
    try:
        r = await backend._get_redis()
        keys = await r.keys(f"{backend.key_prefix}*")
        if keys:
            await r.delete(*keys)
        await backend.close()
    except Exception:
        pass  # Best effort cleanup


@pytest.mark.asyncio
class TestRedisLimitBackend:
    """Test Redis limit backend functionality."""
    
    async def test_health_check_success(self, redis_backend):
        health = await redis_backend.health_check()
        assert health["status"] == "healthy"
        assert "connection" in health
        assert health["connection"] == "ok"
    
    async def test_health_check_failure(self):
        if not redis_available:
            pytest.skip("Redis not available")
            
        # Test with invalid Redis URL
        backend = RedisLimitBackend(redis_url="redis://invalid-host:6379/0")
        health = await backend.health_check()
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    async def test_get_usage_empty(self, redis_backend):
        usage = await redis_backend.get_usage("user:nonexistent", "1d")
        assert usage == 0.0
    
    async def test_increment_and_get_usage(self, redis_backend):
        constraint_key = "user:test_increment"
        
        # Increment usage
        await redis_backend.increment_usage(constraint_key, 5.5, "1d")
        
        # Check usage
        usage = await redis_backend.get_usage(constraint_key, "1d")
        assert usage == 5.5
    
    async def test_multiple_increments(self, redis_backend):
        constraint_key = "user:test_multiple"
        
        await redis_backend.increment_usage(constraint_key, 2.0, "1d")
        await redis_backend.increment_usage(constraint_key, 3.0, "1d")
        
        usage = await redis_backend.get_usage(constraint_key, "1d")
        assert usage == 5.0
    
    async def test_separate_keys_isolation(self, redis_backend):
        await redis_backend.increment_usage("user:test1", 10.0, "1d")
        await redis_backend.increment_usage("user:test2", 20.0, "1d")
        
        usage1 = await redis_backend.get_usage("user:test1", "1d")
        usage2 = await redis_backend.get_usage("user:test2", "1d")
        
        assert usage1 == 10.0
        assert usage2 == 20.0
    
    async def test_different_windows_isolation(self, redis_backend):
        constraint_key = "user:test_windows"
        
        await redis_backend.increment_usage(constraint_key, 5.0, "1h")
        await redis_backend.increment_usage(constraint_key, 10.0, "1d")
        
        usage_1h = await redis_backend.get_usage(constraint_key, "1h")
        usage_1d = await redis_backend.get_usage(constraint_key, "1d")
        
        assert usage_1h == 5.0
        assert usage_1d == 10.0
    
    async def test_rate_usage_empty(self, redis_backend):
        usage = await redis_backend.get_rate_usage("user:nonexistent")
        assert usage == 0
    
    async def test_increment_and_get_rate_usage(self, redis_backend):
        constraint_key = "user:test_rate"
        
        await redis_backend.increment_rate_usage(constraint_key)
        await redis_backend.increment_rate_usage(constraint_key)
        
        usage = await redis_backend.get_rate_usage(constraint_key)
        assert usage == 2
    
    async def test_rate_usage_different_keys(self, redis_backend):
        await redis_backend.increment_rate_usage("user:rate1")
        await redis_backend.increment_rate_usage("user:rate2")
        await redis_backend.increment_rate_usage("user:rate2")
        
        usage1 = await redis_backend.get_rate_usage("user:rate1")
        usage2 = await redis_backend.get_rate_usage("user:rate2")
        
        assert usage1 == 1
        assert usage2 == 2
    
    async def test_key_expiration(self, redis_backend):
        """Test that keys expire properly."""
        constraint_key = "user:test_expiry"
        
        # Set a very short TTL backend for this test
        backend = RedisLimitBackend(
            redis_url=redis_backend.redis_url,
            key_prefix="test_expire:",
            budget_key_ttl=1,  # 1 second
            rate_key_ttl=1
        )
        
        try:
            # Add usage
            await backend.increment_usage(constraint_key, 10.0, "1d")
            usage_before = await backend.get_usage(constraint_key, "1d")
            assert usage_before == 10.0
            
            # Wait for expiration
            time.sleep(2)
            
            # Should be expired
            usage_after = await backend.get_usage(constraint_key, "1d")
            assert usage_after == 0.0
            
        finally:
            await backend.close()
    
    async def test_detailed_usage(self, redis_backend):
        constraint_key = "user:test_detailed"
        window = "1h"
        
        await redis_backend.increment_usage(constraint_key, 7.5, window)
        
        details = await redis_backend.get_detailed_usage(constraint_key, window)
        
        assert details["constraint_key"] == constraint_key
        assert details["window"] == window
        assert details["usage"] == 7.5
        assert details["ttl_seconds"] > 0  # Should have TTL set
        assert "redis_key" in details
    
    async def test_reset_usage(self, redis_backend):
        constraint_key = "user:test_reset"
        
        # Add some usage
        await redis_backend.increment_usage(constraint_key, 15.0, "1d")
        
        # Verify it exists
        usage_before = await redis_backend.get_usage(constraint_key, "1d")
        assert usage_before == 15.0
        
        # Reset usage
        was_reset = await redis_backend.reset_usage(constraint_key, "1d")
        assert was_reset is True
        
        # Verify it's gone
        usage_after = await redis_backend.get_usage(constraint_key, "1d")
        assert usage_after == 0.0
        
        # Reset non-existent key
        was_reset_again = await redis_backend.reset_usage(constraint_key, "1d")
        assert was_reset_again is False
    
    async def test_reset_rate_usage(self, redis_backend):
        constraint_key = "user:test_rate_reset"
        
        # Add some rate usage
        await redis_backend.increment_rate_usage(constraint_key)
        await redis_backend.increment_rate_usage(constraint_key)
        
        # Verify it exists
        usage_before = await redis_backend.get_rate_usage(constraint_key)
        assert usage_before == 2
        
        # Reset rate usage
        was_reset = await redis_backend.reset_rate_usage(constraint_key)
        assert was_reset is True
        
        # Verify it's gone
        usage_after = await redis_backend.get_rate_usage(constraint_key)
        assert usage_after == 0


@pytest.mark.asyncio
class TestRedisBackendErrorHandling:
    """Test error handling in Redis backend."""
    
    async def test_get_usage_connection_error(self, redis_url):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(redis_url="redis://invalid-host:6379/0")
        
        # Should return 0.0 on connection error (fail open)
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 0.0
    
    async def test_increment_usage_connection_error(self, redis_url):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(redis_url="redis://invalid-host:6379/0")
        
        # Should not raise exception on connection error
        await backend.increment_usage("user:test", 5.0, "1d")
        # No assertions - just checking it doesn't crash
    
    async def test_get_rate_usage_connection_error(self, redis_url):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(redis_url="redis://invalid-host:6379/0")
        
        # Should return 0 on connection error (fail open)
        usage = await backend.get_rate_usage("user:test")
        assert usage == 0
    
    async def test_increment_rate_usage_connection_error(self, redis_url):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(redis_url="redis://invalid-host:6379/0")
        
        # Should not raise exception on connection error
        await backend.increment_rate_usage("user:test")
        # No assertions - just checking it doesn't crash


@pytest.mark.asyncio
class TestRedisBackendIntegrationWithMiddleware:
    """Test Redis backend integration with limit middleware."""
    
    async def test_middleware_with_redis_backend(self, redis_backend):
        from simple_llm_proxy.middleware.limit import LimitEnforcementMiddleware, Constraint, BudgetLimit
        from simple_llm_proxy.pipeline import create_request_context, PipelineAction
        
        # Create constraint with budget limit
        constraint = Constraint(
            name="User Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Daily budget per user"
        )
        
        middleware = LimitEnforcementMiddleware(
            constraints=[constraint], 
            backend=redis_backend
        )
        
        # Test request under limit
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'redis_test_user'
        
        # Should allow request
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        
        # Simulate response and update usage
        ctx.response = {"usage": {"total_tokens": 500}}
        await middleware.after_llm(ctx)
        
        # Verify usage was stored in Redis
        usage = await redis_backend.get_usage("user:redis_test_user", "1d")
        assert usage > 0.0
        
        # Verify tracking metadata
        assert 'budget_tracking' in ctx.metadata
        assert ctx.metadata['budget_tracking']['actual_cost'] > 0
    
    async def test_middleware_budget_exceeded_with_redis(self, redis_backend):
        from simple_llm_proxy.middleware.limit import LimitEnforcementMiddleware, Constraint, BudgetLimit
        from simple_llm_proxy.pipeline import create_request_context, PipelineAction
        
        # Pre-populate Redis with usage near limit
        constraint_key = "user:redis_budget_test"
        await redis_backend.increment_usage(constraint_key, 0.999, "1d")
        
        # Create very tight constraint
        constraint = Constraint(
            name="Tight Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Tight budget limit"
        )
        
        middleware = LimitEnforcementMiddleware(
            constraints=[constraint], 
            backend=redis_backend
        )
        
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'redis_budget_test'
        
        # Should reject request
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.HALT
        assert "Budget limit exceeded" in result_ctx.response["error"]["message"]


@pytest.mark.asyncio 
class TestRedisClusterBackend:
    """Test Redis Cluster backend (if available)."""
    
    async def test_cluster_backend_creation(self):
        if not redis_available:
            pytest.skip("Redis not available")
            
        # Just test that we can create the backend
        startup_nodes = [{"host": "localhost", "port": 7000}]
        backend = RedisClusterLimitBackend(
            startup_nodes=startup_nodes,
            key_prefix="cluster_test:"
        )
        
        # Don't test actual functionality since we likely don't have a cluster
        # Just verify the backend was created with correct configuration
        assert backend.startup_nodes == startup_nodes
        assert backend.key_prefix == "cluster_test:"
        
        await backend.close()


class TestRedisBackendConfiguration:
    """Test Redis backend configuration options."""
    
    def test_redis_backend_initialization(self):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(
            redis_url="redis://custom-host:6380/5",
            key_prefix="custom_prefix:",
            budget_key_ttl=3600,
            rate_key_ttl=300
        )
        
        assert backend.redis_url == "redis://custom-host:6380/5"
        assert backend.key_prefix == "custom_prefix:"
        assert backend.budget_key_ttl == 3600
        assert backend.rate_key_ttl == 300
    
    def test_window_ttl_calculation(self, redis_url):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(redis_url=redis_url)
        
        assert backend._get_window_ttl("1h") == 3600
        assert backend._get_window_ttl("1d") == 86400
        assert backend._get_window_ttl("7d") == 604800
        assert backend._get_window_ttl("30d") == 2592000
        assert backend._get_window_ttl("unknown") == 86400  # Default
    
    def test_key_generation(self, redis_url):
        if not redis_available:
            pytest.skip("Redis not available")
            
        backend = RedisLimitBackend(redis_url=redis_url, key_prefix="test:")
        
        budget_key = backend._get_budget_key("user:test", "1d")
        rate_key = backend._get_rate_key("user:test")
        
        assert budget_key == "test:budget:user:test:1d"
        assert rate_key == "test:rate:user:test"


class TestRedisImportError:
    """Test behavior when Redis is not installed."""
    
    def test_import_error_handling(self):
        # Test that module handles Redis import error gracefully  
        import sys
        import importlib
        
        # Save original redis module if it exists
        original_redis = sys.modules.get('redis.asyncio')
        limit_redis_module = None
        
        try:
            # First import the module normally to get a reference
            from simple_llm_proxy.middleware.limit import limit_redis as limit_redis_module
            
            # Remove redis from modules to simulate import error
            if 'redis.asyncio' in sys.modules:
                del sys.modules['redis.asyncio']
            
            # Set up import error simulation
            sys.modules['redis.asyncio'] = None
            
            # Reload the module so the import error takes effect
            importlib.reload(limit_redis_module)
            
            # Now try to create - should fail due to import error check
            with pytest.raises(ImportError, match="Redis is not installed"):
                limit_redis_module.RedisLimitBackend()
                
        finally:
            # Restore original state
            if original_redis is not None:
                sys.modules['redis.asyncio'] = original_redis
            elif 'redis.asyncio' in sys.modules and sys.modules['redis.asyncio'] is None:
                del sys.modules['redis.asyncio']
            
            # Reload the module back to original state if we have a reference
            if limit_redis_module is not None:
                importlib.reload(limit_redis_module)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "redis"])