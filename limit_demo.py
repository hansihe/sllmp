#!/usr/bin/env python3
"""
Demo of limit enforcement middleware with different backends.

This script demonstrates:
1. Budget enforcement with in-memory backend
2. Rate limiting with in-memory backend  
3. Redis backend usage (if Redis is available)
4. Multi-dimensional constraints
5. Production-like configuration
"""

import asyncio
from simple_llm_proxy.builder import PipelineBuilder
from simple_llm_proxy.pipeline import create_request_context
from simple_llm_proxy.middleware import (
    LimitEnforcementMiddleware, 
    BudgetLimit, 
    RateLimit,
    Constraint,
    InMemoryLimitBackend,
    AuthMiddleware,
    LoggingMiddleware
)

try:
    from simple_llm_proxy.middleware import RedisLimitBackend
    redis_available = True
except (ImportError, AttributeError):
    redis_available = False


async def demo_in_memory_backend():
    """Demonstrate limit enforcement with in-memory backend."""
    print("=== In-Memory Backend Demo ===")
    
    # Create constraints
    constraints = [
        # Per-user daily budget of $5
        Constraint(
            name="User Daily Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=5.0, window="1d"),
            rate_limit=None,
            description="$5 daily budget per user"
        ),
        # Per-user rate limit of 10 requests per minute
        Constraint(
            name="User Rate Limit", 
            dimensions=["user_id"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=10),
            description="10 requests per minute per user"
        ),
        # Per-organization monthly budget of $1000
        Constraint(
            name="Organization Monthly Budget",
            dimensions=["organization"],
            budget_limit=BudgetLimit(limit=1000.0, window="30d"),
            rate_limit=None,
            description="$1000 monthly budget per organization"
        )
    ]
    
    # Create backend and middleware
    backend = InMemoryLimitBackend()
    limit_middleware = LimitEnforcementMiddleware(
        constraints=constraints,
        backend=backend
    )
    
    # Build pipeline
    pipeline = (PipelineBuilder()
        .add(AuthMiddleware(require_user_id=False))
        .add(limit_middleware)
        .add(LoggingMiddleware())
        .build())
    
    # Test normal request
    print("Testing normal request...")
    request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "metadata": {
            "user_id": "demo_user",
            "organization": "demo_org"
        }
    }
    
    ctx = create_request_context(request)
    result_ctx = await pipeline.execute(ctx)
    
    print(f"Request result: {result_ctx.action}")
    if result_ctx.response and 'budget_tracking' in result_ctx.metadata:
        tracking = result_ctx.metadata['budget_tracking']
        print(f"Estimated cost: ${tracking['estimated_cost']:.6f}")
        print(f"Actual cost: ${tracking['actual_cost']:.6f}")
    
    # Check current usage
    print("\nCurrent usage after request:")
    user_usage = await backend.get_usage("user:demo_user", "1d")
    org_usage = await backend.get_usage("org:demo_org", "30d")
    user_rate = await backend.get_rate_usage("user:demo_user")
    
    print(f"User daily budget usage: ${user_usage:.6f} / $5.00")
    print(f"Organization monthly usage: ${org_usage:.6f} / $1000.00")
    print(f"User rate limit usage: {user_rate} / 10 requests/minute")
    print()


async def demo_budget_exceeded():
    """Demonstrate budget limit enforcement."""
    print("=== Budget Limit Demo ===")
    
    # Create very tight budget constraint
    tight_constraint = Constraint(
        name="Tight Budget",
        dimensions=["user_id"],
        budget_limit=BudgetLimit(limit=0.001, window="1d"),  # Very tight
        rate_limit=None,
        description="Tight budget limit for demo"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([tight_constraint], backend)
    
    request = {
        "model": "openai:gpt-4",  # More expensive model
        "messages": [{"role": "user", "content": "Write a long analysis" * 50}]
    }
    
    ctx = create_request_context(request)
    ctx.client_metadata['user_id'] = 'budget_demo_user'
    
    result_ctx = await middleware.before_llm(ctx)
    
    print(f"Request with tight budget: {result_ctx.action}")
    if result_ctx.action.value == "halt":
        error_msg = result_ctx.response["error"]["message"]
        print(f"Blocked: {error_msg}")
    print()


async def demo_rate_limit_exceeded():
    """Demonstrate rate limit enforcement."""
    print("=== Rate Limit Demo ===")
    
    # Create tight rate limit
    rate_constraint = Constraint(
        name="Tight Rate Limit",
        dimensions=["user_id"],
        budget_limit=None,
        rate_limit=RateLimit(per_minute=2),  # Only 2 requests per minute
        description="Tight rate limit for demo"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([rate_constraint], backend)
    
    request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Quick test"}]
    }
    
    # Make multiple requests rapidly
    for i in range(4):
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'rate_demo_user'
        
        result_ctx = await middleware.before_llm(ctx)
        
        print(f"Request {i+1}: {result_ctx.action}")
        if result_ctx.action.value == "halt":
            error_msg = result_ctx.response["error"]["message"]
            print(f"  Blocked: {error_msg}")
        
        # Small delay between requests
        await asyncio.sleep(0.1)
    print()


async def demo_redis_backend():
    """Demonstrate Redis backend (if available)."""
    if not redis_available:
        print("=== Redis Backend Demo ===")
        print("Redis backend not available (redis package not installed)")
        print("Install with: pip install redis")
        print()
        return
    
    print("=== Redis Backend Demo ===")
    
    try:
        # Create Redis backend
        redis_backend = RedisLimitBackend(
            redis_url="redis://localhost:6379/0",
            key_prefix="demo_llm_limit:",
            budget_key_ttl=3600  # 1 hour
        )
        
        # Test health check
        health = await redis_backend.health_check()
        if health["status"] != "healthy":
            print(f"Redis not available: {health.get('error', 'Unknown error')}")
            print("Make sure Redis is running on localhost:6379")
            print()
            return
        
        print("✅ Redis connection healthy")
        
        # Create constraint
        constraint = Constraint(
            name="Redis User Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=10.0, window="1h"),
            rate_limit=None,
            description="Hourly budget stored in Redis"
        )
        
        middleware = LimitEnforcementMiddleware([constraint], redis_backend)
        
        # Test request
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test Redis backend"}]
        }
        
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'redis_demo_user'
        
        result_ctx = await middleware.before_llm(ctx)
        print(f"Request result: {result_ctx.action}")
        
        # Simulate response
        ctx.response = {"usage": {"total_tokens": 250}}
        await middleware.after_llm(ctx)
        
        # Check Redis directly
        usage = await redis_backend.get_usage("user:redis_demo_user", "1h")
        detailed = await redis_backend.get_detailed_usage("user:redis_demo_user", "1h")
        
        print(f"Usage stored in Redis: ${usage:.6f}")
        print(f"Redis key: {detailed['redis_key']}")
        print(f"TTL: {detailed['ttl_seconds']} seconds")
        
        # Cleanup
        await redis_backend.reset_usage("user:redis_demo_user", "1h")
        await redis_backend.close()
        print("✅ Redis demo completed and cleaned up")
        
    except Exception as e:
        print(f"Redis demo failed: {e}")
        print("Make sure Redis is running on localhost:6379")
    
    print()


async def demo_multi_dimensional_constraints():
    """Demonstrate multi-dimensional constraints."""
    print("=== Multi-Dimensional Constraints Demo ===")
    
    # Constraint that combines user + organization
    constraint = Constraint(
        name="User-Org Combined Budget",
        dimensions=["user_id", "organization"],
        budget_limit=BudgetLimit(limit=50.0, window="7d"),
        rate_limit=None,
        description="Weekly budget per user per organization"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([constraint], backend)
    
    # Test requests from same user in different orgs
    requests = [
        {"user": "alice", "org": "engineering"},
        {"user": "alice", "org": "marketing"}, 
        {"user": "bob", "org": "engineering"}
    ]
    
    for req_info in requests:
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Multi-dim test"}]
        }
        
        ctx = create_request_context(request)
        ctx.client_metadata.update({
            'user_id': req_info["user"],
            'organization': req_info["org"]
        })
        
        await middleware.before_llm(ctx)
        
        # Simulate response
        ctx.response = {"usage": {"total_tokens": 100}}
        await middleware.after_llm(ctx)
        
        # Check usage for this specific user+org combination
        key_parts = [f"org:{req_info['org']}", f"user:{req_info['user']}"]
        combined_key = "|".join(sorted(key_parts))
        usage = await backend.get_usage(combined_key, "7d")
        
        print(f"{req_info['user']}@{req_info['org']}: ${usage:.6f}")
    
    print()


async def demo_production_configuration():
    """Show a production-like configuration."""
    print("=== Production Configuration Demo ===")
    
    # Production-like constraints
    constraints = [
        # Free tier: $1/day per user
        Constraint(
            name="Free Tier Daily",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=RateLimit(per_minute=10),
            description="Free tier: $1/day, 10 req/min"
        ),
        # Organization limits
        Constraint(
            name="Organization Monthly", 
            dimensions=["organization"],
            budget_limit=BudgetLimit(limit=5000.0, window="30d"),
            rate_limit=None,
            description="Organization: $5000/month"
        ),
        # Team limits within organization
        Constraint(
            name="Team Weekly",
            dimensions=["organization", "team"],
            budget_limit=BudgetLimit(limit=500.0, window="7d"),
            rate_limit=RateLimit(per_minute=100),
            description="Team: $500/week, 100 req/min"
        )
    ]
    
    backend = InMemoryLimitBackend()
    
    # Build production-like pipeline
    pipeline = (PipelineBuilder()
        .add(AuthMiddleware(require_user_id=True))  # Strict auth
        .add(LimitEnforcementMiddleware(constraints, backend))
        .add(LoggingMiddleware())
        .set_monitoring_interval(3)  # Monitor every 3 chunks
        .build())
    
    print(f"✅ Production pipeline configured with {len(constraints)} constraints")
    print("Pipeline includes:")
    print("  - Strict authentication")
    print("  - Multi-tier budget enforcement")  
    print("  - Rate limiting")
    print("  - Observability logging")
    print("  - Streaming monitoring")
    
    # Test with a valid user
    request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Production test"}],
        "metadata": {
            "user_id": "prod_user",
            "organization": "acme_corp",
            "team": "engineering"
        }
    }
    
    ctx = create_request_context(request)
    result_ctx = await pipeline.execute(ctx)
    
    print(f"Production request result: {result_ctx.action}")
    if 'budget_tracking' in result_ctx.metadata:
        tracking = result_ctx.metadata['budget_tracking']
        print(f"Checked {tracking['constraints_checked']} constraints")
    
    print()


async def main():
    """Run all demos."""
    print("🚀 Limit Enforcement Middleware Demo\n")
    
    await demo_in_memory_backend()
    await demo_budget_exceeded()
    await demo_rate_limit_exceeded()
    await demo_redis_backend()
    await demo_multi_dimensional_constraints()
    await demo_production_configuration()
    
    print("✅ All demos completed!")


if __name__ == "__main__":
    asyncio.run(main())