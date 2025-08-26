#!/usr/bin/env python3
"""
Simple test to demonstrate limit middleware functionality without LLM calls.
"""

import asyncio
from simple_llm_proxy.middleware.limit import (
    LimitEnforcementMiddleware, 
    BudgetLimit, 
    RateLimit,
    Constraint,
    InMemoryLimitBackend
)
from simple_llm_proxy.pipeline import create_request_context, PipelineAction, LimitError


async def test_budget_enforcement():
    """Test budget limit enforcement."""
    print("=== Budget Enforcement Test ===")
    
    # Create a tight budget constraint
    constraint = Constraint(
        name="Daily Budget",
        dimensions=["user_id"],
        budget_limit=BudgetLimit(limit=0.001, window="1d"),  # Very tight
        rate_limit=None,
        description="$0.001 daily budget per user"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([constraint], backend)
    
    # Create test request
    request = {
        "model": "openai:gpt-4",
        "messages": [{"role": "user", "content": "This is a test request"}]
    }
    
    ctx = create_request_context(request)
    ctx.client_metadata['user_id'] = 'test_user'
    
    # Test the middleware
    print("Testing request with tight budget...")
    result_ctx = await middleware.before_llm(ctx)
    
    if result_ctx.action == PipelineAction.HALT:
        print("✅ Request blocked due to budget limit")
        if isinstance(result_ctx.response, LimitError):
            print(f"   Reason: {result_ctx.response.message}")
        else:
            print(f"   Reason: {result_ctx.response}")
    else:
        print("❌ Request allowed - budget limit not enforced")
    print()


async def test_rate_limiting():
    """Test rate limiting."""
    print("=== Rate Limiting Test ===")
    
    constraint = Constraint(
        name="Rate Limit",
        dimensions=["user_id"],
        budget_limit=None,
        rate_limit=RateLimit(per_minute=2),  # Only 2 requests per minute
        description="2 requests per minute per user"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([constraint], backend)
    
    request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Rate limit test"}]
    }
    
    print("Making 4 rapid requests...")
    
    for i in range(4):
        ctx = create_request_context(request)
        ctx.client_metadata['user_id'] = 'rate_test_user'
        
        result_ctx = await middleware.before_llm(ctx)
        
        if result_ctx.action == PipelineAction.CONTINUE:
            print(f"  Request {i+1}: ✅ Allowed")
            # Check current usage
            usage = await backend.get_rate_usage("user:rate_test_user")
            print(f"    Current usage: {usage}/2 requests/minute")
        else:
            print(f"  Request {i+1}: ❌ Blocked")
            if isinstance(result_ctx.response, LimitError):
                print(f"    Reason: {result_ctx.response.message}")
            else:
                print(f"    Reason: {result_ctx.response}")
    print()


async def test_usage_tracking():
    """Test usage tracking after simulated LLM call."""
    print("=== Usage Tracking Test ===")
    
    constraint = Constraint(
        name="Usage Tracking",
        dimensions=["user_id"],
        budget_limit=BudgetLimit(limit=10.0, window="1d"),
        rate_limit=None,
        description="$10 daily budget with usage tracking"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([constraint], backend)
    
    request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Usage tracking test"}]
    }
    
    ctx = create_request_context(request)
    ctx.client_metadata['user_id'] = 'tracking_user'
    
    # Check usage before
    usage_before = await backend.get_usage("user:tracking_user", "1d")
    print(f"Usage before: ${usage_before:.6f}")
    
    # Simulate middleware processing
    result_ctx = await middleware.before_llm(ctx)
    print(f"Request allowed: {result_ctx.action == PipelineAction.CONTINUE}")
    
    # Simulate LLM response
    ctx.response = {
        "usage": {"total_tokens": 500}
    }
    
    # Process after_llm
    await middleware.after_llm(ctx)
    
    # Check usage after
    usage_after = await backend.get_usage("user:tracking_user", "1d")
    print(f"Usage after: ${usage_after:.6f}")
    
    # Check tracking metadata
    if 'budget_tracking' in ctx.metadata:
        tracking = ctx.metadata['budget_tracking']
        print(f"Estimated cost: ${tracking['estimated_cost']:.6f}")
        print(f"Actual cost: ${tracking['actual_cost']:.6f}")
        print(f"Constraints checked: {tracking['constraints_checked']}")
    print()


async def test_multi_dimensional():
    """Test multi-dimensional constraints."""
    print("=== Multi-Dimensional Constraints Test ===")
    
    constraint = Constraint(
        name="User + Org Budget",
        dimensions=["user_id", "organization"],
        budget_limit=BudgetLimit(limit=5.0, window="1d"),
        rate_limit=None,
        description="$5 daily budget per user per organization"
    )
    
    backend = InMemoryLimitBackend()
    middleware = LimitEnforcementMiddleware([constraint], backend)
    
    # Test different user/org combinations
    test_cases = [
        {"user": "alice", "org": "engineering"},
        {"user": "alice", "org": "marketing"}, 
        {"user": "bob", "org": "engineering"}
    ]
    
    for case in test_cases:
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Multi-dimensional test"}]
        }
        
        ctx = create_request_context(request)
        ctx.client_metadata.update({
            'user_id': case["user"],
            'organization': case["org"]
        })
        
        # Process request
        await middleware.before_llm(ctx)
        
        # Simulate response and track usage
        ctx.response = {"usage": {"total_tokens": 200}}
        await middleware.after_llm(ctx)
        
        # Check usage for this specific combination
        key_parts = [f"org:{case['org']}", f"user:{case['user']}"]
        combined_key = "|".join(sorted(key_parts))
        usage = await backend.get_usage(combined_key, "1d")
        
        print(f"{case['user']}@{case['org']}: ${usage:.6f}")
    print()


async def main():
    """Run all tests."""
    print("🧪 Limit Middleware Functionality Test\n")
    
    await test_budget_enforcement()
    await test_rate_limiting()
    await test_usage_tracking()
    await test_multi_dimensional()
    
    print("✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())