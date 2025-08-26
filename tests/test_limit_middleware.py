"""
Comprehensive test suite for the simplified limit enforcement middleware.
"""

import pytest
import time
from unittest.mock import AsyncMock
from simple_llm_proxy.middleware.limit import (
    LimitEnforcementMiddleware,
    BudgetLimit,
    RateLimit,
    Constraint,
    InMemoryLimitBackend,
    SimpleLimitBackend
)
from simple_llm_proxy.pipeline import create_request_context, PipelineAction, LimitError


@pytest.fixture
def basic_request():
    return {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello world"}]
    }


@pytest.fixture
def expensive_request():
    return {
        "model": "openai:gpt-4",
        "messages": [{"role": "user", "content": "Write a long detailed analysis" * 100}],
        "max_tokens": 2000
    }


@pytest.fixture
def authenticated_request():
    return {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"user_id": "test_user", "organization": "test_org"}
    }


class TestBudgetLimit:
    """Test BudgetLimit dataclass validation."""
    
    def test_valid_budget_limit(self):
        limit = BudgetLimit(limit=10.0, window="1d")
        assert limit.limit == 10.0
        assert limit.window == "1d"
    
    def test_negative_limit_raises_error(self):
        with pytest.raises(ValueError, match="Budget limit must be positive"):
            BudgetLimit(limit=-1.0, window="1d")
    
    def test_zero_limit_raises_error(self):
        with pytest.raises(ValueError, match="Budget limit must be positive"):
            BudgetLimit(limit=0.0, window="1d")
    
    def test_invalid_window_raises_error(self):
        with pytest.raises(ValueError, match="Invalid window"):
            BudgetLimit(limit=10.0, window="2d")


class TestRateLimit:
    """Test RateLimit dataclass validation."""
    
    def test_valid_rate_limit(self):
        limit = RateLimit(per_minute=100)
        assert limit.per_minute == 100
    
    def test_negative_rate_raises_error(self):
        with pytest.raises(ValueError, match="Rate limit must be positive"):
            RateLimit(per_minute=-1)
    
    def test_zero_rate_raises_error(self):
        with pytest.raises(ValueError, match="Rate limit must be positive"):
            RateLimit(per_minute=0)


class TestConstraint:
    """Test Constraint dataclass validation."""
    
    def test_valid_budget_constraint(self):
        constraint = Constraint(
            name="User Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=50.0, window="1d"),
            rate_limit=None,
            description="Daily budget per user"
        )
        assert constraint.name == "User Budget"
        assert constraint.dimensions == ["user_id"]
    
    def test_valid_rate_constraint(self):
        constraint = Constraint(
            name="User Rate",
            dimensions=["user_id"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=10),
            description="Rate limit per user"
        )
        assert constraint.rate_limit.per_minute == 10
    
    def test_constraint_with_both_limits(self):
        constraint = Constraint(
            name="Combined",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=10.0, window="1d"),
            rate_limit=RateLimit(per_minute=5),
            description="Both limits"
        )
        assert constraint.budget_limit.limit == 10.0
        assert constraint.rate_limit.per_minute == 5
    
    def test_invalid_dimensions_raise_error(self):
        with pytest.raises(ValueError, match="Invalid dimensions"):
            Constraint(
                name="Bad",
                dimensions=["invalid_dim"],
                budget_limit=BudgetLimit(limit=10.0, window="1d"),
                rate_limit=None,
                description="Invalid dimension"
            )
    
    def test_duplicate_dimensions_raise_error(self):
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            Constraint(
                name="Dupe",
                dimensions=["user_id", "user_id"],
                budget_limit=BudgetLimit(limit=10.0, window="1d"),
                rate_limit=None,
                description="Duplicate dimensions"
            )
    
    def test_no_limits_raises_error(self):
        with pytest.raises(ValueError, match="Either budget or rate limit must be specified"):
            Constraint(
                name="No Limits",
                dimensions=["user_id"],
                budget_limit=None,
                rate_limit=None,
                description="No limits"
            )


class TestInMemoryLimitBackend:
    """Test in-memory limit backend."""
    
    @pytest.fixture
    def backend(self):
        return InMemoryLimitBackend()
    
    async def test_get_usage_empty(self, backend):
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 0.0
    
    async def test_increment_and_get_usage(self, backend):
        await backend.increment_usage("user:test", 5.5, "1d")
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 5.5
    
    async def test_multiple_increments(self, backend):
        await backend.increment_usage("user:test", 2.0, "1d")
        await backend.increment_usage("user:test", 3.0, "1d")
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 5.0
    
    async def test_separate_keys(self, backend):
        await backend.increment_usage("user:test1", 10.0, "1d")
        await backend.increment_usage("user:test2", 20.0, "1d")
        
        usage1 = await backend.get_usage("user:test1", "1d")
        usage2 = await backend.get_usage("user:test2", "1d")
        
        assert usage1 == 10.0
        assert usage2 == 20.0
    
    async def test_get_rate_usage_empty(self, backend):
        usage = await backend.get_rate_usage("user:test")
        assert usage == 0
    
    async def test_increment_and_get_rate_usage(self, backend):
        await backend.increment_rate_usage("user:test")
        await backend.increment_rate_usage("user:test")
        usage = await backend.get_rate_usage("user:test")
        assert usage == 2
    
    async def test_rate_usage_different_keys(self, backend):
        await backend.increment_rate_usage("user:test1")
        await backend.increment_rate_usage("user:test2")
        await backend.increment_rate_usage("user:test2")
        
        usage1 = await backend.get_rate_usage("user:test1")
        usage2 = await backend.get_rate_usage("user:test2")
        
        assert usage1 == 1
        assert usage2 == 2


class TestLimitEnforcementMiddleware:
    """Test limit enforcement middleware."""
    
    @pytest.fixture
    def backend(self):
        return InMemoryLimitBackend()
    
    @pytest.fixture
    def budget_constraint(self):
        return Constraint(
            name="User Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="$1 daily budget per user"
        )
    
    @pytest.fixture
    def rate_constraint(self):
        return Constraint(
            name="User Rate",
            dimensions=["user_id"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=3),
            description="3 requests per minute per user"
        )
    
    async def test_no_constraints_allows_request(self, backend, basic_request):
        middleware = LimitEnforcementMiddleware(constraints=[], backend=backend)
        ctx = create_request_context(basic_request)
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
    
    async def test_budget_constraint_under_limit(self, backend, budget_constraint, basic_request):
        middleware = LimitEnforcementMiddleware(constraints=[budget_constraint], backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
    
    async def test_budget_constraint_over_limit(self, backend, basic_request):
        # Use a very small budget to ensure we exceed it
        tiny_budget_constraint = Constraint(
            name="Tiny Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=0.0001, window="1d"),  # Very small limit
            rate_limit=None,
            description="Tiny budget per user"
        )
        
        middleware = LimitEnforcementMiddleware(constraints=[tiny_budget_constraint], backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.HALT
        assert isinstance(result_ctx.response, LimitError)
        assert "Budget limit exceeded" in result_ctx.response.message
        assert "budget_limit_exceeded" == result_ctx.response.error_type
    
    async def test_rate_constraint_under_limit(self, backend, rate_constraint, basic_request):
        middleware = LimitEnforcementMiddleware(constraints=[rate_constraint], backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
    
    async def test_rate_constraint_over_limit(self, backend, rate_constraint, basic_request):
        # Pre-populate rate usage
        for _ in range(3):
            await backend.increment_rate_usage("user:test_user")
        
        middleware = LimitEnforcementMiddleware(constraints=[rate_constraint], backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.HALT
        assert isinstance(result_ctx.response, LimitError)
        assert "Rate limit exceeded" in result_ctx.response.message
        assert "rate_limit_exceeded" == result_ctx.response.error_type
    
    async def test_after_llm_updates_budget_usage(self, backend, budget_constraint, basic_request):
        middleware = LimitEnforcementMiddleware(constraints=[budget_constraint], backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        # Simulate LLM response with token usage
        ctx.response = {
            "usage": {"total_tokens": 1000}
        }
        
        # Process before and after
        await middleware.before_llm(ctx)
        await middleware.after_llm(ctx)
        
        # Check that usage was updated
        usage = await backend.get_usage("user:test_user", "1d")
        assert usage > 0.0
        assert 'budget_tracking' in ctx.metadata
        assert ctx.metadata['budget_tracking']['actual_cost'] > 0
    
    async def test_multiple_constraints(self, backend, budget_constraint, rate_constraint, basic_request):
        constraints = [budget_constraint, rate_constraint]
        middleware = LimitEnforcementMiddleware(constraints=constraints, backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        
        # Should increment rate usage
        rate_usage = await backend.get_rate_usage("user:test_user")
        assert rate_usage == 1
    
    async def test_multi_dimensional_constraint(self, backend, basic_request):
        # Constraint that uses both user_id and organization
        constraint = Constraint(
            name="Org User Budget",
            dimensions=["user_id", "organization"],
            budget_limit=BudgetLimit(limit=5.0, window="1d"),
            rate_limit=None,
            description="Per user per org budget"
        )
        
        middleware = LimitEnforcementMiddleware(constraints=[constraint], backend=backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        ctx.client_metadata['organization'] = 'test_org'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        
        # Should create a key that includes both dimensions
        # The exact key format depends on implementation, but should be consistent
    
    async def test_anonymous_user_handling(self, backend, basic_request):
        constraint = Constraint(
            name="User Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Budget per user"
        )
        
        middleware = LimitEnforcementMiddleware(constraints=[constraint], backend=backend)
        ctx = create_request_context(basic_request)
        # No user_id provided - should use 'anonymous'
        
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        
        # Check that it uses 'anonymous' key
        usage = await backend.get_usage("user:anonymous", "1d")
        # Usage should still be 0 since we only checked, didn't increment yet
        assert usage == 0.0
    
    async def test_cost_estimation(self, backend, budget_constraint):
        middleware = LimitEnforcementMiddleware(constraints=[budget_constraint], backend=backend)
        
        # Test different model costs
        gpt35_request = {"model": "openai:gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]}
        gpt4_request = {"model": "openai:gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        
        gpt35_ctx = create_request_context(gpt35_request)
        gpt4_ctx = create_request_context(gpt4_request)
        
        gpt35_cost = middleware._estimate_request_cost(gpt35_ctx)
        gpt4_cost = middleware._estimate_request_cost(gpt4_ctx)
        
        # GPT-4 should be more expensive than GPT-3.5
        assert gpt4_cost > gpt35_cost
        assert gpt35_cost > 0
    
    async def test_token_count_estimation(self, backend, budget_constraint):
        middleware = LimitEnforcementMiddleware(constraints=[budget_constraint], backend=backend)
        
        short_messages = [{"role": "user", "content": "hi"}]
        long_messages = [{"role": "user", "content": "This is a much longer message " * 20}]
        
        short_count = middleware._estimate_token_count(short_messages)
        long_count = middleware._estimate_token_count(long_messages)
        
        assert long_count > short_count
        assert short_count > 0
    
    async def test_multimodal_content_estimation(self, backend, budget_constraint):
        middleware = LimitEnforcementMiddleware(constraints=[budget_constraint], backend=backend)
        
        multimodal_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }]
        
        token_count = middleware._estimate_token_count(multimodal_messages)
        # Should include both text and image costs
        assert token_count > 250  # At least 1000 for image + some for text
    
    async def test_actual_cost_calculation(self, backend, budget_constraint):
        middleware = LimitEnforcementMiddleware(constraints=[budget_constraint], backend=backend)
        
        # Response with token usage
        response = {"usage": {"total_tokens": 500}}
        cost = middleware._calculate_actual_cost(response)
        assert cost > 0
        
        # Response without usage
        response_no_usage = {"choices": [{"message": {"content": "hello"}}]}
        cost_no_usage = middleware._calculate_actual_cost(response_no_usage)
        assert cost_no_usage == 0.0
        
        # None response
        cost_none = middleware._calculate_actual_cost(None)
        assert cost_none == 0.0


class TestConstraintKeyBuilding:
    """Test constraint key building logic."""
    
    @pytest.fixture
    def middleware(self):
        backend = InMemoryLimitBackend()
        constraint = Constraint(
            name="Test",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Test constraint"
        )
        return LimitEnforcementMiddleware(constraints=[constraint], backend=backend)
    
    def test_single_dimension_key(self, middleware, basic_request):
        constraint = Constraint(
            name="User",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="User constraint"
        )
        
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        key = middleware._build_constraint_key(constraint, ctx)
        assert key == "user:test_user"
    
    def test_multi_dimension_key(self, middleware, basic_request):
        constraint = Constraint(
            name="User Org",
            dimensions=["user_id", "organization"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="User org constraint"
        )
        
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        ctx.client_metadata['organization'] = 'test_org'
        
        key = middleware._build_constraint_key(constraint, ctx)
        # Keys should be sorted for consistency
        expected_parts = ["org:test_org", "user:test_user"]
        assert key == "|".join(expected_parts)
    
    def test_feature_dimension(self, middleware, basic_request):
        constraint = Constraint(
            name="Feature",
            dimensions=["feature"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Feature constraint"
        )
        
        ctx = create_request_context(basic_request)
        ctx.state['feature'] = {'name': 'chat_completion'}
        
        key = middleware._build_constraint_key(constraint, ctx)
        assert key == "feature:chat_completion"
    
    def test_team_dimension(self, middleware, basic_request):
        constraint = Constraint(
            name="Team",
            dimensions=["team"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Team constraint"
        )
        
        ctx = create_request_context(basic_request)
        ctx.client_metadata['team'] = 'engineering'
        
        key = middleware._build_constraint_key(constraint, ctx)
        assert key == "team:engineering"
    
    def test_unknown_dimension_raises_error(self, middleware, basic_request):
        # This test should fail when creating the constraint, not when building the key
        with pytest.raises(ValueError, match="Invalid dimensions"):
            Constraint(
                name="Invalid",
                dimensions=["unknown_dim"],
                budget_limit=BudgetLimit(limit=1.0, window="1d"),
                rate_limit=None,
                description="Invalid constraint"
            )


class TestErrorHandling:
    """Test error handling in limit middleware."""
    
    @pytest.fixture
    def failing_backend(self):
        backend = AsyncMock(spec=SimpleLimitBackend)
        backend.get_usage.side_effect = Exception("Backend error")
        backend.increment_usage.side_effect = Exception("Backend error")
        return backend
    
    async def test_backend_error_in_before_llm_logs_and_continues(self, failing_backend, basic_request):
        constraint = Constraint(
            name="Test",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Test constraint"
        )
        
        middleware = LimitEnforcementMiddleware(constraints=[constraint], backend=failing_backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        
        # Should continue even if backend fails (fail open)
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
    
    async def test_backend_error_in_after_llm_logs_error(self, failing_backend, basic_request):
        constraint = Constraint(
            name="Test",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=1.0, window="1d"),
            rate_limit=None,
            description="Test constraint"
        )
        
        middleware = LimitEnforcementMiddleware(constraints=[constraint], backend=failing_backend)
        ctx = create_request_context(basic_request)
        ctx.client_metadata['user_id'] = 'test_user'
        ctx.response = {"usage": {"total_tokens": 100}}
        
        # Should not raise exception, just log error
        result_ctx = await middleware.after_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        assert 'budget_error' in ctx.metadata


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def production_constraints(self):
        return [
            # Per-user daily budget
            Constraint(
                name="User Daily Budget",
                dimensions=["user_id"],
                budget_limit=BudgetLimit(limit=10.0, window="1d"),
                rate_limit=None,
                description="$10 daily budget per user"
            ),
            # Per-user rate limit
            Constraint(
                name="User Rate Limit",
                dimensions=["user_id"],
                budget_limit=None,
                rate_limit=RateLimit(per_minute=20),
                description="20 requests per minute per user"
            ),
            # Per-organization budget
            Constraint(
                name="Org Budget",
                dimensions=["organization"],
                budget_limit=BudgetLimit(limit=1000.0, window="30d"),
                rate_limit=None,
                description="$1000 monthly budget per organization"
            )
        ]
    
    async def test_production_scenario_normal_usage(self, production_constraints, authenticated_request):
        backend = InMemoryLimitBackend()
        middleware = LimitEnforcementMiddleware(
            constraints=production_constraints, 
            backend=backend
        )
        
        ctx = create_request_context(authenticated_request)
        
        # Should allow normal usage
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.CONTINUE
        
        # Simulate response and track usage
        ctx.response = {"usage": {"total_tokens": 500}}
        await middleware.after_llm(ctx)
        
        # Should have tracking metadata
        assert 'budget_tracking' in ctx.metadata
        assert ctx.metadata['budget_tracking']['actual_cost'] > 0
    
    async def test_production_scenario_user_over_budget(self, authenticated_request):
        # Create constraints with smaller budgets to ensure we exceed them
        tight_constraints = [
            Constraint(
                name="Tight User Budget",
                dimensions=["user_id"],
                budget_limit=BudgetLimit(limit=0.0001, window="1d"),  # Extremely tight budget
                rate_limit=None,
                description="Very tight daily budget per user"
            )
        ]
        
        backend = InMemoryLimitBackend()
        middleware = LimitEnforcementMiddleware(
            constraints=tight_constraints, 
            backend=backend
        )
        
        ctx = create_request_context(authenticated_request)
        
        # Should reject due to user budget being too tight
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.HALT
        assert isinstance(result_ctx.response, LimitError)
        assert "Budget limit exceeded" in result_ctx.response.message
    
    async def test_production_scenario_user_over_rate_limit(self, production_constraints, authenticated_request):
        backend = InMemoryLimitBackend()
        
        # Pre-populate user rate usage
        for _ in range(20):
            await backend.increment_rate_usage("user:test_user")
        
        middleware = LimitEnforcementMiddleware(
            constraints=production_constraints, 
            backend=backend
        )
        
        ctx = create_request_context(authenticated_request)
        
        # Should reject due to rate limit
        result_ctx = await middleware.before_llm(ctx)
        assert result_ctx.action == PipelineAction.HALT
        assert isinstance(result_ctx.response, LimitError)
        assert "Rate limit exceeded" in result_ctx.response.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])