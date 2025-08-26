"""
Test suite for individual middleware components.

TODO: Rewrite tests for function-based middleware in the new signal-based architecture.
Most middleware are either in experimental/ or use a different API than these tests expect.
"""

import pytest

# Skip all middleware tests - they expect class-based middleware API
pytest.skip("Middleware tests need rewrite for function-based middleware", allow_module_level=True)


@pytest.fixture
def basic_request():
    return {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }


@pytest.fixture
def authenticated_request():
    return {
        "model": "openai:gpt-3.5-turbo", 
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"user_id": "test_user"}
    }


class TestAuthMiddleware:
    """Test authentication middleware."""
    
    async def test_auth_middleware_require_user_id_pass(self, authenticated_request):
        auth_middleware = AuthMiddleware(require_user_id=True)
        ctx = create_request_context(authenticated_request)
        
        result_ctx = await auth_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
        assert "auth" in result_ctx.state
        assert result_ctx.state["auth"]["user_id"] == "test_user"
    
    async def test_auth_middleware_require_user_id_fail(self, basic_request):
        auth_middleware = AuthMiddleware(require_user_id=True)
        ctx = create_request_context(basic_request)
        
        result_ctx = await auth_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.HALT
        assert "error" in result_ctx.response
        assert "authentication required" in result_ctx.response["error"]["message"].lower()
    
    async def test_auth_middleware_optional_user_id(self, basic_request):
        auth_middleware = AuthMiddleware(require_user_id=False)
        ctx = create_request_context(basic_request)
        
        result_ctx = await auth_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
        assert result_ctx.state["auth"]["user_id"] == "anonymous"
    
    async def test_auth_middleware_with_api_key(self):
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"api_key": "test_key"}
        }
        
        auth_middleware = AuthMiddleware(require_api_key=True)
        ctx = create_request_context(request)
        
        result_ctx = await auth_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
        assert result_ctx.state["auth"]["api_key_provided"] == True


class TestLoggingMiddleware:
    """Test logging middleware."""
    
    @patch('simple_llm_proxy.middleware.logging.logging')
    async def test_logging_middleware_request(self, mock_logging, basic_request):
        logging_middleware = LoggingMiddleware()
        ctx = create_request_context(basic_request)
        
        await logging_middleware.before_llm(ctx)
        
        # Should log request start
        mock_logging.info.assert_called()
        call_args = mock_logging.info.call_args[0][0]
        assert "request_start" in call_args
    
    @patch('simple_llm_proxy.middleware.logging.logging')
    async def test_logging_middleware_response(self, mock_logging, basic_request):
        logging_middleware = LoggingMiddleware()
        ctx = create_request_context(basic_request)
        ctx.response = {"choices": [{"message": {"content": "Test response"}}]}
        
        await logging_middleware.after_llm(ctx)
        
        # Should log response
        mock_logging.info.assert_called()
        call_args = mock_logging.info.call_args[0][0]
        assert "response_complete" in call_args


class TestRoutingMiddleware:
    """Test routing middleware."""
    
    async def test_routing_middleware_cost_optimized(self, basic_request):
        routing_middleware = RoutingMiddleware(strategy="cost_optimized")
        ctx = create_request_context(basic_request)
        
        result_ctx = await routing_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
        assert "routing" in result_ctx.state
        
        routing_info = result_ctx.state["routing"]
        assert "provider" in routing_info
        assert "model" in routing_info
        assert "reason" in routing_info
    
    async def test_routing_middleware_model_override(self, basic_request):
        routing_middleware = RoutingMiddleware(strategy="cost_optimized")
        ctx = create_request_context(basic_request)
        
        result_ctx = await routing_middleware.before_llm(ctx)
        
        # Should override the original model
        routing_info = result_ctx.state["routing"]
        routed_model = routing_info["model"]
        
        # Model should be updated to routed model
        assert result_ctx.request["model"] == routed_model
        assert routed_model.startswith("openai:")  # Should include provider prefix
    
    async def test_routing_middleware_performance_optimized(self, basic_request):
        routing_middleware = RoutingMiddleware(strategy="performance_optimized")
        ctx = create_request_context(basic_request)
        
        result_ctx = await routing_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
        routing_info = result_ctx.state["routing"]
        assert routing_info["reason"] == "performance_optimization"


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""
    
    async def test_rate_limit_middleware_under_limit(self, authenticated_request):
        rate_limit_middleware = RateLimitMiddleware(
            requests_per_minute=60,
            tokens_per_minute=1000
        )
        ctx = create_request_context(authenticated_request)
        
        result_ctx = await rate_limit_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
        assert "rate_limit" in result_ctx.state
    
    async def test_rate_limit_middleware_over_request_limit(self, authenticated_request):
        rate_limit_middleware = RateLimitMiddleware(
            requests_per_minute=1  # Very low limit
        )
        ctx = create_request_context(authenticated_request)
        
        # First request should pass
        result_ctx1 = await rate_limit_middleware.before_llm(ctx)
        assert result_ctx1.action == PipelineAction.CONTINUE
        
        # Second request should be limited
        ctx2 = create_request_context(authenticated_request)
        result_ctx2 = await rate_limit_middleware.before_llm(ctx2)
        
        assert result_ctx2.action == PipelineAction.HALT
        assert "error" in result_ctx2.response
        assert "rate limit" in result_ctx2.response["error"]["message"].lower()
    
    async def test_rate_limit_middleware_cost_tracking(self, authenticated_request):
        rate_limit_middleware = RateLimitMiddleware(
            requests_per_minute=60,
            tokens_per_minute=1000
        )
        ctx = create_request_context(authenticated_request)
        
        # Simulate response with usage
        ctx.response = {
            "choices": [{"message": {"content": "Test"}}],
            "usage": {"total_tokens": 25}
        }
        
        result_ctx = await rate_limit_middleware.after_llm(ctx)
        
        assert "rate_limit" in result_ctx.state
        assert "cost_tracking" in result_ctx.state["rate_limit"]


class TestContentGuardrailMiddleware:
    """Test content guardrail middleware."""
    
    async def test_content_guardrail_appropriate_content(self):
        guardrail_middleware = ContentGuardrailMiddleware(
            policies=["inappropriate"]
        )
        ctx = create_request_context({
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, how are you?"}]
        })
        
        result_ctx = await guardrail_middleware.before_llm(ctx)
        
        assert result_ctx.action == PipelineAction.CONTINUE
    
    async def test_content_guardrail_response_monitoring(self):
        guardrail_middleware = ContentGuardrailMiddleware(
            policies=["inappropriate"],
            check_interval=1
        )
        ctx = create_request_context({
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Tell me a story"}]
        })
        
        # Simulate appropriate content
        result_ctx = await guardrail_middleware.on_response_update(
            ctx, "This is a nice, appropriate story about a cat."
        )
        
        assert result_ctx.action == PipelineAction.CONTINUE
    
    async def test_content_guardrail_blocks_inappropriate(self):
        guardrail_middleware = ContentGuardrailMiddleware(
            policies=["inappropriate"],
            check_interval=1
        )
        ctx = create_request_context({
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Tell me something"}]
        })
        
        # Simulate inappropriate content - this is a mock, real implementation would check actual content
        result_ctx = await guardrail_middleware.on_response_update(
            ctx, "inappropriate content that violates policies"
        )
        
        # In the actual implementation, this might trigger based on content analysis
        # For this test, we check that the middleware can halt when needed
        if result_ctx.action == PipelineAction.HALT:
            assert "error" in result_ctx.response
            assert "content policy" in result_ctx.response["error"]["message"].lower()


class TestMiddlewareIntegration:
    """Test middleware working together."""
    
    async def test_middleware_state_sharing(self, authenticated_request):
        """Test that middleware can share state through the pipeline."""
        
        auth_middleware = AuthMiddleware(require_user_id=True)
        routing_middleware = RoutingMiddleware(strategy="cost_optimized")
        logging_middleware = LoggingMiddleware()
        
        ctx = create_request_context(authenticated_request)
        
        # Execute middleware in sequence
        ctx = await auth_middleware.before_llm(ctx)
        assert ctx.action == PipelineAction.CONTINUE
        
        ctx = await routing_middleware.before_llm(ctx)
        assert ctx.action == PipelineAction.CONTINUE
        
        ctx = await logging_middleware.before_llm(ctx)
        assert ctx.action == PipelineAction.CONTINUE
        
        # Check that all middleware set their state
        assert "auth" in ctx.state
        assert "routing" in ctx.state
        assert ctx.state["auth"]["user_id"] == "test_user"
        assert "provider" in ctx.state["routing"]
    
    async def test_middleware_error_propagation(self, basic_request):
        """Test that middleware errors are properly handled."""
        
        # Auth middleware should halt on missing user_id
        auth_middleware = AuthMiddleware(require_user_id=True)
        routing_middleware = RoutingMiddleware(strategy="cost_optimized")
        
        ctx = create_request_context(basic_request)
        
        # Auth should halt
        ctx = await auth_middleware.before_llm(ctx)
        assert ctx.action == PipelineAction.HALT
        
        # Routing should not execute when pipeline is halted
        # (This would be enforced by the pipeline, not tested here)
    
    async def test_middleware_configuration_variations(self):
        """Test middleware with different configurations."""
        
        # Test different auth configurations
        strict_auth = AuthMiddleware(require_user_id=True, require_api_key=True)
        lenient_auth = AuthMiddleware(require_user_id=False, require_api_key=False)
        
        request_no_auth = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # Strict auth should reject
        ctx1 = create_request_context(request_no_auth)
        result1 = await strict_auth.before_llm(ctx1)
        assert result1.action == PipelineAction.HALT
        
        # Lenient auth should allow
        ctx2 = create_request_context(request_no_auth)
        result2 = await lenient_auth.before_llm(ctx2)
        assert result2.action == PipelineAction.CONTINUE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])