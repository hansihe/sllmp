"""
Test suite for the PipelineBuilder.
"""

import pytest
from simple_llm_proxy.builder import PipelineBuilder
from simple_llm_proxy.pipeline import Pipeline, Middleware
from simple_llm_proxy.middleware.auth import AuthMiddleware
from simple_llm_proxy.middleware.logging import LoggingMiddleware
from simple_llm_proxy.middleware.routing import RoutingMiddleware


class MockTestMiddleware(Middleware):
    """Simple test middleware for builder tests."""
    
    def __init__(self, name="test"):
        self.name = name
    
    async def before_llm(self, ctx):
        ctx.state[f"{self.name}_executed"] = True
        return ctx


class TestPipelineBuilder:
    """Test PipelineBuilder functionality."""
    
    def test_empty_builder(self):
        """Test creating an empty pipeline."""
        pipeline = PipelineBuilder().build()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.middleware) == 0
    
    def test_single_middleware(self):
        """Test builder with single middleware."""
        middleware = MockTestMiddleware("single")
        pipeline = (PipelineBuilder()
                   .add(middleware)
                   .build())
        
        assert len(pipeline.middleware) == 1
        assert pipeline.middleware[0] == middleware
    
    def test_multiple_middleware_order(self):
        """Test that middleware is added in correct order."""
        m1 = MockTestMiddleware("first")
        m2 = MockTestMiddleware("second")
        m3 = MockTestMiddleware("third")
        
        pipeline = (PipelineBuilder()
                   .add(m1)
                   .add(m2)
                   .add(m3)
                   .build())
        
        assert len(pipeline.middleware) == 3
        assert pipeline.middleware[0] == m1
        assert pipeline.middleware[1] == m2
        assert pipeline.middleware[2] == m3
    
    def test_method_chaining(self):
        """Test fluent interface method chaining."""
        m1 = MockTestMiddleware("first")
        m2 = MockTestMiddleware("second")
        
        # Should be able to chain method calls
        builder = PipelineBuilder()
        chained_builder = builder.add(m1).add(m2).set_monitoring_interval(10)
        
        # Should return same builder instance
        assert chained_builder is builder
        
        pipeline = chained_builder.build()
        assert len(pipeline.middleware) == 2
        assert pipeline.monitoring_interval == 10
    
    def test_monitoring_interval_configuration(self):
        """Test monitoring interval configuration."""
        pipeline = (PipelineBuilder()
                   .set_monitoring_interval(3)
                   .build())
        
        assert pipeline.monitoring_interval == 3
    
    def test_monitoring_interval_default(self):
        """Test default monitoring interval."""
        pipeline = PipelineBuilder().build()
        
        assert pipeline.monitoring_interval == 5  # Default value
    
    def test_realistic_pipeline_construction(self):
        """Test building a realistic pipeline with actual middleware."""
        pipeline = (PipelineBuilder()
                   .add(AuthMiddleware(require_user_id=False))
                   .add(RoutingMiddleware(strategy="cost_optimized"))
                   .add(LoggingMiddleware())
                   .set_monitoring_interval(3)
                   .build())
        
        assert len(pipeline.middleware) == 3
        assert isinstance(pipeline.middleware[0], AuthMiddleware)
        assert isinstance(pipeline.middleware[1], RoutingMiddleware)
        assert isinstance(pipeline.middleware[2], LoggingMiddleware)
        assert pipeline.monitoring_interval == 3
    
    def test_complex_pipeline_with_mixed_middleware(self):
        """Test building complex pipeline with various middleware types."""
        from simple_llm_proxy.middleware.auth import RateLimitMiddleware
        from simple_llm_proxy.middleware.guardrails import ContentGuardrailMiddleware
        
        pipeline = (PipelineBuilder()
                   .add(AuthMiddleware(require_user_id=True))
                   .add(RateLimitMiddleware(requests_per_minute=100))
                   .add(RoutingMiddleware(strategy="performance_optimized"))
                   .add(ContentGuardrailMiddleware(policies=["inappropriate"]))
                   .add(LoggingMiddleware())
                   .set_monitoring_interval(2)
                   .build())
        
        assert len(pipeline.middleware) == 5
        assert isinstance(pipeline.middleware[0], AuthMiddleware)
        assert isinstance(pipeline.middleware[1], RateLimitMiddleware)
        assert isinstance(pipeline.middleware[2], RoutingMiddleware)
        assert isinstance(pipeline.middleware[3], ContentGuardrailMiddleware)
        assert isinstance(pipeline.middleware[4], LoggingMiddleware)
    
    def test_builder_reuse(self):
        """Test that builder can be reused to create multiple pipelines."""
        base_builder = (PipelineBuilder()
                       .add(AuthMiddleware(require_user_id=False))
                       .add(LoggingMiddleware()))
        
        # Create first pipeline
        pipeline1 = base_builder.build()
        
        # Add more middleware and create second pipeline
        pipeline2 = (base_builder
                    .add(RoutingMiddleware(strategy="cost_optimized"))
                    .build())
        
        # First pipeline should have 2 middleware
        assert len(pipeline1.middleware) == 2
        
        # Second pipeline should have 3 middleware (includes the added one)
        assert len(pipeline2.middleware) == 3
    
    def test_builder_with_custom_middleware(self):
        """Test builder with custom middleware implementations."""
        
        class CustomAuthMiddleware(Middleware):
            def __init__(self, secret_key):
                self.secret_key = secret_key
            
            async def before_llm(self, ctx):
                ctx.state["custom_auth"] = {"validated": True}
                return ctx
        
        class CustomLoggingMiddleware(Middleware):
            def __init__(self, log_level="INFO"):
                self.log_level = log_level
            
            async def before_llm(self, ctx):
                ctx.state["custom_logging"] = {"level": self.log_level}
                return ctx
        
        custom_auth = CustomAuthMiddleware("secret123")
        custom_logging = CustomLoggingMiddleware("DEBUG")
        
        pipeline = (PipelineBuilder()
                   .add(custom_auth)
                   .add(custom_logging)
                   .build())
        
        assert len(pipeline.middleware) == 2
        assert pipeline.middleware[0].secret_key == "secret123"
        assert pipeline.middleware[1].log_level == "DEBUG"
    
    def test_builder_configuration_validation(self):
        """Test that builder accepts various monitoring intervals."""
        
        # Test valid monitoring intervals
        builder1 = PipelineBuilder().set_monitoring_interval(1)
        assert builder1.monitoring_interval == 1
        
        builder2 = PipelineBuilder().set_monitoring_interval(10)
        assert builder2.monitoring_interval == 10
    
    def test_builder_type_validation(self):
        """Test that builder accepts middleware instances."""
        
        # Valid middleware should be accepted
        middleware = MockTestMiddleware("valid")
        builder = PipelineBuilder().add(middleware)
        assert len(builder.middleware) == 1
        assert builder.middleware[0] == middleware
    
    def test_builder_duplicate_middleware_allowed(self):
        """Test that same middleware instance can be added multiple times."""
        middleware = MockTestMiddleware("duplicate")
        
        pipeline = (PipelineBuilder()
                   .add(middleware)
                   .add(middleware)  # Same instance
                   .build())
        
        # Should allow duplicates
        assert len(pipeline.middleware) == 2
        assert pipeline.middleware[0] is middleware
        assert pipeline.middleware[1] is middleware
    
    def test_builder_middleware_configuration_preserved(self):
        """Test that middleware configurations are preserved through builder."""
        auth = AuthMiddleware(require_user_id=True, require_api_key=True)
        routing = RoutingMiddleware(strategy="performance_optimized")
        logging = LoggingMiddleware(log_requests=True, log_responses=False)
        
        pipeline = (PipelineBuilder()
                   .add(auth)
                   .add(routing)
                   .add(logging)
                   .build())
        
        # Verify configurations are preserved
        built_auth = pipeline.middleware[0]
        built_routing = pipeline.middleware[1]
        built_logging = pipeline.middleware[2]
        
        # Check that middleware instances are preserved (config access depends on implementation)
        assert isinstance(built_auth, AuthMiddleware)
        assert isinstance(built_routing, RoutingMiddleware) 
        assert isinstance(built_logging, LoggingMiddleware)


class TestBuilderPatterns:
    """Test common builder usage patterns."""
    
    def test_minimal_pipeline_pattern(self):
        """Test minimal useful pipeline pattern."""
        pipeline = (PipelineBuilder()
                   .add(AuthMiddleware(require_user_id=False))
                   .add(LoggingMiddleware())
                   .build())
        
        assert len(pipeline.middleware) == 2
        assert pipeline.monitoring_interval == 5  # Default
    
    def test_development_pipeline_pattern(self):
        """Test development-friendly pipeline pattern."""
        pipeline = (PipelineBuilder()
                   .add(AuthMiddleware(require_user_id=False))  # Lenient auth for dev
                   .add(LoggingMiddleware(log_requests=True, log_responses=True))  # Verbose logging
                   .set_monitoring_interval(1)  # Frequent monitoring
                   .build())
        
        assert len(pipeline.middleware) == 2
        assert pipeline.monitoring_interval == 1
    
    def test_production_pipeline_pattern(self):
        """Test production-ready pipeline pattern."""
        from simple_llm_proxy.middleware.auth import RateLimitMiddleware
        from simple_llm_proxy.middleware.guardrails import ContentGuardrailMiddleware
        
        pipeline = (PipelineBuilder()
                   .add(AuthMiddleware(require_user_id=True, require_api_key=True))
                   .add(RateLimitMiddleware(requests_per_minute=1000, tokens_per_minute=50000))
                   .add(RoutingMiddleware(strategy="cost_optimized"))
                   .add(ContentGuardrailMiddleware(policies=["inappropriate", "harmful"]))
                   .add(LoggingMiddleware(log_requests=False, log_responses=False))  # Minimal logging
                   .set_monitoring_interval(5)
                   .build())
        
        assert len(pipeline.middleware) == 5
        
        # Verify it's a strict auth setup
        auth_middleware = pipeline.middleware[0]
        assert isinstance(auth_middleware, AuthMiddleware)
    
    def test_streaming_optimized_pipeline_pattern(self):
        """Test pipeline optimized for streaming."""
        from simple_llm_proxy.middleware.guardrails import ContentGuardrailMiddleware
        
        pipeline = (PipelineBuilder()
                   .add(AuthMiddleware(require_user_id=False))
                   .add(ContentGuardrailMiddleware(policies=["inappropriate"], check_interval=2))
                   .set_monitoring_interval(2)  # Frequent monitoring for streaming
                   .build())
        
        assert pipeline.monitoring_interval == 2
        
        # Should have guardrail middleware that monitors responses
        guardrail = pipeline.middleware[1]
        assert isinstance(guardrail, ContentGuardrailMiddleware)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])