"""
Integration tests for the rate limit middleware.

These tests validate the limit_enforcement_middleware with the full HTTP server flow,
including budget limits, rate limits, and proper error responses.
"""

import pytest
import httpx
import asyncio
from unittest.mock import patch

from sllmp import SimpleProxyServer
from sllmp.context import Pipeline, RequestContext
from sllmp.middleware.limit import (
    limit_enforcement_middleware,
    BudgetLimit,
    RateLimit,
    Constraint,
    InMemoryLimitBackend,
)
from any_llm.types.completion import ChatCompletion


@pytest.fixture
def mock_llm_completion():
    """Mock any_llm.acompletion for integration tests."""

    def create_completion(**kwargs):
        model_id = kwargs.get("model", kwargs.get("model_id", "openai:gpt-3.5-turbo"))
        return ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model=model_id,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def mock_completion(stream=False, **kwargs):
        return create_completion(**kwargs)

    with patch("sllmp.pipeline.any_llm.acompletion", side_effect=mock_completion):
        yield


@pytest.fixture
def backend():
    """Create a fresh InMemoryLimitBackend for each test."""
    return InMemoryLimitBackend()


@pytest.fixture
def rate_limit_constraint():
    """Constraint with rate limit of 3 requests per minute."""
    return Constraint(
        name="User Rate Limit",
        dimensions=["user_id"],
        budget_limit=None,
        rate_limit=RateLimit(per_minute=3),
        description="3 requests per minute per user",
    )


@pytest.fixture
def budget_limit_constraint():
    """Constraint with a budget limit."""
    return Constraint(
        name="User Budget",
        dimensions=["user_id"],
        budget_limit=BudgetLimit(limit=1.0, window="1d"),
        rate_limit=None,
        description="$1 daily budget per user",
    )


@pytest.fixture
def tiny_budget_constraint():
    """Constraint with a very small budget that will be exceeded immediately."""
    return Constraint(
        name="Tiny Budget",
        dimensions=["user_id"],
        budget_limit=BudgetLimit(limit=0.0001, window="1d"),
        rate_limit=None,
        description="Tiny budget to trigger limit exceeded",
    )


class TestRateLimitIntegration:
    """Integration tests for rate limiting via HTTP."""

    async def test_requests_under_rate_limit_succeed(
        self, mock_llm_completion, backend, rate_limit_constraint
    ):
        """Requests under the rate limit should succeed."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[rate_limit_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Send 3 requests (at the limit)
            for i in range(3):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "metadata": {"user_id": "test_user"},
                    },
                )
                assert response.status_code == 200, f"Request {i} failed: {response.text}"

    async def test_requests_over_rate_limit_rejected(
        self, mock_llm_completion, backend, rate_limit_constraint
    ):
        """Requests over the rate limit should be rejected with 429."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[rate_limit_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Send 3 requests (at the limit) - should all succeed
            for i in range(3):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "metadata": {"user_id": "test_user"},
                    },
                )
                assert response.status_code == 200

            # 4th request should be rate limited
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Excess request"}],
                    "metadata": {"user_id": "test_user"},
                },
            )
            assert response.status_code == 429

            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "rate_limit_exceeded"
            assert "Rate limit exceeded" in data["error"]["message"]

    async def test_rate_limit_per_user_isolation(
        self, mock_llm_completion, backend, rate_limit_constraint
    ):
        """Rate limits should be tracked separately per user."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[rate_limit_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Exhaust rate limit for user_a
            for i in range(3):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "metadata": {"user_id": "user_a"},
                    },
                )
                assert response.status_code == 200

            # user_a should be rate limited now
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Excess"}],
                    "metadata": {"user_id": "user_a"},
                },
            )
            assert response.status_code == 429

            # user_b should still be able to make requests
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "From user B"}],
                    "metadata": {"user_id": "user_b"},
                },
            )
            assert response.status_code == 200


class TestBudgetLimitIntegration:
    """Integration tests for budget limiting via HTTP."""

    async def test_requests_under_budget_succeed(
        self, mock_llm_completion, backend, budget_limit_constraint
    ):
        """Requests under the budget limit should succeed."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[budget_limit_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "test_user"},
                },
            )
            assert response.status_code == 200

    async def test_requests_over_budget_rejected(
        self, mock_llm_completion, backend, tiny_budget_constraint
    ):
        """Requests that would exceed the budget should be rejected."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[tiny_budget_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Request should be rejected immediately due to tiny budget
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "test_user"},
                },
            )
            assert response.status_code == 429

            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "budget_limit_exceeded"
            assert "Budget limit exceeded" in data["error"]["message"]

    async def test_budget_accumulation(self, mock_llm_completion, backend):
        """Budget usage should accumulate across requests."""
        # Budget is checked against estimated cost (based on input tokens + max_tokens)
        # Estimated cost for short message ≈ $0.0003 (at 150 tokens estimate)
        # Actual cost per request ≈ $0.00006 (30 tokens from mock)
        # Use $0.001 budget: allows ~3 requests (estimated blocks 4th)
        moderate_budget = Constraint(
            name="Moderate Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=0.001, window="1d"),
            rate_limit=None,
            description="$0.001 daily budget",
        )

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[moderate_budget], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Make several requests until budget is exhausted
            successful_requests = 0
            for i in range(20):  # Try up to 20 requests
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "metadata": {"user_id": "test_user"},
                    },
                )
                if response.status_code == 200:
                    successful_requests += 1
                else:
                    assert response.status_code == 429
                    break

            # Should have made at least one successful request
            assert successful_requests >= 1
            # But should eventually hit the limit (budget accumulates)
            assert successful_requests < 20


class TestCombinedConstraintsIntegration:
    """Integration tests for multiple constraints."""

    async def test_multiple_constraints_all_pass(self, mock_llm_completion, backend):
        """When all constraints pass, request should succeed."""
        constraints = [
            Constraint(
                name="Rate Limit",
                dimensions=["user_id"],
                budget_limit=None,
                rate_limit=RateLimit(per_minute=10),
                description="10 requests per minute",
            ),
            Constraint(
                name="Budget Limit",
                dimensions=["user_id"],
                budget_limit=BudgetLimit(limit=10.0, window="1d"),
                rate_limit=None,
                description="$10 daily budget",
            ),
        ]

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(constraints=constraints, backend=backend)
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "test_user"},
                },
            )
            assert response.status_code == 200

    async def test_rate_limit_fails_before_budget_check(
        self, mock_llm_completion, backend
    ):
        """When rate limit is exceeded, request should fail even if budget is available."""
        constraints = [
            Constraint(
                name="Tight Rate Limit",
                dimensions=["user_id"],
                budget_limit=None,
                rate_limit=RateLimit(per_minute=1),
                description="1 request per minute",
            ),
            Constraint(
                name="Generous Budget",
                dimensions=["user_id"],
                budget_limit=BudgetLimit(limit=1000.0, window="1d"),
                rate_limit=None,
                description="$1000 daily budget",
            ),
        ]

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(constraints=constraints, backend=backend)
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # First request succeeds
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "test_user"},
                },
            )
            assert response.status_code == 200

            # Second request fails due to rate limit
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello again"}],
                    "metadata": {"user_id": "test_user"},
                },
            )
            assert response.status_code == 429
            assert response.json()["error"]["type"] == "rate_limit_exceeded"


class TestMultiDimensionConstraintsIntegration:
    """Integration tests for multi-dimensional constraints."""

    async def test_organization_dimension(self, mock_llm_completion, backend):
        """Constraints can track by organization."""
        org_constraint = Constraint(
            name="Org Rate Limit",
            dimensions=["organization"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=2),
            description="2 requests per minute per organization",
        )

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[org_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Two requests from different users in same org
            for user in ["user_a", "user_b"]:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "metadata": {"user_id": user, "organization": "acme"},
                    },
                )
                assert response.status_code == 200

            # Third request from same org should be limited
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "user_c", "organization": "acme"},
                },
            )
            assert response.status_code == 429

            # But request from different org should succeed
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "user_a", "organization": "other_org"},
                },
            )
            assert response.status_code == 200

    async def test_combined_user_and_org_dimensions(self, mock_llm_completion, backend):
        """Constraints can track by user_id AND organization combined."""
        combined_constraint = Constraint(
            name="User-Org Rate Limit",
            dimensions=["user_id", "organization"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=2),
            description="2 requests per minute per user per organization",
        )

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[combined_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Same user in same org - 2 requests succeed, 3rd fails
            for i in range(2):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "metadata": {"user_id": "user_a", "organization": "acme"},
                    },
                )
                assert response.status_code == 200

            # 3rd request should fail
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Excess"}],
                    "metadata": {"user_id": "user_a", "organization": "acme"},
                },
            )
            assert response.status_code == 429

            # Same user in different org should succeed
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Different org"}],
                    "metadata": {"user_id": "user_a", "organization": "other_org"},
                },
            )
            assert response.status_code == 200


class TestErrorResponseFormat:
    """Tests verifying the error response format matches OpenAI API spec."""

    async def test_rate_limit_error_format(
        self, mock_llm_completion, backend, rate_limit_constraint
    ):
        """Rate limit errors should have proper error format."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[rate_limit_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Exhaust rate limit
            for _ in range(3):
                await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "metadata": {"user_id": "test_user"},
                    },
                )

            # Get the error response
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Excess"}],
                    "metadata": {"user_id": "test_user"},
                },
            )

            assert response.status_code == 429
            data = response.json()

            # Verify error structure
            assert "error" in data
            error = data["error"]
            assert "message" in error
            assert "type" in error
            assert error["type"] == "rate_limit_exceeded"
            assert "Rate limit exceeded" in error["message"]

    async def test_budget_limit_error_format(
        self, mock_llm_completion, backend, tiny_budget_constraint
    ):
        """Budget limit errors should have proper error format."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[tiny_budget_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "metadata": {"user_id": "test_user"},
                },
            )

            assert response.status_code == 429
            data = response.json()

            # Verify error structure
            assert "error" in data
            error = data["error"]
            assert "message" in error
            assert "type" in error
            assert error["type"] == "budget_limit_exceeded"
            assert "Budget limit exceeded" in error["message"]


class TestNoConstraints:
    """Tests for server with no constraints configured."""

    async def test_no_constraints_allows_all_requests(
        self, mock_llm_completion, backend
    ):
        """Server with empty constraints list should allow all requests."""

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(constraints=[], backend=backend)
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Many requests should all succeed
            for i in range(10):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "metadata": {"user_id": "test_user"},
                    },
                )
                assert response.status_code == 200


class TestConcurrentRequests:
    """Tests for concurrent request handling with rate limits."""

    async def test_concurrent_requests_respect_rate_limit(
        self, mock_llm_completion, backend
    ):
        """Concurrent requests should still respect rate limits."""
        rate_constraint = Constraint(
            name="Rate Limit",
            dimensions=["user_id"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=5),
            description="5 requests per minute",
        )

        def create_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                limit_enforcement_middleware(
                    constraints=[rate_constraint], backend=backend
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Send 10 concurrent requests
            tasks = [
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"Concurrent {i}"}],
                        "metadata": {"user_id": "test_user"},
                    },
                )
                for i in range(10)
            ]

            responses = await asyncio.gather(*tasks)

            # Should have exactly 5 successful and 5 rate-limited
            success_count = sum(1 for r in responses if r.status_code == 200)
            rate_limited_count = sum(1 for r in responses if r.status_code == 429)

            assert success_count == 5
            assert rate_limited_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
