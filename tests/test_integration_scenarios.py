"""
Integration tests for middleware chains and complex scenarios.

These tests validate middleware integration, custom server configurations,
concurrent request handling, and real-world usage patterns.

Note: Basic endpoint tests (health, models, chat completions, error handling)
are in test_main.py. This file focuses on middleware behavior and advanced scenarios.
"""

import pytest
import httpx
import json
import asyncio
from unittest.mock import patch

from sllmp import SimpleProxyServer
from sllmp.context import Pipeline, RequestContext
from sllmp.middleware import logging_middleware, retry_middleware
from sllmp.error import ValidationError, ProviderRateLimitError

from helpers import create_chat_completion, create_stream_chunks


# Note: mock_llm_completion is provided by conftest.py but we need a custom version
# for integration tests that includes the request content in the response


@pytest.fixture
def integration_mock_llm():
    """Mock any_llm.acompletion that echoes request content in response."""

    def create_completion(**kwargs):
        model_id = kwargs.get("model", kwargs.get("model_id", "openai:gpt-3.5-turbo"))
        messages = kwargs.get("messages", [{}])
        last_content = messages[-1].get("content", "unknown") if messages else "unknown"
        return create_chat_completion(
            model=model_id,
            content=f"Response to: {last_content}",
            completion_id="chatcmpl-integration",
        )

    async def create_stream(**kwargs):
        model_id = kwargs.get("model", kwargs.get("model_id", "openai:gpt-4"))
        chunks = create_stream_chunks(
            model=model_id,
            content_parts=["Streaming", " response"],
            completion_id="chatcmpl-integration",
        )
        for chunk in chunks:
            yield chunk

    async def mock_completion(stream=False, **kwargs):
        if stream:
            return create_stream(**kwargs)
        else:
            return create_completion(**kwargs)

    with patch("sllmp.pipeline.any_llm.acompletion", side_effect=mock_completion):
        yield


@pytest.fixture
async def basic_client(integration_mock_llm):
    """HTTP client for basic server without custom middleware."""
    server = SimpleProxyServer(pipeline_factory=Pipeline)
    app = server.create_asgi_app(debug=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
async def custom_pipeline_client(integration_mock_llm):
    """HTTP client with custom middleware pipeline."""

    def create_custom_pipeline():
        pipeline = Pipeline()

        # Add custom middleware for testing
        def rate_limiting_middleware(ctx: RequestContext):
            async def check_rate_limit(ctx: RequestContext):
                user_id = ctx.client_metadata.get("user_id", "anonymous")
                if user_id == "rate_limited_user":
                    raise ProviderRateLimitError(
                        "Rate limit exceeded",
                        request_id=ctx.request_id,
                        provider="test",
                        retry_after=60,
                    )

            ctx.pipeline.pre.connect(check_rate_limit)

        def content_filter_middleware(ctx: RequestContext):
            async def filter_content(ctx: RequestContext):
                if ctx.request.messages:
                    last_message = ctx.request.messages[-1]
                    if (
                        isinstance(last_message, dict)
                        and "blocked" in last_message.get("content", "").lower()
                    ):
                        raise ValidationError(
                            "Content blocked by policy", request_id=ctx.request_id
                        )

            ctx.pipeline.pre.connect(filter_content)

        # Add basic middleware
        pipeline.setup.connect(rate_limiting_middleware)
        pipeline.setup.connect(content_filter_middleware)
        pipeline.setup.connect(logging_middleware())

        return pipeline

    server = SimpleProxyServer(pipeline_factory=create_custom_pipeline)
    app = server.create_asgi_app(debug=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


class TestConcurrentRequests:
    """Test server handles concurrent requests correctly."""

    async def test_concurrent_requests(self, basic_client):
        """Test server handles concurrent requests correctly."""
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Concurrent test"}],
        }

        # Send 5 concurrent requests
        tasks = [
            basic_client.post(
                "/v1/chat/completions",
                json={
                    **request_data,
                    "messages": [{"role": "user", "content": f"Concurrent test {i}"}],
                },
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # Each should have unique content reflecting the request
        contents = [r.json()["choices"][0]["message"]["content"] for r in responses]
        assert len(set(contents)) == 5  # All unique responses


class TestMiddlewareIntegration:
    """Test middleware chains and complex processing scenarios."""

    async def test_rate_limiting_middleware(self, custom_pipeline_client):
        """Test rate limiting middleware blocks requests."""
        # Normal request should pass
        normal_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Normal request"}],
            "metadata": {"user_id": "normal_user"},
        }

        response = await custom_pipeline_client.post(
            "/v1/chat/completions", json=normal_request
        )
        assert response.status_code == 200

        # Rate limited user should be blocked
        limited_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Normal message"}],
            "metadata": {"user_id": "rate_limited_user"},
        }

        response = await custom_pipeline_client.post(
            "/v1/chat/completions", json=limited_request
        )
        assert response.status_code == 429  # Rate limit error

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "provider_rate_limit_error"

    async def test_content_filtering_middleware(self, custom_pipeline_client):
        """Test content filtering middleware blocks inappropriate content."""
        # Normal content should pass
        normal_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Tell me about the weather"}],
        }

        response = await custom_pipeline_client.post(
            "/v1/chat/completions", json=normal_request
        )
        assert response.status_code == 200

        # Blocked content should be rejected
        blocked_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "This content should be blocked"}],
        }

        response = await custom_pipeline_client.post(
            "/v1/chat/completions", json=blocked_request
        )
        assert response.status_code == 422  # Validation error

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "blocked" in data["error"]["message"].lower()

    async def test_middleware_chain_execution_order(self, custom_pipeline_client):
        """Test middleware executes in correct order and state is shared."""
        # Create request that would trigger multiple middleware
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test middleware chain"}],
            "metadata": {"user_id": "normal_user"},
        }

        response = await custom_pipeline_client.post(
            "/v1/chat/completions", json=request_data
        )
        assert response.status_code == 200

        # Verify response shows middleware processing occurred
        data = response.json()
        assert "id" in data
        assert data["choices"][0]["message"]["content"] is not None


class TestRealWorldScenarios:
    """Test realistic usage patterns and edge cases."""

    async def test_large_context_request(self, basic_client):
        """Test handling of requests with large context."""
        large_context = [
            {"role": "user", "content": "Context message " + "a" * 1000},
            {"role": "assistant", "content": "Response " + "b" * 1000},
            {
                "role": "user",
                "content": "Follow up question with more context " + "c" * 1000,
            },
        ]

        request_data = {"model": "openai:gpt-3.5-turbo", "messages": large_context}

        response = await basic_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    async def test_multimodal_content_structure(self, basic_client):
        """Test handling of multimodal content in messages."""
        multimodal_request = {
            "model": "openai:gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
                            },
                        },
                    ],
                }
            ],
        }

        response = await basic_client.post(
            "/v1/chat/completions", json=multimodal_request
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "openai:gpt-4-vision-preview"

    async def test_various_completion_parameters(self, basic_client):
        """Test requests with various OpenAI parameters."""
        advanced_request = {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Generate creative text"}],
            "temperature": 0.8,
            "max_tokens": 150,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "user": "integration_test_user",
        }

        response = await basic_client.post(
            "/v1/chat/completions", json=advanced_request
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "openai:gpt-4"
        assert "usage" in data

    async def test_streaming_with_early_termination(self, basic_client):
        """Test streaming requests handle early client disconnection gracefully."""
        request_data = {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Long story"}],
            "stream": True,
        }

        # Start streaming request
        response = await basic_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        # Verify we get streaming response format
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content


class TestCustomServerConfiguration:
    """Test custom server configurations and middleware setups."""

    async def test_server_with_retry_middleware(self):
        """Test server configured with retry middleware."""

        def create_retry_pipeline():
            pipeline = Pipeline()
            pipeline.setup.connect(
                retry_middleware(
                    max_attempts=2,
                    base_delay=0.1,  # Fast for testing
                    max_delay=1.0,
                )
            )
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_retry_pipeline)
        app = server.create_asgi_app()

        # Test with mocked failing then succeeding LLM call
        call_count = 0

        def failing_then_success(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return create_chat_completion(
                model="test",
                content="Success after retry",
                completion_id="retry-test",
            )

        with patch(
            "sllmp.pipeline.any_llm.acompletion", side_effect=failing_then_success
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai:gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Test retry"}],
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert "Success after retry" in data["choices"][0]["message"]["content"]
                assert call_count == 2  # Verify retry occurred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
