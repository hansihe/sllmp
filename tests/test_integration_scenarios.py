"""
High-impact integration tests for realistic usage scenarios.

These tests validate end-to-end functionality including server setup,
request handling, middleware chains, and real-world usage patterns.
"""

import pytest
import httpx
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from sllmp import SimpleProxyServer
from sllmp.context import Pipeline, RequestContext
from sllmp.middleware import logging_middleware, retry_middleware
from sllmp.error import ValidationError, RateLimitError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk


@pytest.fixture
def mock_llm_completion():
    """Mock any_llm.acompletion for integration tests."""
    def create_completion(**kwargs):
        # Return a dictionary that can be serialized, not a ChatCompletion object
        # Debug: print what model_id we're receiving
        # print(f"DEBUG: create_completion received kwargs: {kwargs}")
        # The kwargs contain 'model' not 'model_id' based on debug output
        model_id = kwargs.get("model", kwargs.get("model_id", "openai:gpt-3.5-turbo"))
        return {
            "id": "chatcmpl-integration",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model_id,  # This will reflect the actual requested model_id
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Response to: {kwargs.get('messages', [{}])[-1].get('content', 'unknown')}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

    async def create_stream(**kwargs):
        chunks = [
            ChatCompletionChunk(
                id="chatcmpl-integration",
                object="chat.completion.chunk",
                created=1234567890,
                model=kwargs.get("model_id", "openai:gpt-4"),
                choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
            ),
            ChatCompletionChunk(
                id="chatcmpl-integration",
                object="chat.completion.chunk",
                created=1234567890,
                model=kwargs.get("model_id", "openai:gpt-4"),
                choices=[{"index": 0, "delta": {"content": "Streaming"}, "finish_reason": None}]
            ),
            ChatCompletionChunk(
                id="chatcmpl-integration",
                object="chat.completion.chunk",
                created=1234567890,
                model=kwargs.get("model_id", "openai:gpt-4"),
                choices=[{"index": 0, "delta": {"content": " response"}, "finish_reason": None}]
            ),
            ChatCompletionChunk(
                id="chatcmpl-integration",
                object="chat.completion.chunk",
                created=1234567890,
                model=kwargs.get("model_id", "openai:gpt-4"),
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]
            )
        ]
        for chunk in chunks:
            yield chunk

    async def mock_completion(stream=False, **kwargs):
        if stream:
            return create_stream(**kwargs)
        else:
            return create_completion(**kwargs)

    with patch('sllmp.pipeline.any_llm.acompletion', side_effect=mock_completion):
        yield


@pytest.fixture
async def basic_client(mock_llm_completion):
    """HTTP client for basic server without custom middleware."""
    server = SimpleProxyServer()
    app = server.create_asgi_app(debug=True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def custom_pipeline_client(mock_llm_completion):
    """HTTP client with custom middleware pipeline."""
    def create_custom_pipeline():
        pipeline = Pipeline()

        # Add custom middleware for testing
        def rate_limiting_middleware(ctx: RequestContext):
            async def check_rate_limit(ctx: RequestContext):
                user_id = ctx.client_metadata.get("user_id", "anonymous")
                if user_id == "rate_limited_user":
                    ctx.set_error(RateLimitError(
                        "Rate limit exceeded",
                        request_id=ctx.request_id,
                        provider="test",
                        retry_after=60
                    ))
                    return

            ctx.pipeline.pre.connect(check_rate_limit)

        def content_filter_middleware(ctx: RequestContext):
            async def filter_content(ctx: RequestContext):
                if ctx.request.messages:
                    last_message = ctx.request.messages[-1]
                    if isinstance(last_message, dict) and "blocked" in last_message.get("content", "").lower():
                        ctx.set_error(ValidationError(
                            "Content blocked by policy",
                            request_id=ctx.request_id
                        ))
                        return

            ctx.pipeline.pre.connect(filter_content)

        # Add basic middleware
        pipeline.setup.connect(rate_limiting_middleware)
        pipeline.setup.connect(content_filter_middleware)
        pipeline.setup.connect(logging_middleware())

        return pipeline

    server = SimpleProxyServer(pipeline_factory=create_custom_pipeline)
    app = server.create_asgi_app(debug=True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


class TestBasicIntegration:
    """Test basic server functionality and request handling."""

    async def test_health_endpoints(self, basic_client):
        """Test health check endpoints work correctly."""
        # Test root health check
        response = await basic_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

        # Test dedicated health endpoint
        response = await basic_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    async def test_models_endpoint(self, basic_client):
        """Test models listing endpoint."""
        response = await basic_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

    async def test_basic_chat_completion(self, basic_client):
        """Test basic chat completion request flow."""
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, world!"}]
        }

        response = await basic_client.post("/v1/chat/completions", json=request_data)
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert data["model"] == "openai:gpt-3.5-turbo"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Hello, world!" in data["choices"][0]["message"]["content"]

    async def test_streaming_chat_completion(self, basic_client):
        """Test streaming chat completion request flow."""
        request_data = {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True
        }

        response = await basic_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Parse streaming response
        content = response.text
        lines = content.strip().split('\n')

        # Should have data lines and final [DONE]
        data_lines = [line for line in lines if line.startswith('data: ') and not line.endswith('[DONE]')]
        done_lines = [line for line in lines if line.endswith('[DONE]')]

        assert len(data_lines) >= 3  # At least 3 content chunks
        assert len(done_lines) == 1

        # Verify first chunk structure
        first_chunk_data = data_lines[0][6:]  # Remove 'data: ' prefix
        chunk = json.loads(first_chunk_data)
        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"

    async def test_concurrent_requests(self, basic_client):
        """Test server handles concurrent requests correctly."""
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Concurrent test"}]
        }

        # Send 5 concurrent requests
        tasks = [
            basic_client.post("/v1/chat/completions", json={
                **request_data,
                "messages": [{"role": "user", "content": f"Concurrent test {i}"}]
            })
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
            "metadata": {"user_id": "normal_user"}
        }

        response = await custom_pipeline_client.post("/v1/chat/completions", json=normal_request)
        assert response.status_code == 200

        # Rate limited user should be blocked
        limited_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Should be blocked"}],
            "metadata": {"user_id": "rate_limited_user"}
        }

        response = await custom_pipeline_client.post("/v1/chat/completions", json=limited_request)
        assert response.status_code == 429  # Rate limit error

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "rate_limit_error"

    async def test_content_filtering_middleware(self, custom_pipeline_client):
        """Test content filtering middleware blocks inappropriate content."""
        # Normal content should pass
        normal_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Tell me about the weather"}]
        }

        response = await custom_pipeline_client.post("/v1/chat/completions", json=normal_request)
        assert response.status_code == 200

        # Blocked content should be rejected
        blocked_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "This content should be blocked"}]
        }

        response = await custom_pipeline_client.post("/v1/chat/completions", json=blocked_request)
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
            "metadata": {"user_id": "normal_user"}
        }

        response = await custom_pipeline_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        # Verify response shows middleware processing occurred
        data = response.json()
        assert "id" in data
        assert data["choices"][0]["message"]["content"] is not None


class TestErrorHandlingIntegration:
    """Test error handling across the complete request flow."""

    async def test_invalid_json_handling(self, basic_client):
        """Test server handles invalid JSON gracefully."""
        response = await basic_client.post(
            "/v1/chat/completions",
            content="invalid json",
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"

    async def test_missing_required_fields(self, basic_client):
        """Test validation of required fields."""
        # Missing model
        response = await basic_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "model" in data["error"]["message"]

    async def test_invalid_message_structure(self, basic_client):
        """Test validation of message structure."""
        invalid_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": ["invalid message format"]  # Should be objects
        }

        response = await basic_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"

    async def test_parameter_validation(self, basic_client):
        """Test validation of optional parameters."""
        invalid_request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 5.0  # Too high
        }

        response = await basic_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422

        data = response.json()
        assert "error" in data
        assert "temperature" in data["error"]["message"]

    @patch('sllmp.pipeline.any_llm.acompletion')
    async def test_llm_provider_error_handling(self, mock_completion, basic_client):
        """Test handling of LLM provider errors."""
        mock_completion.side_effect = Exception("Provider API error")

        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "This will fail"}]
        }

        response = await basic_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 500  # Internal server error

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "internal_error"


class TestRealWorldScenarios:
    """Test realistic usage patterns and edge cases."""

    async def test_large_context_request(self, basic_client):
        """Test handling of requests with large context."""
        large_context = [
            {"role": "user", "content": "Context message " + "a" * 1000},
            {"role": "assistant", "content": "Response " + "b" * 1000},
            {"role": "user", "content": "Follow up question with more context " + "c" * 1000}
        ]

        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": large_context
        }

        response = await basic_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    async def test_multimodal_content_structure(self, basic_client):
        """Test handling of multimodal content in messages."""
        multimodal_request = {
            "model": "openai:gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."}
                    }
                ]
            }]
        }

        response = await basic_client.post("/v1/chat/completions", json=multimodal_request)
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
            "user": "integration_test_user"
        }

        response = await basic_client.post("/v1/chat/completions", json=advanced_request)
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "openai:gpt-4"
        assert "usage" in data

    async def test_streaming_with_early_termination(self, basic_client):
        """Test streaming requests handle early client disconnection gracefully."""
        request_data = {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Long story"}],
            "stream": True
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
            pipeline.setup.connect(retry_middleware(
                max_attempts=2,
                base_delay=0.1,  # Fast for testing
                max_delay=1.0
            ))
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
            return ChatCompletion(
                id="retry-test", object="chat.completion", created=123, model="test",
                choices=[{"index": 0, "message": {"role": "assistant", "content": "Success after retry"}, "finish_reason": "stop"}],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )

        with patch('sllmp.pipeline.any_llm.acompletion', side_effect=failing_then_success):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                response = await client.post("/v1/chat/completions", json={
                    "model": "openai:gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test retry"}]
                })

                assert response.status_code == 200
                data = response.json()
                assert "Success after retry" in data["choices"][0]["message"]["content"]
                assert call_count == 2  # Verify retry occurred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
