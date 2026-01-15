"""
Shared fixtures for the SLLMP test suite.

This module provides common fixtures for mocking LLM completions, creating test
requests, and setting up HTTP test clients to reduce duplication across tests.
"""

import pytest
import httpx
from unittest.mock import patch
from typing import Any

from any_llm.types.completion import ChatCompletionChunk
from sllmp import SimpleProxyServer
from sllmp.context import Pipeline, NCompletionParams

# Import factory functions from helpers module
# Use absolute import path for pytest
import sys
from pathlib import Path

# Add tests directory to path for imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from helpers import create_chat_completion, create_stream_chunks


# =============================================================================
# Request Fixtures
# =============================================================================


@pytest.fixture
def basic_request() -> dict[str, Any]:
    """Basic chat completion request as a dictionary."""
    return {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.fixture
def streaming_request() -> dict[str, Any]:
    """Streaming chat completion request as a dictionary."""
    return {
        "model": "openai:gpt-4",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
    }


@pytest.fixture
def basic_completion_params() -> NCompletionParams:
    """Basic completion parameters for pipeline testing."""
    return NCompletionParams(
        model_id="openai:gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        metadata={},
    )


@pytest.fixture
def streaming_completion_params() -> NCompletionParams:
    """Streaming completion parameters for pipeline testing."""
    return NCompletionParams(
        model_id="openai:gpt-4",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
        metadata={},
    )


# =============================================================================
# Mock LLM Response Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_response():
    """Mock successful LLM response fixture."""
    return create_chat_completion()


@pytest.fixture
def mock_stream_chunks() -> list[ChatCompletionChunk]:
    """Mock streaming response chunks fixture."""
    return create_stream_chunks(content_parts=["Once", " upon"])


# =============================================================================
# Mock LLM Completion Fixture
# =============================================================================


@pytest.fixture
def mock_llm_completion():
    """
    Mock any_llm.acompletion to avoid needing real API keys.

    This fixture patches the acompletion function and handles both streaming
    and non-streaming requests automatically based on the 'stream' parameter.
    """

    def _create_completion(**kwargs):
        model = kwargs.get("model", kwargs.get("model_id", "openai:gpt-3.5-turbo"))
        content = kwargs.get("_test_content", "Hello! This is a test response.")
        return create_chat_completion(model=model, content=content)

    async def _create_stream(**kwargs):
        model = kwargs.get("model", kwargs.get("model_id", "openai:gpt-4"))
        chunks = create_stream_chunks(model=model)
        for chunk in chunks:
            yield chunk

    async def mock_acompletion(stream: bool = False, **kwargs):
        if stream:
            return _create_stream(**kwargs)
        else:
            return _create_completion(**kwargs)

    with patch("sllmp.pipeline.any_llm.acompletion") as mock_completion:
        mock_completion.side_effect = mock_acompletion
        yield mock_completion


# =============================================================================
# HTTP Test Client Fixtures
# =============================================================================


@pytest.fixture
async def client(mock_llm_completion):
    """
    HTTP client for testing with a basic pipeline (no custom middleware).

    This fixture provides an httpx.AsyncClient configured to send requests
    to a test server with mocked LLM completions.
    """

    def create_basic_pipeline():
        return Pipeline()

    server = SimpleProxyServer(pipeline_factory=create_basic_pipeline)
    app = server.create_asgi_app(debug=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as test_client:
        yield test_client


async def create_test_client(
    pipeline_factory=None,
    mock_completion=True,
):
    """
    Factory function to create a test client with custom pipeline configuration.

    Args:
        pipeline_factory: Optional callable that returns a Pipeline instance.
                         If None, creates a basic empty pipeline.
        mock_completion: If True, patches any_llm.acompletion. If a callable,
                        uses that as the side_effect.

    Returns:
        An async context manager that yields an httpx.AsyncClient.

    Usage:
        async with create_test_client(pipeline_factory=my_factory) as client:
            response = await client.post("/v1/chat/completions", json=request)
    """
    if pipeline_factory is None:

        def pipeline_factory():
            return Pipeline()

    server = SimpleProxyServer(pipeline_factory=pipeline_factory)
    app = server.create_asgi_app(debug=True)

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
