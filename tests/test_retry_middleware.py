"""
Tests for retry middleware functionality.

These tests verify that the retry middleware properly handles different error types,
implements exponential backoff, and integrates correctly with the pipeline.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from sllmp.context import RequestContext, Pipeline, PipelineState
from sllmp.middleware.retry import retry_middleware, _is_retryable_error
from sllmp.error import (
    RateLimitError,
    NetworkError,
    ServiceUnavailableError,
    InternalError,
    AuthenticationError,
    ValidationError,
    ContentPolicyError,
    ModelNotFoundError
)
from sllmp.context import NCompletionParams


@pytest.fixture
def mock_completion_params():
    """Create a mock completion params for testing."""
    return NCompletionParams(
        model_id="openai:gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        metadata={}
    )


@pytest.fixture
def mock_context(mock_completion_params):
    """Create a mock request context for testing."""
    ctx = RequestContext(
        original_request=mock_completion_params,
        request=mock_completion_params,
        pipeline=Pipeline()
    )
    return ctx


class TestRetryMiddleware:

    def test_retryable_error_detection(self):
        """Test that retryable errors are correctly identified."""
        # Retryable errors
        retryable_errors = {RateLimitError, NetworkError, ServiceUnavailableError, InternalError}

        rate_limit_error = RateLimitError("Rate limited", "req_123", "openai")
        network_error = NetworkError("Connection failed", "req_123", "openai")
        service_error = ServiceUnavailableError("Service down", "req_123", "openai")
        internal_error = InternalError("Internal error", "req_123")

        assert _is_retryable_error(rate_limit_error, retryable_errors)
        assert _is_retryable_error(network_error, retryable_errors)
        assert _is_retryable_error(service_error, retryable_errors)
        assert _is_retryable_error(internal_error, retryable_errors)

    def test_non_retryable_error_detection(self):
        """Test that non-retryable errors are correctly identified."""
        retryable_errors = {RateLimitError, NetworkError, ServiceUnavailableError, InternalError}

        auth_error = AuthenticationError("Bad API key", "req_123")
        validation_error = ValidationError("Invalid request", "req_123")
        content_error = ContentPolicyError("Content blocked", "req_123", "openai")
        model_error = ModelNotFoundError("Model not found", "req_123", "openai")

        assert not _is_retryable_error(auth_error, retryable_errors)
        assert not _is_retryable_error(validation_error, retryable_errors)
        assert not _is_retryable_error(content_error, retryable_errors)
        assert not _is_retryable_error(model_error, retryable_errors)

    def test_middleware_setup(self, mock_context):
        """Test that retry middleware sets up correctly."""
        middleware = retry_middleware(max_attempts=5, base_delay=2.0)
        middleware(mock_context)

        # Check that retry state is stored in both metadata and state
        assert 'retry_state' in mock_context.state
        retry_state = mock_context.state['retry_state']
        assert retry_state.max_attempts == 5
        assert retry_state.base_delay == 2.0
        assert retry_state.max_delay == 60.0
        assert retry_state.log_retries is True

        # Check that retry tracking is initialized
        assert retry_state.attempts == 0
        assert retry_state.history == []

        # Check that error signal handler is connected
        assert len(mock_context.pipeline.error) > 0

    @pytest.mark.asyncio
    async def test_non_retryable_error_passthrough(self, mock_context):
        """Test that non-retryable errors pass through without retry."""
        middleware = retry_middleware(max_attempts=3)
        middleware(mock_context)

        # Simulate authentication error
        auth_error = AuthenticationError("Invalid API key", mock_context.request_id)
        mock_context.error = auth_error
        mock_context.response = None

        # Trigger error signal
        await mock_context.pipeline.error.emit(mock_context)

        # Error should still be present (no retry)
        assert mock_context.has_error
        assert isinstance(mock_context.error, AuthenticationError)
        assert mock_context.state['retry_state'].attempts == 0

    @pytest.mark.asyncio
    async def test_retryable_error_with_retry(self, mock_context):
        """Test that retryable errors trigger retry logic."""
        middleware = retry_middleware(max_attempts=3, base_delay=0.1)  # Fast for testing
        middleware(mock_context)

        # Simulate network error
        network_error = NetworkError("Connection timeout", mock_context.request_id, "openai")
        mock_context.error = network_error
        mock_context.response = None

        # Trigger error signal
        await mock_context.pipeline.error.emit(mock_context)

        # Error should be cleared and pipeline state should be reset for retry
        assert not mock_context.has_error
        assert mock_context.next_pipeline_state == PipelineState.LLM_CALL
        assert mock_context.state['retry_state'].attempts == 1
        assert len(mock_context.state['retry_state'].history) == 1

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, mock_context):
        """Test that retries are exhausted after max attempts."""
        middleware = retry_middleware(max_attempts=2, base_delay=0.1)
        middleware(mock_context)

        # Simulate retries until exhaustion
        for attempt in range(3):  # One more than max_attempts
            service_error = ServiceUnavailableError("Service down", mock_context.request_id, "openai")
            mock_context.error = service_error
            mock_context.response = None

            await mock_context.pipeline.error.emit(mock_context)

            # If we exceeded max attempts, error should remain
            if attempt >= 2:  # max_attempts
                break

        # After exhaustion, error should remain
        assert mock_context.has_error
        assert isinstance(mock_context.error, ServiceUnavailableError)
        assert mock_context.state['retry_state'].attempts == 2  # Max attempts reached

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, mock_context):
        """Test that exponential backoff delays are applied correctly."""
        middleware = retry_middleware(
            max_attempts=3,
            base_delay=0.1,
            backoff_multiplier=2.0,
            max_delay=1.0
        )
        middleware(mock_context)

        delays = []

        async def mock_sleep(delay):
            delays.append(delay)

        with patch('asyncio.sleep', side_effect=mock_sleep):
            for attempt in range(2):  # Two retries
                rate_limit_error = RateLimitError("Rate limited", mock_context.request_id, "openai")
                mock_context.error = rate_limit_error
                mock_context.response = None

                await mock_context.pipeline.error.emit(mock_context)

        # Check exponential backoff: 0.1, 0.2
        assert len(delays) == 2
        assert delays[0] == 0.1  # base_delay * (2^0)
        assert delays[1] == 0.2  # base_delay * (2^1)

    @pytest.mark.asyncio
    async def test_rate_limit_retry_after_respect(self, mock_context):
        """Test that RateLimitError retry_after is respected."""
        middleware = retry_middleware(max_attempts=3, base_delay=0.1)
        middleware(mock_context)

        delays = []

        async def mock_sleep(delay):
            delays.append(delay)

        with patch('asyncio.sleep', side_effect=mock_sleep):
            # Rate limit error with retry_after
            rate_limit_error = RateLimitError(
                "Rate limited",
                mock_context.request_id,
                "openai",
                retry_after=5  # 5 seconds
            )
            mock_context.error = rate_limit_error
            mock_context.response = None

            await mock_context.pipeline.error.emit(mock_context)

        # Should respect the retry_after value (5s > base_delay of 0.1s)
        assert len(delays) == 1
        assert delays[0] == 5.0

    @pytest.mark.asyncio
    async def test_max_delay_cap(self, mock_context):
        """Test that delays are capped at max_delay."""
        middleware = retry_middleware(
            max_attempts=10,
            base_delay=1.0,
            backoff_multiplier=10.0,  # Very aggressive multiplier
            max_delay=5.0  # Low cap
        )
        middleware(mock_context)

        delays = []

        async def mock_sleep(delay):
            delays.append(delay)

        with patch('asyncio.sleep', side_effect=mock_sleep):
            for attempt in range(5):
                network_error = NetworkError("Connection failed", mock_context.request_id, "openai")
                mock_context.error = network_error
                mock_context.response = None

                await mock_context.pipeline.error.emit(mock_context)

        # All delays should be capped at max_delay
        for delay in delays:
            assert delay <= 5.0

    @pytest.mark.asyncio
    async def test_custom_retryable_errors(self, mock_context):
        """Test retry middleware with custom retryable error set."""
        # Only retry rate limit errors
        custom_retryable = {RateLimitError}
        middleware = retry_middleware(
            max_attempts=3,
            retryable_errors=custom_retryable
        )
        middleware(mock_context)

        # Network error should not be retried with custom set
        network_error = NetworkError("Connection failed", mock_context.request_id, "openai")
        mock_context.error = network_error
        mock_context.response = None

        await mock_context.pipeline.error.emit(mock_context)

        # Error should remain (no retry)
        assert mock_context.has_error
        assert isinstance(mock_context.error, NetworkError)
        assert mock_context.state['retry_state'].attempts == 0

        # Rate limit error should be retried
        mock_context.error = None  # Clear error
        mock_context.response = None
        rate_limit_error = RateLimitError("Rate limited", mock_context.request_id, "openai")
        mock_context.error = rate_limit_error

        await mock_context.pipeline.error.emit(mock_context)

        # Should have been retried
        assert not mock_context.has_error
        assert mock_context.next_pipeline_state == PipelineState.LLM_CALL
        assert mock_context.state['retry_state'].attempts == 1


class TestRetryIntegration:
    """Integration tests for retry middleware with the pipeline."""

    @pytest.mark.asyncio
    async def test_retry_history_tracking(self, mock_context):
        """Test that retry history is properly tracked."""
        middleware = retry_middleware(max_attempts=3, base_delay=0.1)
        middleware(mock_context)

        # Simulate multiple retry attempts
        error_messages = ["First failure", "Second failure"]

        for i, message in enumerate(error_messages):
            service_error = ServiceUnavailableError(message, mock_context.request_id, "openai")
            mock_context.error = service_error
            mock_context.response = None

            await mock_context.pipeline.error.emit(mock_context)

        # Check retry history
        history = mock_context.state['retry_state'].history
        assert len(history) == 2

        assert history[0]['attempt'] == 1
        assert history[0]['error_type'] == 'ServiceUnavailableError'
        assert history[0]['error_message'] == "First failure"

        assert history[1]['attempt'] == 2
        assert history[1]['error_type'] == 'ServiceUnavailableError'
        assert history[1]['error_message'] == "Second failure"

    @pytest.mark.asyncio
    async def test_no_retry_without_config(self, mock_context):
        """Test that errors pass through when no retry config is present."""
        # Don't apply retry middleware

        network_error = NetworkError("Connection failed", mock_context.request_id, "openai")
        mock_context.error = network_error
        mock_context.response = None

        # Trigger error signal (should be no-op)
        await mock_context.pipeline.error.emit(mock_context)

        # Error should remain unchanged
        assert mock_context.has_error
        assert isinstance(mock_context.error, NetworkError)
        assert 'retry_state' not in mock_context.state
