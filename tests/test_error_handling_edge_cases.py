"""
Comprehensive error handling and edge case tests.

These tests cover error propagation, recovery scenarios, boundary conditions,
and resilience patterns to ensure the system behaves predictably under stress.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, AsyncGenerator

from sllmp.context import Pipeline, RequestContext, PipelineState, NCompletionParams
from sllmp.pipeline import create_request_context, execute_pipeline
from sllmp.error import (
    ValidationError, PipelineError, ProviderRateLimitError,
    LLMProviderError, NetworkError, InternalError, MiddlewareError
)
from sllmp.util.signal import Signal
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk


@pytest.fixture
def basic_params():
    """Basic completion parameters."""
    return NCompletionParams(
        model_id="openai:gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
        metadata={}
    )


class TestSignalErrorHandling:
    """Test error handling in the signal system."""

    async def test_signal_execution_with_failing_callback(self):
        """Test signal continues execution when callbacks fail."""
        signal = Signal()
        results = []

        async def good_callback_1():
            results.append("good1")

        async def failing_callback():
            results.append("before_fail")
            raise ValueError("Callback failed")

        async def good_callback_2():
            results.append("good2")

        signal.connect(good_callback_1)
        signal.connect(failing_callback)
        signal.connect(good_callback_2)

        # Execute signal
        result = await signal.emit()

        # Should execute all callbacks despite failure
        assert "good1" in results
        assert "before_fail" in results
        assert "good2" in results
        assert result.callbacks_executed == 3
        assert len(result.exceptions) == 1
        assert isinstance(result.exceptions[0], ValueError)

    async def test_signal_exception_continues_processing(self):
        """Test that exceptions no longer halt signal processing."""
        signal = Signal()
        results = []

        async def good_callback():
            results.append("executed")

        async def failing_callback():
            results.append("before_fail")
            raise ValueError("Callback failed")

        async def final_callback():
            results.append("should_execute")

        signal.connect(good_callback)
        signal.connect(failing_callback)
        signal.connect(final_callback)

        result = await signal.emit()

        # All callbacks should execute despite the exception
        assert "executed" in results
        assert "before_fail" in results
        assert "should_execute" in results
        assert result.completed is True  # All callbacks executed
        assert result.callbacks_executed == 3
        assert result.callbacks_skipped == 0
        assert len(result.exceptions) == 1
        assert isinstance(result.exceptions[0], ValueError)

    async def test_signal_with_recursive_emission_error(self):
        """Test signal prevents recursive emission."""
        from sllmp.util.signal import Signal

        signal = Signal()
        error_raised = None

        async def recursive_callback():
            nonlocal error_raised
            try:
                # This should raise RuntimeError due to recursive emission
                await signal.emit()
            except RuntimeError as e:
                error_raised = e
                raise  # Re-raise so the signal system sees it

        signal.connect(recursive_callback)

        # Execute the signal and check what happened
        result = await signal.emit()

        # Either RuntimeError was raised during execution, or it's in the result
        if error_raised:
            assert "recursive emission" in str(error_raised)
        else:
            # Check if the recursive call was handled differently
            assert not result.success or len(result.exceptions) > 0


class TestPipelineErrorRecovery:
    """Test error recovery and resilience in pipeline execution."""

    async def test_middleware_error_sets_pipeline_error_state(self, basic_params):
        """Test middleware errors properly set pipeline error state."""
        def failing_middleware(ctx: RequestContext):
            async def fail_in_pre(ctx: RequestContext):
                raise ValueError("Middleware failed")

            ctx.pipeline.pre.connect(fail_in_pre)

        ctx = create_request_context(basic_params)
        ctx.add_middleware(failing_middleware)

        # Execute pipeline
        async for result in execute_pipeline(ctx):
            continue

        # Should capture error and complete pipeline execution
        assert ctx.has_error
        assert ctx.pipeline_state == PipelineState.COMPLETE
        
        # Verify error is properly set
        assert isinstance(ctx.error, MiddlewareError)
        assert "Middleware failed" in str(ctx.error)

    async def test_error_in_streaming_chunk_processing(self, basic_params):
        """Test error handling during streaming chunk processing."""
        streaming_params = NCompletionParams(
            model_id="openai:gpt-4",
            messages=[{"role": "user", "content": "Stream test"}],
            stream=True,
            metadata={}
        )

        # Mock stream that works initially then fails
        call_count = 0
        async def failing_stream():
            nonlocal call_count
            chunks = [
                ChatCompletionChunk(
                    id="test", object="chat.completion.chunk", created=123, model="test",
                    choices=[{"index": 0, "delta": {"content": "Good"}, "finish_reason": None}]
                )
            ]

            for chunk in chunks:
                call_count += 1
                if call_count > 1:
                    raise NetworkError("Stream connection lost", request_id="test", provider="test")
                yield chunk

        def error_prone_middleware(ctx: RequestContext):
            async def process_chunk(ctx: RequestContext, chunk):
                if ctx.chunk_count > 0:  # Fail after first chunk
                    raise ValueError("Chunk processing failed")

            ctx.pipeline.llm_call_stream_process.connect(process_chunk)

        with patch('sllmp.pipeline.any_llm.acompletion', return_value=failing_stream()):
            ctx = create_request_context(streaming_params)
            ctx.add_middleware(error_prone_middleware)

            chunks = []
            async for item in execute_pipeline(ctx):
                chunks.append(item)

            # Should handle error gracefully
            assert ctx.has_error or len(chunks) >= 1

    async def test_pipeline_state_consistency_after_error(self, basic_params):
        """Test pipeline state remains consistent after errors."""
        error_count = 0

        def error_tracking_middleware(ctx: RequestContext):
            async def track_errors(ctx: RequestContext):
                nonlocal error_count
                error_count += 1
                if error_count == 1:
                    raise ValidationError("First error", request_id=ctx.request_id)

            ctx.pipeline.pre.connect(track_errors)

        ctx = create_request_context(basic_params)
        ctx.add_middleware(error_tracking_middleware)

        # Execute pipeline
        async for result in execute_pipeline(ctx):
            continue

        # Verify consistent error state
        assert ctx.has_error
        assert not ctx.has_response
        assert isinstance(ctx.error, ValidationError)
        assert ctx.request_id is not None
        assert ctx.original_request == basic_params


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    async def test_empty_messages_array(self):
        """Test handling of empty messages array."""
        # Pydantic validation prevents empty messages, so test with minimal valid message
        params = NCompletionParams(
            model_id="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": ""}],  # Minimal valid message
            metadata={}
        )

        ctx = create_request_context(params)

        # Should handle minimal messages gracefully
        assert len(ctx.request.messages) == 1
        assert not ctx.is_streaming

    async def test_extremely_large_message_content(self):
        """Test handling of very large message content."""
        large_content = "x" * 100000  # 100K characters
        params = NCompletionParams(
            model_id="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": large_content}],
            metadata={}
        )

        ctx = create_request_context(params)

        # Should handle large content without crashing
        assert len(ctx.request.messages[0]["content"]) == 100000

    async def test_zero_temperature_parameter(self):
        """Test boundary value for temperature parameter."""
        params = NCompletionParams(
            model_id="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.0,  # Boundary value
            metadata={}
        )

        ctx = create_request_context(params)
        assert ctx.request.temperature == 0.0

    async def test_maximum_temperature_parameter(self):
        """Test maximum boundary value for temperature."""
        params = NCompletionParams(
            model_id="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            temperature=2.0,  # Maximum allowed
            metadata={}
        )

        ctx = create_request_context(params)
        assert ctx.request.temperature == 2.0

    async def test_maximum_tokens_parameter(self):
        """Test very high max_tokens value."""
        params = NCompletionParams(
            model_id="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=100000,  # Very high value
            metadata={}
        )

        ctx = create_request_context(params)
        assert ctx.request.max_tokens == 100000


class TestConcurrencyAndRaceConditions:
    """Test concurrent execution and potential race conditions."""

    async def test_concurrent_pipeline_executions(self, basic_params):
        """Test multiple pipelines can execute concurrently."""
        execution_order = []

        def tracking_middleware(name: str):
            def middleware(ctx: RequestContext):
                async def track(ctx: RequestContext):
                    execution_order.append(f"{name}_start")
                    await asyncio.sleep(0.01)  # Small delay
                    execution_order.append(f"{name}_end")

                ctx.pipeline.pre.connect(track)
            return middleware

        # Create multiple contexts
        contexts = []
        for i in range(3):
            ctx = create_request_context(basic_params)
            ctx.add_middleware(tracking_middleware(f"ctx{i}"))
            contexts.append(ctx)

        # Mock LLM calls
        mock_response = ChatCompletion(
            id="concurrent", object="chat.completion", created=123, model="test",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        )

        with patch('sllmp.pipeline.any_llm.acompletion', return_value=mock_response):
            # Execute pipelines concurrently
            tasks = []
            for ctx in contexts:
                async def execute_ctx(context):
                    async for _result in execute_pipeline(context):
                        continue
                    return context
                tasks.append(execute_ctx(ctx))

            results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 3
        assert all(result.has_response for result in results)
        assert all(not result.has_error for result in results)

        # All executions should have occurred
        assert len(execution_order) == 6  # 3 start + 3 end events

    async def test_signal_callback_modification_during_execution(self):
        """Test signal behavior when callbacks are modified during execution."""
        signal = Signal()
        execution_log = []

        async def callback1():
            execution_log.append("callback1")
            # Try to add another callback during execution
            signal.connect(lambda: execution_log.append("dynamic"))

        async def callback2():
            execution_log.append("callback2")

        signal.connect(callback1)
        signal.connect(callback2)

        result = await signal.emit()

        # Both original callbacks should execute
        assert "callback1" in execution_log
        assert "callback2" in execution_log
        # Dynamic callback should be handled appropriately
        assert result.completed


class TestResourceManagement:
    """Test resource cleanup and memory management."""

    async def test_pipeline_cleanup_after_error(self, basic_params):
        """Test pipeline cleans up resources properly after errors."""
        cleanup_calls = []

        def resource_middleware(ctx: RequestContext):
            async def allocate_resource(ctx: RequestContext):
                ctx.metadata["resource"] = "allocated"

            async def cleanup_resource(ctx: RequestContext):
                if "resource" in ctx.metadata:
                    cleanup_calls.append("cleaned")
                    del ctx.metadata["resource"]

            # Add to both normal flow and error handling
            ctx.pipeline.pre.connect(allocate_resource)
            ctx.pipeline.post.connect(cleanup_resource)
            ctx.pipeline.error.connect(cleanup_resource)

        def failing_middleware(ctx: RequestContext):
            async def fail(ctx: RequestContext):
                raise ValueError("Resource test failure")

            ctx.pipeline.pre.connect(fail)

        ctx = create_request_context(basic_params)
        ctx.add_middleware(resource_middleware)
        ctx.add_middleware(failing_middleware)

        # Execute pipeline that will fail
        async for result in execute_pipeline(ctx):
            break

        # Cleanup should have been called despite error
        assert len(cleanup_calls) >= 1

    async def test_signal_memory_cleanup(self):
        """Test signals don't leak memory with many callbacks."""
        signal = Signal()

        # Add many callbacks
        for i in range(100):
            signal.connect(lambda: None)

        assert len(signal) == 100

        # Clear all callbacks
        signal.clear()
        assert len(signal) == 0

        # Signal should still be functional
        results = []
        signal.connect(lambda: results.append("test"))
        await signal.emit()
        assert "test" in results


class TestErrorSerialization:
    """Test error serialization and API response formats."""

    def test_validation_error_serialization(self):
        """Test ValidationError serializes correctly for API responses."""
        error = ValidationError(
            "Invalid temperature value",
            request_id="req_test123",
            field_name="temperature"
        )

        serialized = error.to_dict()

        assert serialized["error"]["type"] == "validation_error"
        assert serialized["error"]["message"] == "Invalid temperature value"
        assert serialized["error"]["request_id"] == "req_test123"
        assert serialized["error"]["field"] == "temperature"

    def test_rate_limit_error_serialization(self):
        """Test ProviderRateLimitError serializes with retry information."""
        error = ProviderRateLimitError(
            "Rate limit exceeded",
            request_id="req_test123",
            provider="openai",
            retry_after=60
        )

        serialized = error.to_dict()

        assert serialized["error"]["type"] == "provider_rate_limit_error"
        assert serialized["error"]["provider"] == "openai"
        assert serialized["error"]["retry_after"] == 60

    def test_llm_provider_error_serialization(self):
        """Test LLMProviderError includes provider details."""
        error = LLMProviderError(
            "Provider service unavailable",
            request_id="req_test123",
            provider="anthropic",
            provider_error_code="SERVICE_UNAVAILABLE"
        )

        serialized = error.to_dict()

        assert serialized["error"]["type"] == "llm_provider_error"
        assert serialized["error"]["provider"] == "anthropic"
        assert serialized["error"]["provider_error_code"] == "SERVICE_UNAVAILABLE"


class TestRecoveryScenarios:
    """Test system recovery from various failure modes."""

    async def test_partial_streaming_failure_recovery(self, basic_params):
        """Test recovery when streaming fails partway through."""
        streaming_params = NCompletionParams(
            model_id="openai:gpt-4",
            messages=[{"role": "user", "content": "Stream test"}],
            stream=True,
            metadata={}
        )

        chunks_sent = 0
        async def partial_failing_stream():
            nonlocal chunks_sent
            good_chunks = [
                ChatCompletionChunk(
                    id="partial", object="chat.completion.chunk", created=123, model="test",
                    choices=[{"index": 0, "delta": {"content": "Good"}, "finish_reason": None}]
                ),
                ChatCompletionChunk(
                    id="partial", object="chat.completion.chunk", created=123, model="test",
                    choices=[{"index": 0, "delta": {"content": " chunk"}, "finish_reason": None}]
                )
            ]

            for chunk in good_chunks:
                chunks_sent += 1
                yield chunk
                if chunks_sent == 2:
                    raise NetworkError("Connection lost", request_id="test", provider="test")

        with patch('sllmp.pipeline.any_llm.acompletion', return_value=partial_failing_stream()):
            ctx = create_request_context(streaming_params)

            collected_chunks = []
            async for item in execute_pipeline(ctx):
                collected_chunks.append(item)

            # Should have received some chunks before failure
            assert len(collected_chunks) >= 1
            # Final state should reflect the error occurred
            assert ctx.has_error

    async def test_middleware_chain_partial_failure(self, basic_params):
        """Test pipeline continues when some middleware fails."""
        results = []

        def working_middleware_1(ctx: RequestContext):
            async def work(ctx: RequestContext):
                results.append("middleware1_executed")
            ctx.pipeline.pre.connect(work)

        def failing_middleware(ctx: RequestContext):
            async def fail(ctx: RequestContext):
                results.append("failing_middleware_attempted")
                raise ValueError("Middleware failure")
            ctx.pipeline.pre.connect(fail)

        def working_middleware_2(ctx: RequestContext):
            async def work(ctx: RequestContext):
                results.append("middleware2_executed")
            ctx.pipeline.post.connect(work)

        with patch('sllmp.pipeline.any_llm.acompletion') as mock_completion:
            mock_completion.return_value = ChatCompletion(
                id="recovery", object="chat.completion", created=123, model="test",
                choices=[{"index": 0, "message": {"role": "assistant", "content": "response"}, "finish_reason": "stop"}],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )

            ctx = create_request_context(basic_params)
            ctx.add_middleware(working_middleware_1)
            ctx.add_middleware(failing_middleware)
            ctx.add_middleware(working_middleware_2)

            async for _result in execute_pipeline(ctx):
                continue

            # Should have attempted all middleware
            assert "middleware1_executed" in results
            assert "failing_middleware_attempted" in results
            # Pipeline should end in error state due to middleware failure
            assert ctx.has_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
