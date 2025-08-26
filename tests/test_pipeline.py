"""
Comprehensive test suite for the pipeline implementation.

Tests cover:
1. Core pipeline functionality
2. Middleware execution order
3. Request context flow
4. Error handling 
5. Streaming vs non-streaming behavior
6. anyllm.acompletion integration
7. Dynamic middleware extension
8. Pipeline state management
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List, AsyncGenerator

from simple_llm_proxy.pipeline import (
    Pipeline, 
    Middleware, 
    RequestContext, 
    PipelineAction,
    create_request_context
)
from simple_llm_proxy.error import ValidationError


class MockResponse:
    """Mock response that mimics any_llm completion response."""
    
    def __init__(self, content="Mock response", model="openai:gpt-3.5-turbo"):
        self.content = content
        self.model = model
        # Mock the ChatCompletion structure
        self.choices = [MockChoice(content)]
        self._data = {
            "id": "mock-completion-id",
            "object": "chat.completion",
            "model": model,
            "choices": [{"message": {"content": content, "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)

class MockChoice:
    """Mock choice for ChatCompletion."""
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    """Mock message for ChatCompletion choice."""
    def __init__(self, content):
        self.content = content


class MockStreamResponse:
    """Mock streaming response."""
    
    def __init__(self, chunks=None):
        self.raw_chunks = chunks or [
            {"id": "chunk-1", "choices": [{"delta": {"content": "Hello"}, "index": 0}]},
            {"id": "chunk-2", "choices": [{"delta": {"content": " from"}, "index": 0}]},
            {"id": "chunk-3", "choices": [{"delta": {"content": " stream"}, "index": 0}]},
            {"id": "chunk-4", "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}
        ]
        # Convert to proper chunk objects
        self.chunks = [MockChunk(chunk) for chunk in self.raw_chunks]
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

class MockChunk:
    """Mock ChatCompletionChunk."""
    def __init__(self, data):
        self._data = data
        self.choices = [MockChunkChoice(data["choices"][0])]
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)

class MockChunkChoice:
    """Mock choice for streaming chunk."""
    def __init__(self, choice_data):
        self._data = choice_data
        self.delta = MockChunkDelta(choice_data.get("delta", {}))
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)

class MockChunkDelta:
    """Mock delta for streaming chunk."""
    def __init__(self, delta_data):
        self._data = delta_data
        self.content = delta_data.get("content")
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)


class TrackingMiddleware(Middleware):
    """Middleware that tracks execution for testing."""
    
    def __init__(self, name: str, **config):
        super().__init__(**config)
        self.name = name
        self.calls = []
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        self.calls.append(f"{self.name}.before_llm")
        ctx.state[f"{self.name}_before"] = True
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        self.calls.append(f"{self.name}.after_llm")
        ctx.state[f"{self.name}_after"] = True
        return ctx
    
    async def on_response_update(self, ctx: RequestContext, accumulated_content: str) -> RequestContext:
        self.calls.append(f"{self.name}.on_response_update")
        ctx.state[f"{self.name}_monitored"] = len(accumulated_content)
        return ctx
    
    async def on_response_complete(self, ctx: RequestContext, final_content: str) -> RequestContext:
        self.calls.append(f"{self.name}.on_response_complete")
        ctx.state[f"{self.name}_completed"] = len(final_content)
        return ctx
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        self.calls.append(f"{self.name}.process_chunk")
        # Add metadata to chunk
        chunk[f"{self.name}_processed"] = True
        yield chunk
    
    async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
        self.calls.append(f"{self.name}.on_stream_start")
        return ctx
    
    async def on_stream_end(self, ctx: RequestContext) -> RequestContext:
        self.calls.append(f"{self.name}.on_stream_end")
        return ctx
    
    async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        self.calls.append(f"{self.name}.on_error")
        return ctx


class HaltingMiddleware(Middleware):
    """Middleware that halts the pipeline for testing."""
    
    def __init__(self, halt_at="before_llm", reason="test_halt"):
        self.halt_at = halt_at
        self.reason = reason
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        if self.halt_at == "before_llm":
            error = ValidationError(message=self.reason, request_id=ctx.request_id)
            self.halt_with_error(ctx, error)
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        if self.halt_at == "after_llm":
            error = ValidationError(message=self.reason, request_id=ctx.request_id)
            self.halt_with_error(ctx, error)
        return ctx


class MonitoringMiddleware(Middleware):
    """Middleware that monitors responses for testing."""
    
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.update_count = 0
    
    async def on_response_update(self, ctx: RequestContext, accumulated_content: str) -> RequestContext:
        self.update_count += 1
        if len(accumulated_content) > self.threshold:
            error = ValidationError(message="Content too long", request_id=ctx.request_id)
            self.halt_with_error(ctx, error)
        return ctx
    
    def monitors_response(self) -> bool:
        return True


class ChunkProcessingMiddleware(Middleware):
    """Middleware that processes chunks for testing."""
    
    def __init__(self, transform_fn=None):
        self.transform_fn = transform_fn or (lambda x: x.upper())
        self.processed_chunks = []
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        self.processed_chunks.append(chunk)
        
        # Transform content if present
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                original_content = choice["delta"]["content"]
                if original_content:
                    transformed = dict(chunk)
                    transformed["choices"] = [{
                        **choice,
                        "delta": {**choice["delta"], "content": self.transform_fn(original_content)}
                    }]
                    yield transformed
                    return
        
        yield chunk
    
    def processes_chunks(self) -> bool:
        return True


@pytest.fixture
def basic_request():
    return {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }


@pytest.fixture
def streaming_request():
    return {
        "model": "openai:gpt-4",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    }


class TestRequestContext:
    """Test RequestContext functionality."""
    
    def test_create_request_context(self, basic_request):
        ctx = create_request_context(basic_request)
        
        assert ctx.original_request == basic_request
        assert ctx.request == basic_request
        assert ctx.action == PipelineAction.CONTINUE
        assert ctx.is_streaming == False
        assert ctx.chunk_count == 0
        assert len(ctx.request_id) > 0
        assert ctx.request_id.startswith("req_")
    
    def test_create_streaming_context(self, streaming_request):
        ctx = create_request_context(streaming_request)
        
        assert ctx.is_streaming == True
        assert ctx.request["stream"] == True
    
    def test_context_with_metadata(self, basic_request):
        metadata = {"user_id": "test_user", "org": "test_org"}
        basic_request["metadata"] = metadata
        
        ctx = create_request_context(basic_request, extra_key="extra_value")
        
        assert ctx.client_metadata["user_id"] == "test_user"
        assert ctx.client_metadata["org"] == "test_org"
        assert ctx.client_metadata["extra_key"] == "extra_value"
    
    def test_context_extend_pipeline(self):
        ctx = RequestContext(
            original_request={},
            request={}
        )
        
        middleware = TrackingMiddleware("test")
        ctx.extend_pipeline(middleware)
        
        assert len(ctx._extended_middleware) == 1
        assert ctx._extended_middleware[0] == middleware
    
    def test_context_content_accumulation(self):
        ctx = RequestContext(
            original_request={},
            request={}
        )
        
        ctx.add_content_chunk("Hello")
        ctx.add_content_chunk(" world")
        ctx.add_content_chunk("!")
        
        assert ctx.get_accumulated_content() == "Hello world!"
        assert ctx.chunk_count == 0  # chunk_count is managed by pipeline


class TestBasicPipeline:
    """Test basic pipeline functionality."""
    
    @patch('any_llm.acompletion')
    async def test_empty_pipeline_non_streaming(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse("Test response")
        
        pipeline = Pipeline([])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        assert result.action == PipelineAction.CONTINUE
        assert result.response.content == "Test response"
        mock_completion.assert_called_once()
    
    @patch('any_llm.acompletion') 
    async def test_empty_pipeline_streaming(self, mock_completion, streaming_request):
        mock_completion.return_value = MockStreamResponse()
        
        pipeline = Pipeline([])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        assert len(chunks) == 4  # 3 content + 1 finish
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        mock_completion.assert_called_once()
    
    @patch('any_llm.acompletion')
    async def test_single_middleware_execution_order(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse("Test response")
        
        middleware = TrackingMiddleware("test")
        pipeline = Pipeline([middleware])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        # The pipeline also calls on_response_complete for complete responses
        expected_calls = ["test.before_llm", "test.on_response_complete", "test.after_llm"]
        assert middleware.calls == expected_calls
        assert result.state["test_before"] == True
        assert result.state["test_after"] == True
    
    @patch('any_llm.acompletion')
    async def test_multiple_middleware_execution_order(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse("Test response")
        
        middleware1 = TrackingMiddleware("m1")
        middleware2 = TrackingMiddleware("m2")
        middleware3 = TrackingMiddleware("m3")
        
        pipeline = Pipeline([middleware1, middleware2, middleware3])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        # Before LLM: forward order
        # After LLM: reverse order
        # Also includes on_response_complete for complete responses
        expected_calls_m1 = ["m1.before_llm", "m1.on_response_complete", "m1.after_llm"]
        expected_calls_m2 = ["m2.before_llm", "m2.on_response_complete", "m2.after_llm"] 
        expected_calls_m3 = ["m3.before_llm", "m3.on_response_complete", "m3.after_llm"]
        
        assert middleware1.calls == expected_calls_m1
        assert middleware2.calls == expected_calls_m2
        assert middleware3.calls == expected_calls_m3
        
        # All state should be set
        for name in ["m1", "m2", "m3"]:
            assert result.state[f"{name}_before"] == True
            assert result.state[f"{name}_after"] == True


class TestPipelineControl:
    """Test pipeline control flow."""
    
    @patch('any_llm.acompletion')
    async def test_halt_before_llm(self, mock_completion, basic_request):
        halting_middleware = HaltingMiddleware("before_llm", "test halt")
        tracking_middleware = TrackingMiddleware("tracker")
        
        pipeline = Pipeline([tracking_middleware, halting_middleware, tracking_middleware])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        assert result.action == PipelineAction.HALT
        assert result.halt_reason == "test halt"
        assert result.has_error
        assert result.error.message == "test halt"
        
        # LLM should not be called
        mock_completion.assert_not_called()
    
    @patch('any_llm.acompletion')
    async def test_halt_after_llm(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse("Test response")
        
        halting_middleware = HaltingMiddleware("after_llm", "post-llm halt")
        pipeline = Pipeline([halting_middleware])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        assert result.action == PipelineAction.HALT
        assert result.halt_reason == "post-llm halt"
        
        # LLM should be called
        mock_completion.assert_called_once()
    
    @patch('any_llm.acompletion')
    async def test_halt_streaming(self, mock_completion, streaming_request):
        mock_completion.return_value = MockStreamResponse()
        
        halting_middleware = HaltingMiddleware("before_llm", "streaming halt")
        pipeline = Pipeline([halting_middleware])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Should get the halt response
        assert len(chunks) == 1
        assert "error" in chunks[0]
        mock_completion.assert_not_called()


class TestStreamingPipeline:
    """Test streaming-specific pipeline behavior."""
    
    @patch('any_llm.acompletion')
    async def test_monitoring_middleware_streaming(self, mock_completion, streaming_request):
        mock_completion.return_value = MockStreamResponse()
        
        monitoring_middleware = MonitoringMiddleware(threshold=50)  # High threshold
        pipeline = Pipeline([monitoring_middleware], monitoring_interval=2)
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Should complete normally
        assert len(chunks) == 4
        assert monitoring_middleware.update_count > 0  # Was called during streaming
    
    @patch('any_llm.acompletion')
    async def test_monitoring_middleware_halt(self, mock_completion, streaming_request):
        # Create chunks that will exceed threshold
        long_chunks = [
            {"choices": [{"delta": {"content": "This is a very long content"}}]},
            {"choices": [{"delta": {"content": " that exceeds the threshold"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        ]
        mock_completion.return_value = MockStreamResponse(long_chunks)
        
        monitoring_middleware = MonitoringMiddleware(threshold=10)  # Low threshold
        pipeline = Pipeline([monitoring_middleware], monitoring_interval=1)
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
            # Break if we get an error response
            if isinstance(chunk, dict) and "error" in chunk:
                break
        
        # Should be halted due to content length
        assert any(isinstance(chunk, dict) and "error" in chunk for chunk in chunks)
    
    @patch('any_llm.acompletion')
    async def test_chunk_processing_middleware(self, mock_completion, streaming_request):
        mock_completion.return_value = MockStreamResponse([
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        ])
        
        processor = ChunkProcessingMiddleware(lambda x: x.upper())
        pipeline = Pipeline([processor])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Content should be transformed
        assert chunks[0]["choices"][0]["delta"]["content"] == "HELLO"
        assert chunks[1]["choices"][0]["delta"]["content"] == " WORLD"
        assert len(processor.processed_chunks) == 3  # Including finish chunk
    
    @patch('any_llm.acompletion')
    async def test_streaming_lifecycle_hooks(self, mock_completion, streaming_request):
        mock_completion.return_value = MockStreamResponse()
        
        middleware = TrackingMiddleware("lifecycle")
        pipeline = Pipeline([middleware])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Should include lifecycle hooks (including on_response_complete for streaming)
        expected_calls = [
            "lifecycle.before_llm",
            "lifecycle.on_stream_start",
            "lifecycle.process_chunk",  # Called for each chunk
            "lifecycle.process_chunk",
            "lifecycle.process_chunk", 
            "lifecycle.process_chunk",
            "lifecycle.on_response_complete",  # Called at end of stream
            "lifecycle.on_stream_end"
        ]
        assert middleware.calls == expected_calls


class TestDynamicMiddleware:
    """Test dynamic middleware extension."""
    
    @patch('any_llm.acompletion')
    async def test_dynamic_middleware_extension_non_streaming(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse("Test response")
        
        # Initial middleware that adds dynamic middleware
        class ExtendingMiddleware(Middleware):
            async def before_llm(self, ctx: RequestContext) -> RequestContext:
                # Dynamically add middleware
                dynamic_middleware = TrackingMiddleware("dynamic")
                ctx.extend_pipeline(dynamic_middleware)
                return ctx
        
        extending_middleware = ExtendingMiddleware()
        initial_middleware = TrackingMiddleware("initial")
        
        pipeline = Pipeline([initial_middleware, extending_middleware])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        # Both initial and dynamic middleware should execute
        assert result.state.get("initial_before") == True
        assert result.state.get("initial_after") == True
        assert result.state.get("dynamic_before") == True
        assert result.state.get("dynamic_after") == True
    
    @patch('any_llm.acompletion')
    async def test_dynamic_middleware_extension_streaming(self, mock_completion, streaming_request):
        mock_completion.return_value = MockStreamResponse()
        
        class ExtendingMiddleware(Middleware):
            async def before_llm(self, ctx: RequestContext) -> RequestContext:
                # Add a chunk processor dynamically
                processor = ChunkProcessingMiddleware(lambda x: f"[DYNAMIC] {x}")
                ctx.extend_pipeline(processor)
                return ctx
        
        extending_middleware = ExtendingMiddleware()
        pipeline = Pipeline([extending_middleware])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Content should be transformed by dynamic middleware
        content_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
        assert all("[DYNAMIC]" in chunk["choices"][0]["delta"]["content"] 
                  for chunk in content_chunks)


class TestAnyllmIntegration:
    """Test anyllm.acompletion integration."""
    
    @patch('any_llm.acompletion')
    async def test_non_streaming_anyllm_call(self, mock_completion, basic_request):
        mock_response = MockResponse("anyllm response")
        mock_completion.return_value = mock_response
        
        pipeline = Pipeline([])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        # Verify anyllm was called correctly
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        
        assert call_args.kwargs['model'] == "openai:gpt-3.5-turbo"
        assert call_args.kwargs['messages'] == [{"role": "user", "content": "Hello"}]
        assert result.response == mock_response
    
    @patch('any_llm.acompletion')
    async def test_streaming_anyllm_call(self, mock_completion, streaming_request):
        mock_stream = MockStreamResponse()
        mock_completion.return_value = mock_stream
        
        pipeline = Pipeline([])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Verify anyllm was called with streaming
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        
        assert call_args.kwargs['model'] == "openai:gpt-4"
        assert call_args.kwargs['stream'] == True
        assert len(chunks) == 4
    
    @patch('any_llm.acompletion')
    async def test_anyllm_with_additional_parameters(self, mock_completion):
        mock_completion.return_value = MockResponse()
        
        request = {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9
        }
        
        pipeline = Pipeline([])
        ctx = create_request_context(request)
        
        await pipeline.execute(ctx)
        
        call_args = mock_completion.call_args
        assert call_args.kwargs['temperature'] == 0.7
        assert call_args.kwargs['max_tokens'] == 100
        assert call_args.kwargs['top_p'] == 0.9
    
    @patch('any_llm.acompletion')
    async def test_anyllm_error_handling(self, mock_completion, basic_request):
        # Simulate anyllm raising an error
        mock_completion.side_effect = Exception("API Error")
        
        pipeline = Pipeline([])
        ctx = create_request_context(basic_request)
        
        with pytest.raises(Exception, match="API Error"):
            await pipeline.execute(ctx)


class TestErrorHandling:
    """Test error handling in pipeline."""
    
    @patch('any_llm.acompletion')
    async def test_middleware_error_handling(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse()
        
        class ErrorMiddleware(Middleware):
            async def before_llm(self, ctx: RequestContext) -> RequestContext:
                raise ValueError("Middleware error")
        
        error_middleware = ErrorMiddleware()
        pipeline = Pipeline([error_middleware])
        ctx = create_request_context(basic_request)
        
        result = await pipeline.execute(ctx)
        
        assert result.action == PipelineAction.HALT
        assert result.has_error
        assert "Middleware error" in result.error.message
    
    @patch('any_llm.acompletion')
    async def test_streaming_error_handling(self, mock_completion, streaming_request):
        # Simulate streaming error
        class ErrorStream:
            async def __aiter__(self):
                return self
            
            async def __anext__(self):
                raise ValueError("Stream error")
        
        mock_completion.return_value = ErrorStream()
        
        pipeline = Pipeline([])
        ctx = create_request_context(streaming_request)
        
        chunks = []
        async for chunk in pipeline.execute_streaming(ctx):
            chunks.append(chunk)
        
        # Should get error response
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert "Stream error" in chunks[0]["error"]["message"]


class TestPipelineOptimization:
    """Test pipeline optimization features."""
    
    def test_middleware_introspection(self):
        """Test that pipeline correctly identifies middleware capabilities."""
        monitoring_middleware = MonitoringMiddleware()
        chunk_processor = ChunkProcessingMiddleware()
        basic_middleware = TrackingMiddleware("basic")
        
        pipeline = Pipeline([monitoring_middleware, chunk_processor, basic_middleware])
        
        # TrackingMiddleware also implements on_response_update, so it's detected as monitoring
        assert len(pipeline.monitoring_middleware) == 2
        assert monitoring_middleware in pipeline.monitoring_middleware
        
        # TrackingMiddleware also implements process_chunk, so it's detected as chunk processing
        assert len(pipeline.chunk_processing_middleware) == 2
        assert chunk_processor in pipeline.chunk_processing_middleware
    
    @patch('any_llm.acompletion')
    async def test_response_completion_middleware(self, mock_completion, basic_request):
        mock_completion.return_value = MockResponse("Complete response content")
        
        class CompletionMiddleware(Middleware):
            def __init__(self):
                self.final_content = None
            
            async def on_response_complete(self, ctx: RequestContext, final_content: str) -> RequestContext:
                self.final_content = final_content
                return ctx
            
            def needs_complete_response(self) -> bool:
                return True
        
        completion_middleware = CompletionMiddleware()
        pipeline = Pipeline([completion_middleware])
        ctx = create_request_context(basic_request)
        
        await pipeline.execute(ctx)
        
        assert completion_middleware.final_content == "Complete response content"


class TestConcurrency:
    """Test concurrent pipeline execution."""
    
    @patch('any_llm.acompletion')
    async def test_concurrent_executions(self, mock_completion, basic_request):
        """Test that multiple pipeline executions can run concurrently."""
        
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MockResponse(f"Response {id(asyncio.current_task())}")
        
        mock_completion.side_effect = delayed_response
        
        middleware = TrackingMiddleware("concurrent")
        pipeline = Pipeline([middleware])
        
        # Run multiple executions concurrently
        tasks = []
        for i in range(5):
            ctx = create_request_context(basic_request.copy())
            ctx.request["messages"][0]["content"] = f"Request {i}"
            tasks.append(pipeline.execute(ctx))
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert result.action == PipelineAction.CONTINUE
            assert result.response is not None
        
        # Each execution should have its own state
        request_ids = [result.request_id for result in results]
        assert len(set(request_ids)) == 5  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])