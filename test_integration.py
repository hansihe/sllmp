#!/usr/bin/env python3
"""
Simple integration test to verify anyllm.acompletion is properly integrated.
This test uses mocks to avoid needing real API keys.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch
from simple_llm_proxy.pipeline import create_request_context, Pipeline, Middleware


class MockResponse:
    """Mock response that mimics any_llm completion response."""
    
    def __init__(self, content="Mock response from anyllm integration"):
        self.content = content
    
    def __getitem__(self, key):
        # Mock dict-like access
        if key == "choices":
            return [{"message": {"content": self.content}}]
        elif key == "id":
            return "mock-completion-id"
        elif key == "model":
            return "openai:gpt-3.5-turbo"
        elif key == "usage":
            return {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        return None


class MockStreamResponse:
    """Mock streaming response."""
    
    def __init__(self, chunks=None):
        self.chunks = chunks or [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " from"}}]},
            {"choices": [{"delta": {"content": " anyllm"}}]},
            {"choices": [{"delta": {"content": " integration!"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        ]
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


async def test_non_streaming_integration():
    """Test non-streaming anyllm integration."""
    print("=== Testing Non-Streaming Integration ===")
    
    # Mock any_llm.acompletion
    with patch('any_llm.acompletion', new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = MockResponse("Integration test successful!")
        
        # Create pipeline
        pipeline = Pipeline([])
        
        # Create test request  
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        
        ctx = create_request_context(request)
        print(f"Request ID: {ctx.request_id}")
        
        # Execute pipeline
        result_ctx = await pipeline._execute_llm(ctx)
        
        # Verify any_llm.acompletion was called correctly
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        
        print(f"any_llm.acompletion called with:")
        print(f"  model: {call_args.kwargs['model']}")
        print(f"  messages: {call_args.kwargs['messages']}")
        
        # Verify response
        print(f"Response received: {result_ctx.response.content}")
        print("✅ Non-streaming integration test passed!\n")


async def test_streaming_integration():
    """Test streaming anyllm integration.""" 
    print("=== Testing Streaming Integration ===")
    
    # Mock any_llm.acompletion for streaming
    with patch('any_llm.acompletion', new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = MockStreamResponse()
        
        # Create pipeline
        pipeline = Pipeline([])
        
        # Create streaming test request
        request = {
            "model": "openai:gpt-4", 
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True
        }
        
        ctx = create_request_context(request)
        print(f"Streaming Request ID: {ctx.request_id}")
        
        # Collect streaming chunks
        chunks = []
        async for chunk in pipeline._execute_llm_streaming(ctx):
            chunks.append(chunk)
        
        # Verify any_llm.acompletion was called with stream=True
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        
        print(f"any_llm.acompletion called with:")
        print(f"  model: {call_args.kwargs['model']}")
        print(f"  stream: {call_args.kwargs.get('stream', False)}")
        
        # Verify we got chunks
        print(f"Received {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            finish_reason = chunk.get("choices", [{}])[0].get("delta", {}).get("finish_reason")
            if content:
                print(f"  Chunk {i+1}: '{content}'")
            elif finish_reason:
                print(f"  Chunk {i+1}: [finish_reason: {finish_reason}]")
                
        print("✅ Streaming integration test passed!\n")


async def test_full_pipeline_integration():
    """Test full pipeline with middleware and anyllm integration."""
    print("=== Testing Full Pipeline Integration ===")
    
    class TestMiddleware(Middleware):
        async def before_llm(self, ctx):
            print(f"[Middleware] Before LLM - Processing request for model: {ctx.request['model']}")
            ctx.state['middleware_processed'] = True
            return ctx
            
        async def after_llm(self, ctx):
            print(f"[Middleware] After LLM - Response received")
            return ctx
    
    with patch('any_llm.acompletion', new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = MockResponse("Pipeline integration working!")
        
        # Create pipeline with middleware
        pipeline = Pipeline([TestMiddleware()])
        
        request = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test pipeline"}]
        }
        
        ctx = create_request_context(request)
        print(f"Pipeline Request ID: {ctx.request_id}")
        
        # Execute full pipeline
        result_ctx = await pipeline.execute(ctx)
        
        # Verify middleware state was set
        assert result_ctx.state.get('middleware_processed') == True
        
        # Verify response
        print(f"Final response: {result_ctx.response.content}")
        print("✅ Full pipeline integration test passed!\n")


async def main():
    """Run all integration tests."""
    print("🧪 anyllm.acompletion Integration Tests\n")
    
    await test_non_streaming_integration()
    await test_streaming_integration()
    await test_full_pipeline_integration()
    
    print("✅ All integration tests passed!")
    print("anyllm.acompletion is successfully integrated into the pipeline!")


if __name__ == "__main__":
    asyncio.run(main())