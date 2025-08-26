#!/usr/bin/env python3
"""
Demo script showing the pipeline architecture in action.

This script demonstrates:
1. Basic pipeline setup with middleware
2. Non-streaming and streaming responses
3. Middleware processing (auth, guardrails, logging, etc.)
4. Error handling and pipeline control flow
"""

import asyncio
import json
from simple_llm_proxy.builder import PipelineBuilder
from simple_llm_proxy.pipeline import create_request_context, PipelineAction
from simple_llm_proxy.middleware import (
    AuthMiddleware, 
    ContentGuardrailMiddleware,
    LoggingMiddleware,
    RoutingMiddleware
)

async def demo_basic_pipeline():
    """Demonstrate basic pipeline functionality."""
    print("=== Basic Pipeline Demo ===")
    
    # Create a simple pipeline
    pipeline = (PipelineBuilder()
        .add(AuthMiddleware(require_user_id=False))  # Allow anonymous for demo
        .add(RoutingMiddleware(strategy="cost_optimized"))
        .add(LoggingMiddleware())
        .build())
    
    # Create a test request
    openai_request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "metadata": {"user_id": "demo_user", "organization": "demo_org"}
    }
    
    ctx = create_request_context(openai_request)
    
    print(f"Request ID: {ctx.request_id}")
    print(f"Client metadata: {ctx.client_metadata}")
    
    # Execute pipeline
    result_ctx = await pipeline.execute(ctx)
    
    print(f"Pipeline action: {result_ctx.action}")
    print(f"Response: {json.dumps(result_ctx.response, indent=2)}")
    print(f"State: {result_ctx.state}")
    print(f"Metadata: {result_ctx.metadata}")
    print()

async def demo_streaming_pipeline():
    """Demonstrate streaming pipeline functionality."""
    print("=== Streaming Pipeline Demo ===")
    
    pipeline = (PipelineBuilder()
        .add(AuthMiddleware(require_user_id=False))
        .add(ContentGuardrailMiddleware(policies=["inappropriate"], check_interval=2))
        .add(LoggingMiddleware())
        .set_monitoring_interval(2)  # Check content every 2 chunks
        .build())
    
    streaming_request = {
        "model": "openai:gpt-4",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
        "metadata": {"user_id": "demo_user"}
    }
    
    ctx = create_request_context(streaming_request)
    print(f"Streaming request ID: {ctx.request_id}")
    
    chunk_count = 0
    async for chunk in pipeline.execute_streaming(ctx):
        chunk_count += 1
        print(f"Chunk {chunk_count}: {json.dumps(chunk)}")
        
        # Stop after a few chunks for demo
        if chunk_count >= 5:
            break
    
    print()

async def demo_guardrail_blocking():
    """Demonstrate content guardrails blocking inappropriate content."""
    print("=== Guardrail Demo (Content Blocking) ===")
    
    pipeline = (PipelineBuilder()
        .add(AuthMiddleware(require_user_id=False))
        .add(ContentGuardrailMiddleware(policies=["inappropriate"], check_interval=1))
        .build())
    
    # Create a request that will trigger guardrails
    bad_request = {
        "model": "openai:gpt-3.5-turbo", 
        "messages": [{"role": "user", "content": "Tell me about inappropriate content"}],
        "stream": True
    }
    
    ctx = create_request_context(bad_request)
    print(f"Request ID: {ctx.request_id}")
    
    chunk_count = 0
    async for chunk in pipeline.execute_streaming(ctx):
        chunk_count += 1
        print(f"Chunk {chunk_count}: {json.dumps(chunk)}")
        
        # Should stop early due to guardrail
        if chunk_count >= 10:  # Safety break
            break
    
    print()

async def demo_pipeline_state():
    """Demonstrate how middleware shares state through the pipeline."""
    print("=== Pipeline State Demo ===")
    
    class StateTrackingMiddleware:
        def __init__(self, name):
            self.name = name
        
        async def before_llm(self, ctx):
            print(f"[{self.name}] Before LLM - State: {ctx.state}")
            ctx.state[self.name] = {"processed": True, "timestamp": "now"}
            return ctx
        
        async def after_llm(self, ctx):
            print(f"[{self.name}] After LLM - State: {ctx.state}")
            return ctx
    
    # Create custom pipeline with state tracking
    from simple_llm_proxy.pipeline import Middleware
    
    class StateMiddleware1(Middleware):
        async def before_llm(self, ctx):
            ctx.state['middleware1'] = {"step": "preprocessing", "data": "some_value"}
            print(f"Middleware 1 set state: {ctx.state}")
            return ctx
        
        async def after_llm(self, ctx):
            # Access state from other middleware
            other_data = ctx.state.get('middleware2', {})
            print(f"Middleware 1 sees state from middleware 2: {other_data}")
            return ctx
    
    class StateMiddleware2(Middleware):
        async def before_llm(self, ctx):
            # Access state from previous middleware
            prev_data = ctx.state.get('middleware1', {})
            print(f"Middleware 2 sees state from middleware 1: {prev_data}")
            
            ctx.state['middleware2'] = {"step": "routing", "provider": "openai"}
            return ctx
    
    pipeline = (PipelineBuilder()
        .add(StateMiddleware1())
        .add(StateMiddleware2())
        .build())
    
    request = {
        "model": "openai:gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    ctx = create_request_context(request)
    result_ctx = await pipeline.execute(ctx)
    
    print(f"Final state: {result_ctx.state}")
    print()

async def main():
    """Run all demos."""
    print("🚀 Pipeline Architecture Demo\n")
    
    await demo_basic_pipeline()
    await demo_streaming_pipeline()
    await demo_guardrail_blocking()
    await demo_pipeline_state()
    
    print("✅ Demo complete!")

if __name__ == "__main__":
    asyncio.run(main())