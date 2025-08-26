# Streaming Pipeline Architecture

## The Streaming Challenge

Streaming responses fundamentally change how the pipeline works because:

1. **Response is a stream, not a complete object** - Can't transform the full response after LLM call
2. **Real-time processing** - Chunks must be processed and forwarded immediately
3. **Partial failures** - Stream can fail mid-way through
4. **Backpressure** - Need to handle client consumption rate
5. **Stateful transformations** - Some transformations need to accumulate state across chunks

## Core Streaming Architecture

### 1. Enhanced RequestContext for Streaming

```python
@dataclass
class RequestContext:
    # ... existing fields ...
    
    # Streaming-specific fields
    is_streaming: bool = False
    stream_response: Optional[AsyncGenerator[Dict, None]] = None
    stream_metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    streaming_state: Dict[str, Any] = field(default_factory=dict)  # State across chunks
    
    # Stream control
    stream_buffer: List[Dict] = field(default_factory=list)  # For buffering middleware
    should_buffer: bool = False  # Whether to buffer chunks
    flush_trigger: Optional[Callable] = None  # When to flush buffer
```

### 2. Streaming-Aware Middleware Interface

```python
class StreamingMiddleware(Middleware):
    """Extended middleware interface for streaming support"""
    
    # Standard middleware methods (unchanged)
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    # NEW: Streaming-specific methods
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        """Process individual chunks as they arrive"""
        yield chunk
    
    async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
        """Called when streaming begins"""
        return ctx
    
    async def on_stream_end(self, ctx: RequestContext) -> RequestContext:
        """Called when stream completes"""
        return ctx
    
    async def on_stream_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        """Called when stream encounters error"""
        return ctx
    
    # Metadata about streaming behavior
    def requires_buffering(self) -> bool:
        """Whether this middleware needs to buffer chunks"""
        return False
    
    def can_process_partial(self) -> bool:
        """Whether this middleware can work on partial responses"""
        return True
```

### 3. Streaming Pipeline Execution

```python
class StreamingPipeline:
    async def execute_streaming(self, ctx: RequestContext) -> AsyncGenerator[Dict, None]:
        """Execute pipeline for streaming responses"""
        
        # Phase 1: Before LLM (same as non-streaming)
        ctx = await self._run_before_phase(ctx)
        
        if ctx.action != PipelineAction.CONTINUE:
            # Early termination - return static response
            if ctx.response:
                yield ctx.response
            return
        
        # Phase 2: Execute LLM and get stream
        try:
            llm_stream = await self._get_llm_stream(ctx)
            
            # Notify middleware that streaming is starting
            for middleware in self.streaming_middleware:
                ctx = await middleware.on_stream_start(ctx)
            
            # Phase 3: Process stream through middleware chain
            async for processed_chunk in self._process_stream(ctx, llm_stream):
                yield processed_chunk
            
            # Notify middleware that streaming is complete
            for middleware in self.streaming_middleware:
                ctx = await middleware.on_stream_end(ctx)
                
        except Exception as e:
            # Handle streaming errors
            ctx = await self._handle_stream_error(ctx, e)
            if ctx.response:
                yield ctx.response
    
    async def _process_stream(self, ctx: RequestContext, llm_stream: AsyncGenerator) -> AsyncGenerator[Dict, None]:
        """Process each chunk through the middleware chain"""
        
        try:
            async for chunk in llm_stream:
                ctx.chunk_count += 1
                
                # Process chunk through each middleware
                processed_chunks = self._process_chunk_through_middleware(ctx, chunk)
                
                async for processed_chunk in processed_chunks:
                    yield processed_chunk
                    
        except Exception as e:
            # Handle mid-stream errors
            for middleware in self.streaming_middleware:
                ctx = await middleware.on_stream_error(ctx, e)
            
            # Yield error chunk if needed
            if ctx.action == PipelineAction.HALT and ctx.response:
                yield ctx.response
    
    async def _process_chunk_through_middleware(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        """Process a single chunk through all streaming middleware"""
        
        current_stream = self._single_chunk_generator(chunk)
        
        # Chain middleware processing
        for middleware in self.streaming_middleware:
            current_stream = self._apply_middleware_to_stream(middleware, ctx, current_stream)
        
        # Yield processed chunks
        async for processed_chunk in current_stream:
            yield processed_chunk
    
    async def _apply_middleware_to_stream(self, middleware: StreamingMiddleware, 
                                        ctx: RequestContext, 
                                        stream: AsyncGenerator) -> AsyncGenerator[Dict, None]:
        """Apply single middleware to stream"""
        async for chunk in stream:
            async for processed_chunk in middleware.process_chunk(ctx, chunk):
                yield processed_chunk
```

## Streaming Middleware Patterns

### 1. Pass-Through Middleware (No Buffering)
```python
class StreamingObservabilityMiddleware(StreamingMiddleware):
    """Log chunks as they pass through"""
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        # Log the chunk
        await self._log_chunk(ctx.request_id, chunk)
        
        # Pass through unchanged
        yield chunk
    
    def can_process_partial(self) -> bool:
        return True
```

### 2. Buffering Middleware (Accumulates Before Processing)
```python
class StreamingContentFilterMiddleware(StreamingMiddleware):
    """Filter inappropriate content - needs full response to analyze"""
    
    def requires_buffering(self) -> bool:
        return True
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        # Buffer the chunk
        ctx.stream_buffer.append(chunk)
        
        # Only process when we have complete response or hit trigger
        if self._should_flush(ctx, chunk):
            # Analyze complete buffered content
            filtered_content = await self._filter_content(ctx.stream_buffer)
            
            # Yield all filtered chunks
            for filtered_chunk in filtered_content:
                yield filtered_chunk
            
            # Clear buffer
            ctx.stream_buffer.clear()
    
    def _should_flush(self, ctx: RequestContext, chunk: Dict) -> bool:
        # Flush on final chunk or when buffer is full
        return (chunk.get('choices', [{}])[0].get('finish_reason') is not None or 
                len(ctx.stream_buffer) >= self.config.get('buffer_size', 50))
```

### 3. Stateful Transformation Middleware
```python
class StreamingPromptInjectionMiddleware(StreamingMiddleware):
    """Inject prompts into the stream at specific points"""
    
    async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
        ctx.streaming_state['injected_intro'] = False
        return ctx
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        # Inject introduction after first few chunks
        if not ctx.streaming_state['injected_intro'] and ctx.chunk_count >= 3:
            intro_chunk = self._create_injection_chunk("Let me think about this...")
            yield intro_chunk
            ctx.streaming_state['injected_intro'] = True
        
        # Process and yield original chunk
        yield chunk
```

### 4. Error Recovery Middleware
```python
class StreamingRetryMiddleware(StreamingMiddleware):
    """Handle mid-stream failures with graceful recovery"""
    
    async def on_stream_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        if self._is_retryable_error(error) and ctx.metadata.get('stream_retry_count', 0) < 3:
            # Log the partial response we got
            await self._log_partial_response(ctx.stream_buffer)
            
            # Mark for retry with continuation
            ctx.action = PipelineAction.RETRY
            ctx.metadata['stream_retry_count'] = ctx.metadata.get('stream_retry_count', 0) + 1
            ctx.streaming_state['retry_from_chunk'] = ctx.chunk_count
            
            # Generate recovery chunk
            ctx.response = {
                "choices": [{
                    "delta": {"content": "\n[Connection restored, continuing...]\n"},
                    "finish_reason": None
                }]
            }
        return ctx
```

## Streaming-Specific Challenges & Solutions

### 1. **Backpressure Handling**
```python
class BackpressureMiddleware(StreamingMiddleware):
    """Manage flow control between provider and client"""
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        # Monitor buffer sizes and apply backpressure
        if ctx.stream_metadata.get('client_buffer_size', 0) > self.config['max_buffer']:
            await asyncio.sleep(0.1)  # Small delay to allow client to catch up
        
        yield chunk
```

### 2. **Partial Response Recovery**
```python
class StreamingFallbackMiddleware(StreamingMiddleware):
    """Switch providers mid-stream if possible"""
    
    async def on_stream_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        if self._can_continue_with_fallback(ctx, error):
            # Switch to fallback provider
            fallback_provider = ctx.fallback_providers.pop(0)
            
            # Create continuation prompt from what we've seen so far
            continuation_request = self._build_continuation_request(ctx)
            
            ctx.selected_provider = fallback_provider
            ctx.request = continuation_request
            ctx.action = PipelineAction.FALLBACK
            
            # Generate transition chunk
            ctx.response = self._create_transition_chunk()
        
        return ctx
```

### 3. **Cost Tracking for Streams**
```python
class StreamingBudgetMiddleware(StreamingMiddleware):
    """Track costs in real-time during streaming"""
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        # Estimate incremental cost
        chunk_cost = self._estimate_chunk_cost(chunk)
        current_cost = ctx.stream_metadata.get('running_cost', 0)
        new_total = current_cost + chunk_cost
        
        # Check if we're approaching budget limit
        if new_total > self.config['budget_warning_threshold']:
            # Inject warning chunk
            warning_chunk = self._create_budget_warning_chunk()
            yield warning_chunk
        
        # Check hard limit
        if new_total > self.config['budget_hard_limit']:
            # Terminate stream gracefully
            termination_chunk = self._create_budget_termination_chunk()
            yield termination_chunk
            return  # Stop processing
        
        ctx.stream_metadata['running_cost'] = new_total
        yield chunk
```

## Integration with Main Pipeline

```python
async def chat_completions(request: Request):
    try:
        body = await request.json()
        ctx = RequestContext(
            original_request=body,
            request=body.copy(),
            is_streaming=body.get('stream', False),
            # ... other fields
        )
        
        if ctx.is_streaming:
            # Use streaming pipeline
            async def stream_generator():
                async for chunk in app.streaming_pipeline.execute_streaming(ctx):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
                headers={"Content-Type": "text/plain; charset=utf-8"}
            )
        else:
            # Use regular pipeline
            ctx = await app.pipeline.execute(ctx)
            return JSONResponse(ctx.response)
            
    except Exception as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=500)
```

## Key Benefits

1. **Real-time Processing**: Chunks processed immediately, no waiting for complete response
2. **Flexible Buffering**: Middleware can choose to buffer or process immediately
3. **Graceful Error Recovery**: Mid-stream failures can be handled elegantly
4. **Stateful Transformations**: Middleware can maintain state across chunks
5. **Flow Control**: Built-in backpressure handling
6. **Cost Control**: Real-time budget monitoring and enforcement

This architecture maintains the composability benefits of the pipeline while properly handling the complexities of streaming responses.