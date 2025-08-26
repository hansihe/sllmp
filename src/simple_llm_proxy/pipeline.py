"""
Core pipeline architecture for LLM proxy middleware system.

This module provides the foundation for a composable middleware pipeline
that can handle both streaming and non-streaming LLM requests.
"""

from abc import ABC
from typing import Any, AsyncGenerator, List, Optional, cast

import any_llm
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from typing import AsyncIterator

from simple_llm_proxy.util.signal import SignalExecutionResult

from .error import PipelineError, MiddlewareError, StreamError
from .context import RequestContext, PipelineAction, NCompletionParams, PipelineState

# class OldMiddleware(ABC):
#     """
#     Base middleware class with progressive complexity disclosure.
#
#     Middleware can implement only the methods they need:
#     - Level 0: Just before_llm/after_llm for simple request/response modification
#     - Level 1: Add monitoring callbacks for content inspection
#     - Level 2: Add chunk processing for streaming transformations
#     - Level 3: Add full streaming lifecycle hooks for advanced use cases
#     """
#
#     def __init__(self, **config: Any) -> None:
#         """Initialize middleware with configuration."""
#         self.config = config
#
#     # Level 0: Basic middleware hooks (most common)
#     async def before_llm(self, ctx: RequestContext) -> RequestContext:
#         """
#         Called before LLM execution for request preprocessing.
#
#         This is where you typically do:
#         - Request validation
#         - Authentication/authorization
#         - Rate limiting
#         - Request routing/modification
#         """
#         return ctx
#
#     async def after_llm(self, ctx: RequestContext) -> RequestContext:
#         """
#         Called after LLM execution for response postprocessing.
#
#         For streaming responses, this receives the complete assembled response.
#         This is where you typically do:
#         - Response validation
#         - Logging/metrics
#         - Cost tracking
#         - Response modification
#         """
#         return ctx
#
#     # Level 1: Response monitoring hooks (callback-based)
#     async def on_response_update(self, ctx: RequestContext) -> RequestContext:
#         """
#         Called periodically during streaming with accumulated content so far.
#
#         Override this for real-time content monitoring (e.g., guardrails).
#         The pipeline calls this every N chunks based on monitoring_interval config.
#         """
#         return ctx
#
#     async def on_response_complete(self, ctx: RequestContext) -> RequestContext:
#         """
#         Called once when response is complete (streaming or non-streaming).
#
#         Override this for final validation/processing of complete responses.
#         """
#         return ctx
#
#     # Level 2: Chunk processing (streaming-aware)
#     async def process_chunk(self, ctx: RequestContext, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
#         """
#         Process individual chunks during streaming.
#
#         Override this for per-chunk transformations. Default implementation
#         passes chunks through unchanged.
#         """
#         return chunk
#
#     # Level 3: Full streaming lifecycle (advanced)
#     async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
#         """Called when streaming response begins."""
#         return ctx
#
#     async def on_stream_end(self, ctx: RequestContext) -> RequestContext:
#         """Called when streaming response completes."""
#         return ctx
#
#     async def on_stream_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
#         """Called when streaming encounters an error."""
#         return ctx
#
#     # Error handling
#     async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
#         """
#         Called when an error occurs during pipeline execution.
#
#         Middleware can inspect the error and potentially recover or modify
#         the pipeline behavior.
#         """
#         return ctx
#
#     # Utility methods for common actions
#     def halt_with_error(self, ctx: RequestContext, error: PipelineError) -> None:
#         """Halt the pipeline with an error response."""
#         ctx.action = PipelineAction.HALT
#         ctx.set_error(error)
#         ctx.halt_reason = error.message
#
#     def halt_with_message(self, ctx: RequestContext, message: str) -> None:
#         """Halt the pipeline with an error response."""
#         self.halt_with_error(ctx, MiddlewareError(
#             message=message,
#             request_id=ctx.request_id,
#             middleware_name=self.__name__
#         ))
#
#     def set_retry(self, ctx: RequestContext, reason: str) -> None:
#         """Mark the request for retry."""
#         ctx.action = PipelineAction.RETRY
#         ctx.metadata['retry_reason'] = reason
#
#     # Metadata about middleware behavior (for pipeline optimization)
#     def monitors_response(self) -> bool:
#         """Whether this middleware monitors response content."""
#         return (hasattr(self, 'on_response_update') and
#                 callable(getattr(self.__class__, 'on_response_update', None)) and
#                 getattr(self.__class__, 'on_response_update') is not Middleware.on_response_update)
#
#     def needs_complete_response(self) -> bool:
#         """Whether this middleware needs to see the complete response."""
#         return (hasattr(self, 'on_response_complete') and
#                 callable(getattr(self.__class__, 'on_response_complete', None)) and
#                 getattr(self.__class__, 'on_response_complete') is not Middleware.on_response_complete)
#
#     def processes_chunks(self) -> bool:
#         """Whether this middleware processes individual chunks."""
#         return (hasattr(self, 'process_chunk') and
#                 callable(getattr(self.__class__, 'process_chunk', None)) and
#                 getattr(self.__class__, 'process_chunk') is not Middleware.process_chunk)

async def execute_pipeline(ctx: RequestContext) -> AsyncGenerator[ChatCompletionChunk | RequestContext, None]:
    # Setup stage
    assert(ctx.pipeline_state == PipelineState.SETUP)
    _handle_signal_errors(ctx, await ctx.pipeline.setup.emit(ctx))
    if ctx.has_error:
        yield ctx
        return

    assert ctx.next_pipeline_state == None
    ctx.pipeline_state = PipelineState.PRE

    # After `setup` the pipeline is closed.
    # This means no more signal handlers are allowed to be
    # registered.
    # This is done for predictability, there is no need for a
    # "self modifying pipeline", this may lead to weird results
    # across retries, etc.
    ctx.pipeline.close()

    # Main state machine
    while True:
        match ctx.pipeline_state:
            case PipelineState.PRE:
                _handle_signal_errors(ctx, await ctx.pipeline.pre.emit(ctx))

                next_state = ctx.next_pipeline_state or PipelineState.LLM_CALL
                assert next_state in [PipelineState.LLM_CALL, PipelineState.POST, PipelineState.ERROR, PipelineState.COMPLETE]
            case PipelineState.LLM_CALL:
                if ctx.is_streaming:
                    async for item in _execute_llm_call_streaming(ctx):
                        if isinstance(item, ChatCompletionChunk):
                            yield item
                else:
                    await _execute_llm_call(ctx)

                if ctx.has_error:
                    next_state = ctx.next_pipeline_state or PipelineState.ERROR
                else:
                    next_state = ctx.next_pipeline_state or PipelineState.POST
                assert next_state in [PipelineState.POST, PipelineState.ERROR]
            case PipelineState.POST:
                _handle_signal_errors(ctx, await ctx.pipeline.post.emit(ctx))

                next_state = ctx.next_pipeline_state or PipelineState.COMPLETE
                assert next_state in [PipelineState.ERROR, PipelineState.COMPLETE]
            case PipelineState.ERROR:
                _handle_signal_errors(ctx, await ctx.pipeline.error.emit(ctx))

                next_state = ctx.next_pipeline_state or PipelineState.COMPLETE
                assert next_state in [PipelineState.PRE, PipelineState.LLM_CALL, PipelineState.POST, PipelineState.COMPLETE]

            case PipelineState.COMPLETE:
                _handle_signal_errors(ctx, await ctx.pipeline.response_complete.emit(ctx))
                yield ctx
                return

        ctx.pipeline_state = next_state
        ctx.next_pipeline_state = None

def _handle_signal_errors(ctx: RequestContext, result: SignalExecutionResult[None]):
    # TODO
    pass

async def _execute_llm_call_streaming(ctx: RequestContext) -> AsyncGenerator[ChatCompletionChunk, None]:
    await ctx.pipeline.llm_call.execute_pre(ctx)
    try:
        # Call any_llm.acompletion with streaming enabled
        completion_stream = await any_llm.acompletion(
            model=ctx.request.model_id,
            messages=ctx.request.messages, # type: ignore
            stream=True,
            **{k: v for k, v in ctx.request.model_dump().items()
            if k not in ['model', 'messages', 'stream']}
        )
        completion_stream = cast(AsyncIterator[ChatCompletionChunk], completion_stream)

        # TODO stream handlers

        # Yield chunks from the any_llm stream
        async for chunk in completion_stream:
            yield chunk
    finally:
        await ctx.pipeline.llm_call.execute_post(ctx)

async def _execute_llm_call(ctx: RequestContext):
    await ctx.pipeline.llm_call.execute_pre(ctx)
    try:
        completion = await any_llm.acompletion(
            model=ctx.request.model_id,
            messages=ctx.request.messages, # type: ignore
            **{k: v for k, v in ctx.request.model_dump().items()
            if k not in ['model', 'messages']} # type: ignore
        )
        completion = cast(ChatCompletion, completion)
        ctx.set_response(completion)
    finally:
        await ctx.pipeline.llm_call.execute_post(ctx)

# class OldPipeline:
#     """
#     Core pipeline execution engine.
#
#     Orchestrates the execution of middleware in the correct order and handles
#     both streaming and non-streaming responses.
#     """
#
#     def __init__(self, middleware: List[Middleware], monitoring_interval: int = 5):
#         """
#         Initialize pipeline with middleware chain.
#
#         Args:
#             middleware: List of middleware in execution order
#             monitoring_interval: How often to call monitoring middleware (in chunks)
#         """
#         self.middleware = middleware
#         self.monitoring_interval = monitoring_interval
#
#         # Pre-compute middleware categories for optimization
#         self.monitoring_middleware = [m for m in middleware if m.monitors_response()]
#         self.chunk_processing_middleware = [m for m in middleware if m.processes_chunks()]
#         self.complete_response_middleware = [m for m in middleware if m.needs_complete_response()]
#
#     async def execute(self, ctx: RequestContext) -> RequestContext:
#         """
#         Execute pipeline for non-streaming requests with dynamic extension support.
#
#         Args:
#             ctx: Request context to process
#
#         Returns:
#             Processed request context with response
#         """
#         # Phase 1: Before LLM processing - initial middleware
#         ctx = await self._run_before_phase(ctx, self.middleware)
#
#         if ctx.action != PipelineAction.CONTINUE:
#             return ctx
#
#         # Phase 2: Before LLM processing - dynamically added middleware
#         if ctx._extended_middleware:
#             ctx = await self._run_before_phase(ctx, ctx._extended_middleware)
#
#             if ctx.action != PipelineAction.CONTINUE:
#                 return ctx
#
#         # Phase 3: LLM execution
#         ctx = await self._execute_llm(ctx)
#
#         # Phase 4: Response completion validation (all middleware)
#         all_middleware = self.middleware + ctx._extended_middleware
#         complete_response_middleware = [m for m in all_middleware if m.needs_complete_response()]
#
#         if ctx.response is not None and complete_response_middleware:
#             assert isinstance(ctx.response, ChatCompletion)
#             for middleware in complete_response_middleware:
#                 ctx = await middleware.on_response_complete(ctx)
#                 if ctx.action != PipelineAction.CONTINUE:
#                     break
#
#         # Phase 5: After LLM processing - extended middleware (reverse order)
#         if ctx._extended_middleware:
#             ctx = await self._run_after_phase(ctx, list(reversed(ctx._extended_middleware)))
#
#         # Phase 6: After LLM processing - initial middleware (reverse order)
#         ctx = await self._run_after_phase(ctx, list(reversed(self.middleware)))
#
#         return ctx
#
#     async def execute_streaming(self, ctx: RequestContext) -> AsyncGenerator[ChatCompletionChunk | RequestContext, None]:
#         """
#         Execute pipeline for streaming requests with dynamic extension support.
#
#         Args:
#             ctx: Request context to process
#
#         Yields:
#             Response chunks as they're processed through the pipeline
#         """
#         # Phase 1: Before LLM processing - initial middleware
#         ctx = await self._run_before_phase(ctx, self.middleware)
#
#         if ctx.action != PipelineAction.CONTINUE:
#             yield ctx
#             return
#
#         # Phase 2: Before LLM processing - dynamically added middleware
#         if ctx._extended_middleware:
#             ctx = await self._run_before_phase(ctx, ctx._extended_middleware)
#
#             if ctx.action != PipelineAction.CONTINUE:
#                 yield ctx
#                 return
#
#         # Phase 3: Streaming LLM execution with all middleware
#         try:
#             # Combine all middleware for streaming processing
#             all_middleware = self.middleware + ctx._extended_middleware
#             monitoring_middleware = [m for m in all_middleware if m.monitors_response()]
#             chunk_processing_middleware = [m for m in all_middleware if m.processes_chunks()]
#             complete_response_middleware = [m for m in all_middleware if m.needs_complete_response()]
#
#             # Notify streaming middleware that stream is starting
#             for middleware in all_middleware:
#                 if hasattr(middleware, 'on_stream_start'):
#                     ctx = await middleware.on_stream_start(ctx)
#
#             last_chunk: Optional[ChatCompletionChunk] = None
#
#             # Execute LLM and process stream
#             async for chunk in self._execute_llm_streaming(ctx):
#                 ctx.chunk_count += 1
#
#                 # Extract content for monitoring
#                 ctx.stream_collector.accumulate_all(chunk.choices)
#
#                 # Check monitoring middleware periodically
#                 if (monitoring_middleware and
#                     ctx.chunk_count % self.monitoring_interval == 0):
#
#                     for middleware in monitoring_middleware:
#                         ctx = await middleware.on_response_update(ctx)
#                         if ctx.action != PipelineAction.CONTINUE:
#                             yield ctx
#                             return
#
#                 # Process chunk through chunk-processing middleware
#                 current_chunk = chunk
#                 for middleware in chunk_processing_middleware:
#                     current_chunk = await middleware.process_chunk(ctx, current_chunk)
#
#                 last_chunk = current_chunk
#                 yield current_chunk
#
#             if last_chunk is not None:
#                 # Process accumulated chunks into ChatCompletion
#                 # for handling by regular middleware.
#                 ctx.response = ChatCompletion(
#                     id=last_chunk.id,
#                     created=last_chunk.created,
#                     model=last_chunk.model,
#                     object="chat.completion",
#                     choices=ctx.stream_collector.to_choices(),
#                 )
#
#             # Final validation for complete response
#             if complete_response_middleware:
#                 for middleware in complete_response_middleware:
#                     ctx = await middleware.on_response_complete(ctx)
#                     # TODO: Handle rejection of complete streaming response
#
#             # Notify streaming middleware that stream is complete
#             for middleware in all_middleware:
#                 if hasattr(middleware, 'on_stream_end'):
#                     ctx = await middleware.on_stream_end(ctx)
#
#         except Exception as e:
#             # Handle streaming errors
#             ctx = await self._handle_streaming_error(ctx, e)
#             yield ctx
#             return
#
#     async def _run_before_phase(self, ctx: RequestContext, middleware_list: List[Middleware]) -> RequestContext:
#         """Run middleware before_llm methods for the given middleware list."""
#         for middleware in middleware_list:
#             try:
#                 ctx = await middleware.before_llm(ctx)
#                 if ctx.action != PipelineAction.CONTINUE:
#                     break
#             except Exception as e:
#                 ctx.errors.append(e)
#                 ctx = await self._handle_middleware_error(ctx, middleware, e)
#                 if ctx.action != PipelineAction.CONTINUE:
#                     break
#         return ctx
#
#     async def _run_after_phase(self, ctx: RequestContext, middleware_list: List[Middleware]) -> RequestContext:
#         """Run middleware after_llm methods for the given middleware list."""
#         # middleware_list should already be in reverse order when called
#         for middleware in middleware_list:
#             try:
#                 ctx = await middleware.after_llm(ctx)
#             except Exception as e:
#                 ctx.errors.append(e)
#                 # Continue processing other middleware even if one fails
#                 # TODO: Make error handling strategy configurable
#         return ctx
#
#     async def _execute_llm(self, ctx: RequestContext) -> RequestContext:
#         """
#         Execute LLM completion using any_llm.
#
#         Supports:
#         - Multiple providers (OpenAI, Anthropic, etc.) through any_llm
#         - Provider selection based on ctx.state['routing'] (future enhancement)
#         - Error handling and retries (handled by any_llm)
#         - Cost tracking (available in response usage field)
#         """
#         # Prepare request parameters, excluding stream to ensure non-streaming
#         request = ctx.request.model_copy()
#         request.stream = False
#
#         completion = await any_llm.acompletion(
#             model=request.model_id,
#             messages=request.messages, # type: ignore
#             **{k: v for k, v in request.model_dump().items()
#                if k not in ['model', 'messages']} # type: ignore
#         )
#         completion = cast(ChatCompletion, completion)
#
#         ctx.set_response(completion)
#         return ctx
#
#     async def _execute_llm_streaming(self, ctx: RequestContext) -> AsyncGenerator[ChatCompletionChunk, None]:
#         """
#         Execute streaming LLM completion using any_llm.
#         """
#         # Create a copy of request with stream=True for any_llm
#         streaming_request = ctx.request.model_copy()
#         streaming_request.stream = True
#
#         # Call any_llm.acompletion with streaming enabled
#         completion_stream = await any_llm.acompletion(
#             model=streaming_request.model_id,
#             messages=streaming_request.messages, # type: ignore
#             stream=True,
#             **{k: v for k, v in streaming_request.model_dump().items()
#                if k not in ['model', 'messages', 'stream']}
#         )
#         completion_stream = cast(AsyncIterator[ChatCompletionChunk], completion_stream)
#
#         # Yield chunks from the any_llm stream
#         async for chunk in completion_stream:
#             yield chunk
#
#     async def _handle_middleware_error(self, ctx: RequestContext, middleware: Middleware, error: Exception) -> RequestContext:
#         """Handle errors that occur in middleware execution."""
#         # Give middleware a chance to handle its own error
#         try:
#             ctx = await middleware.on_error(ctx, error)
#         except Exception:
#             pass  # If error handling itself fails, continue
#
#         # If middleware didn't handle the error, apply default handling
#         if ctx.action == PipelineAction.CONTINUE:
#             middleware_error = MiddlewareError(
#                 message=f"Middleware error: {str(error)}",
#                 request_id=ctx.request_id,
#                 middleware_name=middleware.__class__.__name__
#             )
#             ctx.action = PipelineAction.HALT
#             ctx.set_error(middleware_error)
#             ctx.halt_reason = middleware_error.message
#
#         return ctx
#
#     async def _handle_streaming_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
#         """Handle errors that occur during streaming execution."""
#         # Let middleware handle the streaming error
#         for middleware in reversed(self.middleware):
#             if hasattr(middleware, 'on_stream_error'):
#                 ctx = await middleware.on_stream_error(ctx, error)
#                 if ctx.action != PipelineAction.CONTINUE:
#                     break
#
#         # Default error handling
#         if ctx.action == PipelineAction.CONTINUE:
#             stream_error = StreamError(
#                 message=f"Stream error: {str(error)}",
#                 request_id=ctx.request_id
#             )
#             ctx.action = PipelineAction.HALT
#             ctx.set_error(stream_error)
#             ctx.halt_reason = stream_error.message
#
#         return ctx


def create_request_context(openai_request: NCompletionParams, **extra_metadata: Any) -> RequestContext:
    """
    Create a RequestContext from an OpenAI API request.

    Extracts client metadata from the OpenAI metadata field and any additional
    metadata from request headers, etc.
    """
    return RequestContext(
        original_request=openai_request,
        request=openai_request.model_copy(),
        client_metadata={
            # Extract from OpenAI metadata field
            **(openai_request.metadata),
            # Add any additional metadata
            **extra_metadata
        }
    )
