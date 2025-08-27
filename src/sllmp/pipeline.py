"""
Core pipeline architecture for LLM proxy middleware system.

This module provides the foundation for a composable middleware pipeline
that can handle both streaming and non-streaming LLM requests.
"""

from typing import Any, AsyncGenerator, cast

import any_llm
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from typing import AsyncIterator

from sllmp.util.signal import SignalExecutionResult

from .error import (
    PipelineError, MiddlewareError, AuthenticationError,
    RateLimitError, ContentPolicyError, ModelNotFoundError, NetworkError,
    ServiceUnavailableError, InternalError
)
from .context import Pipeline, RequestContext, NCompletionParams, PipelineState

async def execute_pipeline(ctx: RequestContext) -> AsyncGenerator[ChatCompletionChunk, None]:
    # Setup stage
    assert(ctx.pipeline_state == PipelineState.SETUP)
    _handle_signal_errors(ctx, await ctx.pipeline.setup.emit(ctx))
    if ctx.has_error:
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
                return

        ctx.pipeline_state = next_state
        ctx.next_pipeline_state = None

def _handle_signal_errors(ctx: RequestContext, result: SignalExecutionResult[None]):
    """Handle errors from signal execution."""
    if not result.success:
        # Convert signal execution errors to middleware errors
        for error in result.exceptions:
            middleware_error = MiddlewareError(
                message=f"Middleware execution error: {str(error)}",
                request_id=ctx.request_id,
                middleware_name="unknown"  # Could be enhanced to track specific middleware
            )
            ctx.set_error(middleware_error)
            ctx.next_pipeline_state = PipelineState.ERROR
            break  # Stop on first error

        raise RuntimeError(f"error not propagated! {result}")


def _extract_provider_from_model(model_id: str) -> str:
    """Extract provider name from model_id like 'openai:gpt-3.5-turbo'."""
    if ':' in model_id:
        return model_id.split(':', 1)[0]
    return "unknown"


def classify_llm_error(exception: Exception, request_id: str, provider: str = "unknown") -> PipelineError:
    """
    Classify an exception from any_llm into appropriate pipeline error.

    Args:
        exception: The original exception from any_llm
        request_id: Request ID for error tracking
        provider: LLM provider name

    Returns:
        Appropriate PipelineError subclass
    """
    error_msg = str(exception)
    error_msg_lower = error_msg.lower()

    # Authentication errors
    if any(term in error_msg_lower for term in ["unauthorized", "invalid api key", "authentication", "api key"]):
        return AuthenticationError(
            message=f"Authentication failed with {provider}: {error_msg}",
            request_id=request_id
        )

    # Rate limit errors
    if any(term in error_msg_lower for term in ["rate limit", "quota exceeded", "too many requests"]):
        # Try to extract retry_after if available
        retry_after = None
        # Common patterns: "Rate limit exceeded. Try again in 60 seconds"
        import re
        match = re.search(r'try again in (\d+) seconds?', error_msg_lower)
        if match:
            retry_after = int(match.group(1))

        return RateLimitError(
            message=f"Rate limit exceeded for {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
            retry_after=retry_after
        )

    # Content policy errors
    if any(term in error_msg_lower for term in ["content policy", "safety", "filtered", "inappropriate"]):
        return ContentPolicyError(
            message=f"Content blocked by {provider}: {error_msg}",
            request_id=request_id,
            provider=provider
        )

    # Model errors
    if any(term in error_msg_lower for term in ["model not found", "invalid model", "model.*not.*available"]):
        return ModelNotFoundError(
            message=f"Model not available on {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
            model_id="unknown"  # Could extract from context
        )

    # Network errors
    if any(term in error_msg_lower for term in ["connection", "timeout", "network", "dns", "unreachable"]):
        return NetworkError(
            message=f"Network error connecting to {provider}: {error_msg}",
            request_id=request_id,
            provider=provider
        )

    # Service errors (5xx status codes)
    if any(term in error_msg_lower for term in ["service unavailable", "internal server error", "502", "503", "504"]):
        return ServiceUnavailableError(
            message=f"Service unavailable from {provider}: {error_msg}",
            request_id=request_id,
            provider=provider
        )

    # Default to internal error for unclassified exceptions
    return InternalError(
        message=f"Unhandled error from {provider}: {error_msg}",
        request_id=request_id
    )


def _create_error_chunk(error: PipelineError, ctx: RequestContext) -> dict:
    """Create an error chunk for streaming responses."""
    import time

    return {
        "id": f"error_{ctx.request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": ctx.request.model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "error"
        }],
        "error": error.to_dict()["error"]  # Include error details
    }

async def _execute_llm_call_streaming(ctx: RequestContext) -> AsyncGenerator[ChatCompletionChunk, None]:
    await ctx.pipeline.llm_call.execute_pre(ctx)
    try:
        # Call any_llm.acompletion with streaming enabled
        completion_stream = await any_llm.acompletion(
            model=ctx.request.model_id,
            messages=ctx.request.messages, # type: ignore
            stream=True,
            **{k: v for k, v in ctx.request.model_dump().items()
            if k not in ['model_id', 'messages', 'metadata', 'stream', 'prompt_id', 'prompt_variables']}
        )
        completion_stream = cast(AsyncIterator[ChatCompletionChunk], completion_stream)

        # TODO stream handlers

        # Yield chunks from the any_llm stream
        async for chunk in completion_stream:
            yield chunk

    except Exception as e:
        # Classify and set pipeline error
        provider = _extract_provider_from_model(ctx.request.model_id)
        pipeline_error = classify_llm_error(e, ctx.request_id, provider)
        ctx.set_error(pipeline_error)
        # Transition to error state
        ctx.next_pipeline_state = PipelineState.ERROR
        # For streaming, just set the error and let the pipeline handle it
        # The final RequestContext with the error will be yielded by the main pipeline
    finally:
        await ctx.pipeline.llm_call.execute_post(ctx)

async def _execute_llm_call(ctx: RequestContext):
    await ctx.pipeline.llm_call.execute_pre(ctx)
    try:
        completion = await any_llm.acompletion(
            model=ctx.request.model_id,
            messages=ctx.request.messages, # type: ignore
            **{k: v for k, v in ctx.request.model_dump().items()
            if k not in ['model_id', 'messages', 'metadata', 'prompt_id', 'prompt_variables']} # type: ignore
        )
        completion = cast(ChatCompletion, completion)
        ctx.set_response(completion)
    except Exception as e:
        # Classify and set pipeline error
        provider = _extract_provider_from_model(ctx.request.model_id)
        pipeline_error = classify_llm_error(e, ctx.request_id, provider)
        ctx.set_error(pipeline_error)
        # Transition to error state
        ctx.next_pipeline_state = PipelineState.ERROR
    finally:
        await ctx.pipeline.llm_call.execute_post(ctx)

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
