"""
Core pipeline architecture for LLM proxy middleware system.

This module provides the foundation for a composable middleware pipeline
that can handle both streaming and non-streaming LLM requests.
"""

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, cast
import logging

import any_llm
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError as AnyLLMAuthenticationError,
    RateLimitError as AnyLLMRateLimitError,
    ContentFilterError,
    ModelNotFoundError as AnyLLMModelNotFoundError,
    ProviderError,
    InvalidRequestError,
    ContextLengthExceededError as AnyLLMContextLengthExceededError,
    MissingApiKeyError,
)
from typing import AsyncIterator

from sllmp.util.signal import SignalExecutionResult

logger = logging.getLogger(__name__)

from .error import (
    PipelineError,
    MiddlewareError,
    AuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ContentPolicyError,
    ModelNotFoundError,
    ContextLengthExceededError,
    NetworkError,
    ServiceUnavailableError,
    InternalError,
)
from .context import RequestContext, NCompletionParams, PipelineState


async def execute_pipeline(
    ctx: RequestContext,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    # Setup stage
    assert ctx.pipeline_state == PipelineState.SETUP
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
                assert next_state in [
                    PipelineState.LLM_CALL,
                    PipelineState.POST,
                    PipelineState.ERROR,
                    PipelineState.COMPLETE,
                ]
            case PipelineState.LLM_CALL:
                try:
                    if ctx.is_streaming:
                        async for item in _execute_llm_call_streaming(ctx):
                            if isinstance(item, ChatCompletionChunk):
                                yield item
                    else:
                        await _execute_llm_call(ctx)
                except PipelineError as e:
                    # Handle pipeline errors from LLM execution
                    ctx.error = e
                    ctx.response = None
                    ctx.next_pipeline_state = PipelineState.ERROR

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
                assert next_state in [
                    PipelineState.PRE,
                    PipelineState.LLM_CALL,
                    PipelineState.POST,
                    PipelineState.COMPLETE,
                ]

            case PipelineState.COMPLETE:
                _handle_signal_errors(
                    ctx, await ctx.pipeline.response_complete.emit(ctx)
                )
                return

        ctx.pipeline_state = next_state
        ctx.next_pipeline_state = None


def _handle_signal_errors(ctx: RequestContext, result: SignalExecutionResult[None]):
    """Handle errors from signal execution."""
    if not result.success:
        # Handle signal execution errors - preserve PipelineErrors, wrap others
        for error in result.exceptions:
            if isinstance(error, PipelineError):
                # Preserve existing PipelineErrors
                logger.error(
                    "Pipeline error in request %s: %s",
                    ctx.request_id,
                    str(error),
                    exc_info=error,
                )
                ctx.error = error
            else:
                # Wrap other exceptions in MiddlewareError
                logger.error(
                    "Middleware execution error in request %s: %s",
                    ctx.request_id,
                    str(error),
                    exc_info=error,
                )
                middleware_error = MiddlewareError(
                    message=f"Middleware execution error: {str(error)}",
                    request_id=ctx.request_id,
                    middleware_name="unknown",  # Could be enhanced to track specific middleware
                )
                ctx.error = middleware_error

            # Clear any response and transition to error state
            ctx.response = None
            ctx.next_pipeline_state = PipelineState.ERROR
            break  # Stop on first error

        # Return to allow pipeline to continue to ERROR state
        return

    # Sanity check: This should never happen with simplified signal logic
    if result.exceptions:
        raise RuntimeError(
            f"Inconsistent signal state - has exceptions but success=True: {result}"
        )


def _extract_provider_from_model(model_id: str) -> str:
    """Extract provider name from model_id like 'openai:gpt-3.5-turbo'."""
    if ":" in model_id:
        return model_id.split(":", 1)[0]
    return "unknown"


@dataclass
class ResolvedProvider:
    """Resolved provider information for LLM calls."""

    underlying_provider: str  # The actual any_llm provider (e.g., "openai")
    model_suffix: str  # The model part after the colon (e.g., "gpt-4")
    full_model_id: str  # The full model_id to pass to any_llm (e.g., "openai:gpt-4")
    extra_options: Dict[str, Any]  # Additional options for acompletion (api_base, etc.)
    api_key_lookup: str  # Provider name to use for API key lookup


def _resolve_provider(model_id: str, providers: Dict[str, Any]) -> ResolvedProvider:
    """
    Resolve provider information from model_id and provider configs.

    Handles both standard providers (openai:gpt-4) and custom providers (my-custom-openai:gpt-4).

    Args:
        model_id: The model identifier (e.g., "my-custom-openai:gpt-4" or "openai:gpt-4")
        providers: Dict of custom provider configurations

    Returns:
        ResolvedProvider with all necessary information for the LLM call
    """
    if ":" not in model_id:
        # No provider prefix, treat as unknown
        return ResolvedProvider(
            underlying_provider="unknown",
            model_suffix=model_id,
            full_model_id=model_id,
            extra_options={},
            api_key_lookup="unknown",
        )

    provider_name, model_suffix = model_id.split(":", 1)

    # Check if this is a custom provider
    if provider_name in providers:
        provider_config = providers[provider_name]
        underlying_provider = provider_config.get("provider", provider_name)

        # Get extra options (everything except 'provider')
        extra_options = {k: v for k, v in provider_config.items() if k != "provider"}

        return ResolvedProvider(
            underlying_provider=underlying_provider,
            model_suffix=model_suffix,
            full_model_id=f"{underlying_provider}:{model_suffix}",  # Map to actual provider
            extra_options=extra_options,
            api_key_lookup=provider_name,  # Use custom provider name for API key lookup
        )

    # Standard provider - no custom config
    return ResolvedProvider(
        underlying_provider=provider_name,
        model_suffix=model_suffix,
        full_model_id=model_id,
        extra_options={},
        api_key_lookup=provider_name,
    )


def classify_llm_error(
    exception: Exception, request_id: str, provider: str = "unknown"
) -> PipelineError:
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

    # Handle typed any-llm exceptions
    if isinstance(exception, (AnyLLMAuthenticationError, MissingApiKeyError)):
        return AuthenticationError(
            message=f"Authentication failed with {provider}: {error_msg}",
            request_id=request_id,
        )

    if isinstance(exception, AnyLLMRateLimitError):
        return ProviderRateLimitError(
            message=f"Rate limit exceeded for {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
            retry_after=None,
        )

    if isinstance(exception, ContentFilterError):
        return ContentPolicyError(
            message=f"Content blocked by {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
        )

    if isinstance(exception, AnyLLMModelNotFoundError):
        return ModelNotFoundError(
            message=f"Model not available on {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
            model_id="unknown",
        )

    if isinstance(exception, AnyLLMContextLengthExceededError):
        return ContextLengthExceededError(
            message=f"Context length exceeded for {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
        )

    if isinstance(exception, InvalidRequestError):
        return ProviderBadRequestError(
            message=f"Invalid request to {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
        )

    if isinstance(exception, ProviderError):
        return ServiceUnavailableError(
            message=f"Provider error from {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
        )

    # Fallback: string-based matching for errors not wrapped by any-llm
    # (e.g., network errors from HTTP client, raw HTTP status codes)
    error_msg_lower = error_msg.lower()

    if any(
        term in error_msg_lower
        for term in ["connection", "timeout", "network", "dns", "unreachable"]
    ):
        return NetworkError(
            message=f"Network error connecting to {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
        )

    if any(
        term in error_msg_lower for term in ["service unavailable", "502", "503", "504"]
    ):
        return ServiceUnavailableError(
            message=f"Service unavailable from {provider}: {error_msg}",
            request_id=request_id,
            provider=provider,
        )

    # Default to internal error for unclassified exceptions
    return InternalError(
        message=f"Unhandled error from {provider}: {error_msg}", request_id=request_id
    )


def _create_error_chunk(error: PipelineError, ctx: RequestContext) -> dict:
    """Create an error chunk for streaming responses."""
    import time

    return {
        "id": f"error_{ctx.request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": ctx.request.model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        "error": error.to_dict()["error"],  # Include error details
    }


def _resolve_meta_kws(ctx: RequestContext) -> tuple[Dict[str, Any], ResolvedProvider]:
    """
    Resolve metadata keywords for LLM call including provider options.

    Returns:
        Tuple of (args dict for acompletion, resolved provider info)
    """
    args: Dict[str, Any] = {}

    # Get providers config from context state (set by config middleware)
    providers = ctx.state.get("providers", {})

    # Resolve provider information
    resolved = _resolve_provider(ctx.request.model_id, providers)

    # Look up API key using the appropriate provider name
    provider_key = ctx.provider_keys.get(resolved.api_key_lookup)
    if provider_key is not None:
        args["api_key"] = provider_key

    # Add any extra provider options (api_base, etc.)
    args.update(resolved.extra_options)

    return args, resolved


async def _execute_llm_call_streaming(
    ctx: RequestContext,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    await ctx.pipeline.llm_call.execute_pre(ctx)

    # Resolve provider information (including custom provider config)
    meta_kws, resolved = _resolve_meta_kws(ctx)

    try:
        # Call any_llm.acompletion with streaming enabled
        completion_stream = await any_llm.acompletion(
            model=resolved.full_model_id,  # Use resolved model_id (maps custom provider to underlying)
            messages=ctx.request.messages,  # type: ignore
            stream=True,
            **meta_kws,
            **{
                k: v
                for k, v in ctx.request.model_dump().items()
                if k
                not in [
                    "model_id",
                    "messages",
                    "metadata",
                    "stream",
                    "prompt_id",
                    "prompt_variables",
                ]
            },
        )
        completion_stream = cast(AsyncIterator[ChatCompletionChunk], completion_stream)

        # TODO stream handlers

        # Yield chunks from the any_llm stream
        async for chunk in completion_stream:
            yield chunk

    except Exception as e:
        # Classify and set pipeline error in context for streaming
        pipeline_error = classify_llm_error(
            e, ctx.request_id, resolved.underlying_provider
        )
        logger.error(
            "LLM call error in streaming request %s: %s",
            ctx.request_id,
            str(pipeline_error),
            exc_info=e,
        )
        ctx.error = pipeline_error
        ctx.response = None
        ctx.next_pipeline_state = PipelineState.ERROR
        # For streaming, don't raise - let the pipeline continue to yield final context
    finally:
        await ctx.pipeline.llm_call.execute_post(ctx)


async def _execute_llm_call(ctx: RequestContext):
    await ctx.pipeline.llm_call.execute_pre(ctx)

    # Resolve provider information (including custom provider config)
    meta_kws, resolved = _resolve_meta_kws(ctx)

    try:
        completion = await any_llm.acompletion(
            model=resolved.full_model_id,  # Use resolved model_id (maps custom provider to underlying)
            messages=ctx.request.messages,  # type: ignore
            **meta_kws,
            **{
                k: v
                for k, v in ctx.request.model_dump().items()
                if k
                not in [
                    "model_id",
                    "messages",
                    "metadata",
                    "prompt_id",
                    "prompt_variables",
                ]
            },  # type: ignore
        )
        completion = cast(ChatCompletion, completion)
        ctx.set_response(completion)
    except Exception as e:
        # Classify and raise pipeline error - this will be caught by signal handler
        pipeline_error = classify_llm_error(
            e, ctx.request_id, resolved.underlying_provider
        )
        logger.error(
            "LLM call error in request %s: %s",
            ctx.request_id,
            str(pipeline_error),
            exc_info=e,
        )
        raise pipeline_error
    finally:
        await ctx.pipeline.llm_call.execute_post(ctx)


def create_request_context(
    openai_request: NCompletionParams, **extra_metadata: Any
) -> RequestContext:
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
            **extra_metadata,
        },
    )
