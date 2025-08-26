"""
Retry middleware for handling transient LLM provider errors.

This middleware provides automatic retry logic for errors that are likely
to be transient, with exponential backoff to avoid overwhelming providers.

The middleware integrates with the pipeline's error signal to implement
retry logic that works correctly with the state machine.
"""

import asyncio
import time
from typing import Optional, Set, Type
from dataclasses import dataclass, field
from typing import Dict, Any, List

from .. import logger
from ..context import RequestContext, PipelineState
from ..error import (
    PipelineError,
    RateLimitError,
    NetworkError,
    ServiceUnavailableError,
    InternalError
)


@dataclass
class RetryState:
    """State tracking for retry attempts within a single request."""
    max_attempts: int
    base_delay: float
    max_delay: float
    backoff_multiplier: float
    retryable_errors: Set[Type[PipelineError]]
    log_retries: bool

    # Runtime state
    attempts: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


def retry_middleware(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    retryable_errors: Optional[Set[Type[PipelineError]]] = None,
    log_retries: bool = True
):
    """
    Retry middleware that automatically retries failed LLM requests.

    This middleware hooks into the pipeline's error signal to catch errors
    and implement retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including original). Default: 3
        base_delay: Base delay in seconds before first retry. Default: 1.0
        max_delay: Maximum delay between retries in seconds. Default: 60.0
        backoff_multiplier: Multiplier for exponential backoff. Default: 2.0
        retryable_errors: Set of error types to retry. Defaults to transient errors.
        log_retries: Whether to log retry attempts. Default: True
    """

    if retryable_errors is None:
        retryable_errors = {
            RateLimitError,
            NetworkError,
            ServiceUnavailableError,
            InternalError
        }

    # Create retry state in closure scope
    retry_state = RetryState(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        retryable_errors=retryable_errors,
        log_retries=log_retries
    )

    def setup(ctx: RequestContext):
        # Store reference to retry state for both metadata and state access
        ctx.metadata['retry_state'] = retry_state
        ctx.state['retry_state'] = retry_state  # For easier test access

        # Connect to the error signal to handle retry logic
        ctx.pipeline.error.connect(_handle_retry_logic)

    async def _handle_retry_logic(ctx: RequestContext):
        """
        Handle retry logic when pipeline enters error state.

        This function is called by the pipeline's error signal and can
        decide whether to retry or let the error propagate.
        """
        if not ctx.has_error:
            return  # No error to handle

        error = ctx.error
        assert error is not None

        # Check if this error type is retryable
        if not _is_retryable_error(error, retry_state.retryable_errors):
            if retry_state.log_retries:
                _log_non_retryable_error(ctx, error, retry_state.attempts + 1)
            return  # Let non-retryable errors propagate

        # Check if we've exceeded max attempts
        if retry_state.attempts >= retry_state.max_attempts:
            if retry_state.log_retries:
                _log_retry_exhausted(ctx, error, retry_state.max_attempts)
            return  # Let error propagate after exhausting retries

        # We can retry this error
        retry_state.attempts += 1

        # Record this attempt
        retry_state.history.append({
            'attempt': retry_state.attempts,
            'error_type': type(error).__name__,
            'error_message': error.message,
            'timestamp': time.time()
        })

        if retry_state.log_retries:
            _log_retry_attempt(ctx, retry_state.attempts, retry_state.max_attempts)

        # Calculate delay for this retry (use attempts - 1 for exponential calculation)
        delay = min(
            retry_state.base_delay * (retry_state.backoff_multiplier ** (retry_state.attempts - 1)),
            retry_state.max_delay
        )

        # For rate limit errors, respect the retry_after if provided
        if isinstance(ctx.error, RateLimitError) and hasattr(ctx.error, 'retry_after') and ctx.error.retry_after:
            delay = max(delay, ctx.error.retry_after)

        if retry_state.log_retries:
            _log_retry_delay(ctx, error, retry_state.attempts, delay)

        # Wait before retry
        await asyncio.sleep(delay)

        # Clear error and reset pipeline state to retry LLM call
        ctx.error = None
        ctx.next_pipeline_state = PipelineState.LLM_CALL

        if retry_state.log_retries and retry_state.attempts == retry_state.max_attempts:
            # This is our final attempt
            _log_final_retry_attempt(ctx, retry_state.attempts, retry_state.max_attempts)

    return setup


def _is_retryable_error(error: PipelineError, retryable_errors: Set[Type[PipelineError]]) -> bool:
    """Check if an error is retryable based on its type."""
    return any(isinstance(error, error_type) for error_type in retryable_errors)


def _log_retry_attempt(ctx: RequestContext, attempt: int, max_attempts: int):
    """Log retry attempt."""
    logger.info(
        f"Retrying request {ctx.request_id} (attempt {attempt}/{max_attempts}) for model {ctx.request.model_id}"
    )


def _log_final_retry_attempt(ctx: RequestContext, attempt: int, max_attempts: int):
    """Log that this is the final retry attempt."""
    logger.warning(
        f"Final retry attempt {attempt}/{max_attempts} for request {ctx.request_id}"
    )


def _log_retry_delay(ctx: RequestContext, error: PipelineError, attempt: int, delay: float):
    """Log retry delay."""
    logger.info(
        f"Retrying request {ctx.request_id} after {delay:.1f}s delay due to {type(error).__name__}: {error.message}"
    )


def _log_non_retryable_error(ctx: RequestContext, error: PipelineError, attempt: int):
    """Log non-retryable error."""
    logger.info(
        f"Not retrying {type(error).__name__} for request {ctx.request_id}: {error.message}"
    )


def _log_retry_exhausted(ctx: RequestContext, final_error: PipelineError, max_attempts: int):
    """Log retry exhaustion."""
    retry_state = ctx.metadata.get('retry_state')
    total_attempts = len(retry_state.history) if retry_state else max_attempts

    logger.warning(
        f"Retry exhausted for request {ctx.request_id} after {total_attempts} attempts. "
        f"Final error: {type(final_error).__name__}: {final_error.message}"
    )
