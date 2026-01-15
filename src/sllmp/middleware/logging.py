"""
Logging and observability middleware.
"""

import json
import time

from ..context import RequestContext


def logging_middleware(
    log_requests: bool = True,
    log_responses: bool = True,
    **kwargs,
):
    """
    Simple logging middleware for request/response tracking.

    Works for both streaming and non-streaming responses by logging
    the complete assembled response.
    """

    def setup(ctx: RequestContext):
        if log_requests:
            ctx.pipeline.llm_call.add_pre(_log_request)

        if log_responses:
            ctx.pipeline.llm_call.add_post(_log_response)

        # Add stream event handlers
        ctx.pipeline.llm_call_stream_process.connect(_log_stream_chunk)

    async def _log_request(ctx: RequestContext):
        """Log request details before LLM execution."""
        # Record start time for duration calculation
        ctx.metadata["request_start_time"] = time.time()

        log_data = {
            "event": "request_start",
            "request_id": ctx.request_id,
            "model": ctx.request.model_id,
            "is_streaming": ctx.is_streaming,
            "user_id": ctx.client_metadata.get("user_id"),
            "timestamp": time.time(),
        }
        print(f"[REQUEST] {json.dumps(log_data)}")

    async def _log_response(ctx: RequestContext):
        """Log response details after LLM execution."""
        # Calculate request duration
        start_time = ctx.metadata.get("request_start_time")
        if start_time:
            ctx.metadata["request_duration"] = time.time() - start_time

        log_data = {
            "event": "request_complete",
            "request_id": ctx.request_id,
            "duration": ctx.metadata.get("request_duration"),
            "chunk_count": ctx.chunk_count if ctx.is_streaming else None,
            "timestamp": time.time(),
        }

        if ctx.response:
            # Add response metadata without content
            if hasattr(ctx.response, "choices") and ctx.response.choices:
                log_data["response_choices"] = len(ctx.response.choices)
            if hasattr(ctx.response, "usage") and ctx.response.usage:
                log_data["token_usage"] = {
                    "prompt_tokens": ctx.response.usage.prompt_tokens,
                    "completion_tokens": ctx.response.usage.completion_tokens,
                    "total_tokens": ctx.response.usage.total_tokens,
                }

        print(f"[RESPONSE] {json.dumps(log_data)}")

    async def _log_stream_chunk(ctx: RequestContext, chunk) -> None:
        """Log streaming chunk information periodically."""
        # Only log every 10th chunk to avoid spam
        if ctx.chunk_count % 10 == 0:
            chunk_data = {
                "event": "stream_chunk",
                "request_id": ctx.request_id,
                "chunk_count": ctx.chunk_count,
                "timestamp": time.time(),
            }
            print(f"[STREAM] {json.dumps(chunk_data)}")

    return setup


def observability_middleware(
    emit_metrics: bool = True,
    trace_middleware: bool = False,
    **kwargs,
):
    """
    Comprehensive observability middleware.

    Provides detailed metrics, tracing, and monitoring for pipeline execution.
    """

    def setup(ctx: RequestContext):
        ctx.pipeline.llm_call.add_pre(_init_observability)
        ctx.pipeline.llm_call.add_post(_finalize_observability)
        ctx.pipeline.llm_call_stream_update.connect(_log_stream_update)

    async def _init_observability(ctx: RequestContext):
        """Initialize observability tracking."""
        ctx.metadata["observability"] = {
            "start_time": time.time(),
            "middleware_timings": [],
            "metrics": {},
        }

        if emit_metrics:
            await _emit_request_metric(ctx)

    async def _finalize_observability(ctx: RequestContext):
        """Finalize observability data and emit metrics."""
        obs_data = ctx.metadata.get("observability", {})
        obs_data["end_time"] = time.time()
        obs_data["total_duration"] = obs_data["end_time"] - obs_data.get(
            "start_time", 0
        )

        if emit_metrics:
            await _emit_completion_metrics(ctx)

        if trace_middleware:
            await _emit_trace_data(ctx)

    async def _log_stream_update(ctx: RequestContext) -> None:
        """Record streaming completion metrics."""
        obs_data = ctx.metadata.setdefault("observability", {})
        obs_data["stream_update_time"] = time.time()
        obs_data["total_chunks"] = ctx.chunk_count

    async def _emit_request_metric(ctx: RequestContext) -> None:
        """Emit request start metric."""
        # TODO: Integrate with actual metrics system (Prometheus, DataDog, etc.)
        metric_data = {
            "metric": "llm_request_started",
            "request_id": ctx.request_id,
            "model": ctx.request.model_id,
            "is_streaming": ctx.is_streaming,
            "user_id": ctx.client_metadata.get("user_id"),
            "timestamp": time.time(),
        }
        print(f"[METRIC] {json.dumps(metric_data)}")

    async def _emit_completion_metrics(ctx: RequestContext) -> None:
        """Emit request completion metrics."""
        # TODO: Integrate with actual metrics system
        obs_data = ctx.metadata.get("observability", {})

        metric_data = {
            "metric": "llm_request_completed",
            "request_id": ctx.request_id,
            "duration": obs_data.get("total_duration"),
            "chunk_count": ctx.chunk_count if ctx.is_streaming else None,
            "timestamp": time.time(),
        }

        # Add token usage if available
        if ctx.response and hasattr(ctx.response, "usage") and ctx.response.usage:
            metric_data.update(
                {
                    "prompt_tokens": ctx.response.usage.prompt_tokens,
                    "completion_tokens": ctx.response.usage.completion_tokens,
                    "total_tokens": ctx.response.usage.total_tokens,
                }
            )

        print(f"[METRIC] {json.dumps(metric_data)}")

    async def _emit_trace_data(ctx: RequestContext) -> None:
        """Emit detailed trace information."""
        # TODO: Integrate with tracing system (Jaeger, Zipkin, etc.)
        obs_data = ctx.metadata.get("observability", {})

        trace_data = {
            "event": "pipeline_trace",
            "request_id": ctx.request_id,
            "total_duration": obs_data.get("total_duration"),
            "middleware_timings": obs_data.get("middleware_timings", []),
            "errors": [str(e) for e in ctx.errors],
            "metadata_keys": list(ctx.metadata.keys()),
            "state_keys": list(ctx.state.keys()),
        }

        print(f"[TRACE] {json.dumps(trace_data, indent=2)}")

    return setup
