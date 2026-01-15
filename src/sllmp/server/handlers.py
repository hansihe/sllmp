"""OpenAI-compatible endpoint handlers."""

import json
import time

from starlette.responses import JSONResponse, StreamingResponse
from starlette.requests import Request
from opentelemetry import trace

from ..pipeline import create_request_context, execute_pipeline
from ..context import NCompletionParams, RequestContext
from ..error import (
    AuthenticationError, ProviderBadRequestError, RateLimitError, InternalError,
    ValidationError, ServiceUnavailableError, NetworkError
)
from ..middleware import create_validation_middleware

tracer = trace.get_tracer(__name__)


async def chat_completions_handler(request: Request, add_middleware):
    """
    OpenAI-compatible chat completions endpoint with pipeline processing.

    This endpoint routes all requests through the middleware pipeline,
    enabling composable request/response processing, monitoring, and control.

    Args:
        request: The incoming HTTP request
        pipeline: The configured middleware pipeline
    """
    span = trace.get_current_span()
    try:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON in request body: {str(e)}",
                        "type": "invalid_request_error"
                    }
                }
            )

        # Extract client metadata from request
        client_metadata = {
            # OpenAI metadata field
            **body.get('metadata', {}),
            # Request headers and client info
            'ip_address': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent'),
            'content_length': request.headers.get('content-length'),
        }

        # Validate the request before creating the context
        # Check for model field early to provide better error message
        if 'model' not in body:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": "Missing required field: model",
                        "type": "validation_error"
                    }
                }
            )

        # Create request context for pipeline processing
        # Convert dict to NCompletionParams with correct field mapping
        body_with_metadata = body.copy()
        # Map OpenAI 'model' to any_llm 'model_id'
        if 'model' in body_with_metadata:
            body_with_metadata['model_id'] = body_with_metadata.pop('model')
        # Add required metadata field
        body_with_metadata['metadata'] = body.get('metadata', {})
        # Add default empty messages if not provided
        if 'messages' not in body_with_metadata:
            body_with_metadata['messages'] = []

        try:
            completion_params = NCompletionParams(**body_with_metadata)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": f"Request validation failed: {str(e)}",
                        "type": "validation_error"
                    }
                }
            )

        ctx = create_request_context(completion_params, **client_metadata)

        # Set span attributes for request tracking
        span.set_attributes({
            "sllmp.request_id": ctx.request_id,
            "sllmp.model_id": ctx.request.model_id,
            "sllmp.is_streaming": ctx.is_streaming,
            "sllmp.messages_count": len(ctx.request.messages),
        })

        # Handle both pipeline factory patterns:
        # 1. Factory that returns Pipeline: pipeline_factory() -> Pipeline
        # 2. Middleware setup function: middleware_setup(ctx) -> None
        import inspect
        if len(inspect.signature(add_middleware).parameters) == 0:
            # Pattern 1: Factory returns Pipeline
            ctx.pipeline = add_middleware()
        else:
            # Pattern 2: Function modifies context's pipeline
            add_middleware(ctx)

        # Setup validation middleware (after custom pipeline setup)
        validation_middleware = create_validation_middleware()
        validation_middleware(ctx)

        if ctx.is_streaming:
            # Handle streaming requests through pipeline
            async def stream_generator():
                async for item in execute_pipeline(ctx):
                    # Check if this is a RequestContext (final result)
                    if isinstance(item, RequestContext):
                        # Check if pipeline resulted in error
                        if item.has_error:
                            error = item.error
                            assert error is not None

                            # Set error status on span
                            span.set_attribute("error", True)
                            span.set_attribute("error.type", type(error).__name__)
                            span.set_attribute("error.message", str(error))
                            span.record_exception(error)

                            # Send error as SSE chunk and terminate
                            error_chunk = {
                                "error": error.to_dict()["error"]
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        # If no error but final context, we're done (success)
                        span.set_attribute("success", True)
                        if item.response:
                            span.set_attribute("response.finish_reason", getattr(item.response, 'choices', [{}])[0].get('finish_reason', 'unknown'))
                        yield "data: [DONE]\n\n"
                        return

                    # This is a ChatCompletionChunk or error chunk dict
                    # Convert to dict if it's a ChatCompletionChunk object
                    if hasattr(item, 'model_dump'):
                        chunk_dict = item.model_dump()
                    elif isinstance(item, dict):
                        chunk_dict = item
                    else:
                        # Convert to JSON serializable format
                        chunk_dict = json.loads(json.dumps(item, default=str))

                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Handle non-streaming requests through pipeline
            async for _result in execute_pipeline(ctx):
                raise RuntimeError("unexpected chunk for non streaming request")

            # Check if pipeline resulted in error
            if ctx.has_error:
                error = ctx.error
                assert error is not None

                # Set error status on span
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))
                span.record_exception(error)

                # Return error response with appropriate status code
                error_dict = error.to_dict()
                status_code = 400  # Default to client error

                # Map specific error types to HTTP status codes
                if isinstance(error, ValidationError):
                    status_code = getattr(error, 'status_code', 422)
                elif isinstance(error, AuthenticationError):
                    status_code = 401
                elif isinstance(error, RateLimitError):
                    status_code = 429
                elif isinstance(error, ServiceUnavailableError):
                    status_code = 503
                elif isinstance(error, NetworkError):
                    status_code = 502
                elif isinstance(error, InternalError):
                    status_code = 500
                elif isinstance(error, ProviderBadRequestError):
                    status_code = 400

                span.set_attribute("http.status_code", status_code)
                return JSONResponse(error_dict, status_code=status_code)

            if ctx.response is None:
                span.set_attribute("error", True)
                span.set_attribute("error.message", "No response generated")
                return JSONResponse({"error": {"message": "No response generated"}})
            else:
                # Return successful response
                span.set_attribute("success", True)
                span.set_attribute("http.status_code", 200)
                if hasattr(ctx.response, 'choices') and ctx.response.choices:
                    span.set_attribute("response.finish_reason", ctx.response.choices[0].finish_reason)
                return JSONResponse(ctx.response.model_dump())

    except Exception as e:
        # Handle unexpected errors
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.message", str(e))
        span.set_attribute("http.status_code", 500)
        span.record_exception(e)
        
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        return JSONResponse(error_response, status_code=500)


async def models_handler(request: Request):
    """List available models endpoint."""
    return JSONResponse({
        "object": "list",
        "data": []
    })


async def health_handler(request: Request):
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "timestamp": int(time.time())})
