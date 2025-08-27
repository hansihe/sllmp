"""OpenAI-compatible endpoint handlers."""

import json
import time

from starlette.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from ..pipeline import create_request_context, execute_pipeline
from ..context import NCompletionParams, RequestContext
from ..error import AuthenticationError, RateLimitError, InternalError, ValidationError
from ..middleware import create_validation_middleware


async def chat_completions_handler(request: Request, add_middleware):
    """
    OpenAI-compatible chat completions endpoint with pipeline processing.

    This endpoint routes all requests through the middleware pipeline,
    enabling composable request/response processing, monitoring, and control.

    Args:
        request: The incoming HTTP request
        pipeline: The configured middleware pipeline
    """
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

        # Setup validation middleware
        validation_middleware = create_validation_middleware()
        validation_middleware(ctx)

        add_middleware(ctx)

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

                            # Send error as SSE chunk and terminate
                            error_chunk = {
                                "error": error.to_dict()["error"]
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        # If no error but final context, we're done
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
                elif isinstance(error, InternalError):
                    status_code = 500

                return JSONResponse(error_dict, status_code=status_code)

            if ctx.response is None:
                return JSONResponse({"error": {"message": "No response generated"}})
            else:
                # Return successful response
                return JSONResponse(ctx.response.model_dump())

    except Exception as e:
        # Handle unexpected errors
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
