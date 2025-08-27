import json
import time
from typing import Dict, List, Optional, Union

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.requests import Request

from .pipeline import create_request_context, PipelineError, execute_pipeline
from .context import PipelineAction, NCompletionParams, RequestContext
from .error import AuthenticationError, RateLimitError, InternalError, ValidationError
from .middleware import (
    logging_middleware,
    observability_middleware,
    limit_enforcement_middleware,
    retry_middleware,
    create_validation_middleware
)



async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint with pipeline processing.

    This endpoint now routes all requests through the middleware pipeline,
    enabling composable request/response processing, monitoring, and control.
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

        if ctx.is_streaming:
            # Handle streaming requests through pipeline
            async def stream_generator():
                async for item in execute_pipeline(ctx):
                    # Check if this is a RequestContext (final result)
                    if isinstance(item, RequestContext):
                        # Check if pipeline resulted in error
                        if item.has_error:
                            # Send error as SSE chunk and terminate
                            error_chunk = {
                                "error": item.error.to_dict()["error"]
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
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        else:
            # Handle non-streaming requests through pipeline
            async for result in execute_pipeline(ctx):
                ctx = result
                break  # Get the final result for non-streaming

            # Check if pipeline resulted in error
            if ctx.has_error:
                # Return error response with appropriate status code
                error_dict = ctx.error.to_dict()
                status_code = 400  # Default to client error
                
                # Map specific error types to HTTP status codes
                if isinstance(ctx.error, ValidationError):
                    status_code = getattr(ctx.error, 'status_code', 422)
                elif isinstance(ctx.error, AuthenticationError):
                    status_code = 401
                elif isinstance(ctx.error, RateLimitError):
                    status_code = 429
                elif isinstance(ctx.error, InternalError):
                    status_code = 500
                    
                return JSONResponse(error_dict, status_code=status_code)

            # Return successful response
            return JSONResponse(ctx.response or {"error": {"message": "No response generated"}})

    except Exception as e:
        # Handle unexpected errors
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        return JSONResponse(error_response, status_code=500)


async def list_models(request: Request):
    """List available models endpoint"""
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }
        ]
    })


async def health_check(request: Request):
    """Health check endpoint"""
    return JSONResponse({"status": "ok", "timestamp": int(time.time())})


# Configure the middleware pipeline
# This is where you can customize the pipeline behavior
def create_default_pipeline():
    """
    Create a simple default middleware pipeline for testing and basic functionality.
    
    Currently includes:
    - Basic logging middleware
    - Observability middleware
    - Retry middleware for transient errors
    
    TODO: Make this configurable via environment variables or config files.
    TODO: Add more middleware as they are implemented.
    """
    from .context import Pipeline
    
    # Create a basic pipeline with minimal middleware
    pipeline = Pipeline()
    
    # Add retry middleware first (so it can handle errors from other middleware)
    pipeline.setup.connect(retry_middleware(
        max_attempts=3,
        base_delay=1.0,
        max_delay=60.0,
        log_retries=True
    ))
    
    # Add logging middleware - connects to pipeline signals
    pipeline.setup.connect(logging_middleware(
        log_requests=True, 
        log_responses=True
    ))
    
    # Add observability middleware
    pipeline.setup.connect(observability_middleware(
        emit_metrics=True
    ))
    
    return pipeline

# Initialize the pipeline
PIPELINE = create_default_pipeline()

app = Starlette(debug=True, routes=[
    Route('/', health_check),
    Route('/health', health_check),
    Route('/v1/models', list_models, methods=['GET']),
    Route('/v1/chat/completions', chat_completions, methods=['POST']),
])

def main():
    """Entry point for the CLI command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
