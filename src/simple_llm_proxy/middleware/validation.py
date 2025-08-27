"""
Request validation middleware for ensuring valid requests before pipeline execution.
"""

from typing import Any, Dict
import json
from pydantic import ValidationError as PydanticValidationError

from ..context import RequestContext, PipelineAction, PipelineState
from ..error import ValidationError


async def _validate_request(ctx: RequestContext) -> None:
    """
    Validate the request structure and content.
    
    This runs early in the pipeline to catch validation errors
    before expensive operations like LLM calls.
    """
    try:
        # The request should already be parsed into ctx.request
        # But let's validate it's properly structured
        
        # Check if model is specified
        if not ctx.request.model_id:
            ctx.set_error(ValidationError(
                "Missing required field: model",
                request_id=ctx.request_id,
                status_code=422
            ))
            ctx.next_pipeline_state = PipelineState.ERROR
            return
        
        # Validate messages structure
        if not hasattr(ctx.request, 'messages') or ctx.request.messages is None:
            # Set default empty messages if not provided
            ctx.request.messages = []
        
        # Validate each message structure
        for i, message in enumerate(ctx.request.messages):
            if not isinstance(message, dict):
                ctx.set_error(ValidationError(
                    f"Message at index {i} must be an object",
                    request_id=ctx.request_id,
                    status_code=422
                ))
                ctx.next_pipeline_state = PipelineState.ERROR
                return
            
            if 'role' not in message:
                ctx.set_error(ValidationError(
                    f"Message at index {i} missing required field: role",
                    request_id=ctx.request_id,
                    status_code=422
                ))
                ctx.next_pipeline_state = PipelineState.ERROR
                return
            
            if 'content' not in message:
                ctx.set_error(ValidationError(
                    f"Message at index {i} missing required field: content",
                    request_id=ctx.request_id,
                    status_code=422
                ))
                ctx.next_pipeline_state = PipelineState.ERROR
                return
        
        # Validate numeric parameters
        numeric_params = {
            'temperature': (0.0, 2.0),
            'top_p': (0.0, 1.0), 
            'presence_penalty': (-2.0, 2.0),
            'frequency_penalty': (-2.0, 2.0),
            'n': (1, 10),
            'max_tokens': (1, 100000)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            value = getattr(ctx.request, param, None)
            if value is not None:
                try:
                    float_val = float(value)
                    if not (min_val <= float_val <= max_val):
                        ctx.set_error(ValidationError(
                            f"Parameter '{param}' must be between {min_val} and {max_val}",
                            request_id=ctx.request_id,
                            status_code=422
                        ))
                        ctx.next_pipeline_state = PipelineState.ERROR
                        return
                except (ValueError, TypeError):
                    ctx.set_error(ValidationError(
                        f"Parameter '{param}' must be a number",
                        request_id=ctx.request_id,
                        status_code=422
                    ))
                    ctx.next_pipeline_state = PipelineState.ERROR
                    return
        
        # If we get here, validation passed
        # Continue with normal pipeline flow
        
    except Exception as e:
        # Catch any unexpected validation errors
        ctx.set_error(ValidationError(
            f"Request validation failed: {str(e)}",
            request_id=ctx.request_id,
            status_code=422
        ))
        ctx.next_pipeline_state = ctx.pipeline_state.ERROR


def create_validation_middleware(**kwargs):
    """
    Create request validation middleware.
    
    This middleware validates requests before they reach the LLM pipeline,
    ensuring proper structure and parameter values.
    
    Returns appropriate HTTP status codes:
    - 422: Validation Error (missing required fields, invalid values)
    """
    
    def setup(ctx: RequestContext):
        # Register validation to run early in the pipeline
        ctx.pipeline.pre.connect(_validate_request)
    
    return setup