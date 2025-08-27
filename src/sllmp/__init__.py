"""Simple LLM Proxy - OpenAI API compatible proxy server."""

__version__ = "0.1.0"

import logging
logger = logging.getLogger(__name__)

# Core server and components
from .server import SimpleProxyServer
from .pipeline import create_request_context, execute_pipeline
from .context import RequestContext, NCompletionParams, Pipeline

# Re-export commonly used middleware
from .middleware import (
    logging_middleware,
    observability_middleware, 
    retry_middleware,
    limit_enforcement_middleware,
    create_validation_middleware,
)

# Re-export error types
from .error import (
    AuthenticationError,
    RateLimitError, 
    InternalError,
    ValidationError,
    PipelineError,
)

__all__ = [
    # Core components
    'SimpleProxyServer',
    'create_request_context',
    'execute_pipeline',
    'RequestContext',
    'NCompletionParams', 
    'Pipeline',
    
    # Middleware
    'logging_middleware',
    'observability_middleware',
    'retry_middleware', 
    'limit_enforcement_middleware',
    'create_validation_middleware',
    
    # Errors
    'AuthenticationError',
    'RateLimitError',
    'InternalError', 
    'ValidationError',
    'PipelineError',
]
