"""Core server abstraction for simple_llm_proxy."""

from typing import Callable, Optional, List
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware

from ..context import Pipeline
from ..middleware import (
    logging_middleware,
    observability_middleware,
    retry_middleware,
)
from .handlers import (
    chat_completions_handler,
    models_handler,
    health_handler,
)


class SimpleProxyServer:
    """
    Core server class that provides OpenAI-compatible endpoints.
    
    This class encapsulates the server logic and provides a clean interface
    for library users to customize the pipeline and add middleware.
    """
    
    def __init__(self, pipeline_factory: Optional[Callable[[], Pipeline]] = None):
        """
        Initialize the server with an optional pipeline factory.
        
        Args:
            pipeline_factory: Function that returns a configured Pipeline.
                             If None, uses the default pipeline.
        """
        self.pipeline_factory = pipeline_factory or self._create_default_pipeline_factory()
        self.pipeline = self.pipeline_factory()
        
    def _create_default_pipeline_factory(self) -> Callable[[], Pipeline]:
        """Create a factory for the default pipeline configuration."""
        def factory():
            return self._create_default_pipeline()
        return factory
    
    def _create_default_pipeline(self) -> Pipeline:
        """
        Create a simple default middleware pipeline for basic functionality.
        
        Currently includes:
        - Basic logging middleware
        - Observability middleware  
        - Retry middleware for transient errors
        """
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
    
    def add_middleware(self, middleware_func: Callable):
        """
        Add middleware to the pipeline.
        
        Args:
            middleware_func: Middleware function to add to the pipeline
        """
        self.pipeline.setup.connect(middleware_func)
    
    def create_asgi_app(self, debug: bool = True, middleware: Optional[List[Middleware]] = None) -> Starlette:
        """
        Create and configure the ASGI application.
        
        Args:
            debug: Enable debug mode
            middleware: Optional list of Starlette middleware to add
            
        Returns:
            Configured Starlette application
        """
        routes = [
            Route('/', health_handler),
            Route('/health', health_handler),
            Route('/v1/models', models_handler, methods=['GET']),
            Route('/v1/chat/completions', self._create_chat_completions_handler(), methods=['POST']),
        ]
        
        return Starlette(
            debug=debug,
            routes=routes,
            middleware=middleware or []
        )
    
    def _create_chat_completions_handler(self):
        """Create the chat completions handler with this server's pipeline."""
        async def handler(request):
            return await chat_completions_handler(request, self.pipeline)
        return handler
    
    def add_route(self, app: Starlette, path: str, handler: Callable, methods: Optional[List[str]] = None):
        """
        Add a custom route to the application.
        
        Args:
            app: The Starlette application
            path: URL path for the route
            handler: Request handler function
            methods: HTTP methods to accept (defaults to ['GET'])
        """
        if methods is None:
            methods = ['GET']
        app.router.routes.append(Route(path, handler, methods=methods))