"""Core server abstraction for sllmp."""

from typing import Callable, Optional, List, Any
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware

from ..pipeline import RequestContext
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

    def __init__(self, pipeline_factory: Callable[[RequestContext], None]):
        """
        Initialize the server with an optional pipeline factory.

        Args:
            pipeline_factory: Function that returns a configured Pipeline.
                             If None, uses the default pipeline.
        """
        self.pipeline_factory = pipeline_factory

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
            return await chat_completions_handler(request, self.pipeline_factory)
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
