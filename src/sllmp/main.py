#!/usr/bin/env python3
"""
Example usage of sllmp library.

This demonstrates how to create a basic OpenAI-compatible proxy server
with logging, retry, and observability middleware.
"""

import logging

from sllmp import SimpleProxyServer
from sllmp.middleware import (
    logging_middleware,
    retry_middleware,
    observability_middleware
)


def create_example_pipeline():
    """Create an example pipeline configuration."""
    from sllmp.context import Pipeline

    pipeline = Pipeline()

    # Add example middleware
    pipeline.setup.connect(retry_middleware(
        max_attempts=3,
        base_delay=1.0,
        max_delay=60.0,
        log_retries=True
    ))
    pipeline.setup.connect(logging_middleware(
        log_requests=True,
        log_responses=True
    ))
    pipeline.setup.connect(observability_middleware(
        emit_metrics=True
    ))

    return pipeline


def main():
    """Run the example server."""
    # Configure logging for stdout output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Configure sllmp library logging - this affects all sllmp.* loggers
    sllmp_logger = logging.getLogger('sllmp')
    sllmp_logger.setLevel(logging.INFO)
    
    # Create server with custom pipeline
    server = SimpleProxyServer(pipeline_factory=create_example_pipeline)

    # Create the ASGI app with tracing enabled
    app = server.create_asgi_app(debug=True, enable_tracing=True)

    # Run the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
