# SLLMP - **S**imple **LLM** **P**roxy

A modular, middleware-based OpenAI API compatible LLM proxy server built with Python and Starlette.

## Overview

SLLMP is a library for building LLM proxy servers with a focus on modularity and extensibility. It provides a middleware pipeline system that allows you to compose request/response processing, monitoring, rate limiting, authentication, and other cross-cutting concerns.

## Key Features

- **OpenAI API Compatible**: Full compatibility with `/v1/chat/completions` and `/v1/models` endpoints
- **Middleware Pipeline**: Composable middleware system for request/response processing
- **Built-in Middleware**: Retry logic, rate limiting, logging, observability, validation
- **Redis Support**: Redis-based rate limiting and caching
- **Observability**: OpenTelemetry tracing and Langfuse integration
- **Streaming Support**: Server-sent events with proper `[DONE]` termination
- **Production Ready**: Comprehensive test coverage, type hints, and error handling

## Installation

```bash
# Install with uv (recommended)
uv add sllmp

# Install with pip
pip install sllmp

# For development
uv sync --dev
```

## Quick Start

```python
from sllmp import SimpleProxyServer
from sllmp.middleware import logging_middleware, retry_middleware

# Create a simple server
server = SimpleProxyServer()
app = server.create_asgi_app()

# With middleware - using pipeline factory approach
def pipeline_factory(ctx):
    ctx.add_middleware(retry_middleware(max_attempts=3))
    ctx.add_middleware(logging_middleware(log_requests=True))

server = SimpleProxyServer(pipeline_factory=pipeline_factory)
app = server.create_asgi_app()
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=sllmp

# Exclude Redis/Docker tests
uv run pytest -m "not redis and not containers"

# Code quality
uv run ruff check
uv run ruff format
uv run mypy src/

# Run example server
uv run python src/sllmp/main.py
```

## API Endpoints

### Health Check
- `GET /health` - Health check

### OpenAI Compatible
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (streaming/non-streaming)

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Middleware System

SLLMP uses a pipeline-based middleware system where each middleware can process requests and responses. Available middleware includes:

### Core Middleware

```python
from sllmp.context import RequestContext
from sllmp.middleware import (
    logging_middleware,
    retry_middleware,
    observability_middleware,
    limit_enforcement_middleware,
    create_validation_middleware
)

def setup_middleware(ctx: RequestContext):
    """Setup function that adds middleware to the request context."""

    # Add middleware using ctx.add_middleware
    ctx.add_middleware(create_validation_middleware())

    # Rate limiting (requires Redis backend)
    ctx.add_middleware(limit_enforcement_middleware(
        limit_backend=redis_backend,
        default_limit=100,
        window_seconds=60
    ))

    # Retry logic
    ctx.add_middleware(retry_middleware(
        max_attempts=3,
        base_delay=1.0,
        max_delay=60.0
    ))

    # Logging
    ctx.add_middleware(logging_middleware(
        log_requests=True,
        log_responses=True
    ))

    # Observability (OpenTelemetry + Langfuse)
    ctx.add_middleware(observability_middleware(
        emit_metrics=True
    ))

# Use the setup function as middleware factory
server = SimpleProxyServer(pipeline_factory=lambda: setup_middleware)
```

### Custom Middleware

You can create custom middleware by implementing the middleware interface:

```python
from sllmp.context import RequestContext

def custom_middleware(**config):
    """Custom middleware factory function."""

    def setup(ctx: RequestContext):
        """Setup function that configures pipeline hooks."""

        # Hook into pre-processing
        async def pre_process(ctx: RequestContext):
            print(f"Processing request to {ctx.request.model_id}")

        # Hook into post-processing
        async def post_process(ctx: RequestContext):
            print(f"Request completed for {ctx.request_id}")

        # Attach hooks to pipeline signals
        ctx.pipeline.llm_call.add_pre(pre_process)
        ctx.pipeline.llm_call.add_post(post_process)

    return setup

# Add to request context
def my_setup_function(ctx: RequestContext):
    ctx.add_middleware(custom_middleware(param1="value"))
```

## Configuration

SLLMP uses a feature-based configuration system implemented as middleware. The configuration middleware (`configuration_middleware`) loads feature configurations and dynamically adds middleware based on the feature's requirements.

### Using Configuration Middleware

```python
from sllmp.config.middleware import configuration_middleware
from sllmp.middleware.limit import InMemoryLimitBackend

# Create server with configuration-driven middleware
def setup_with_config(ctx):
    ctx.add_middleware(configuration_middleware(
        config_file="config.yaml",
        limit_backend=InMemoryLimitBackend()  # or RedisLimitBackend()
    ))

server = SimpleProxyServer(pipeline_factory=lambda: setup_with_config)
```

### Configuration File Format

The configuration system uses a feature-centric YAML format with defaults inheritance:

```yaml
defaults:
  # Global defaults applied to all features
  provider_api_keys:
    openai: "${OPENAI_API_KEY}"
    anthropic: "${ANTHROPIC_API_KEY}"

  # Default budget constraints
  budget_constraints:
    daily_budget:
      max_cost: 100.0
      window_seconds: 86400
    rate_limit:
      max_requests: 1000
      window_seconds: 3600

  # Default Langfuse config
  langfuse:
    public_key: "${LANGFUSE_PUBLIC_KEY}"
    secret_key: "${LANGFUSE_SECRET_KEY}"
    base_url: "https://cloud.langfuse.com"
    enabled: true

features:
  chat_assistant:
    description: "Main chat assistant feature"
    owner: "ai-team"
    model: "openai:gpt-4"
    # Inherits defaults, can override specific values

  code_assistant:
    description: "Code generation and review"
    owner: "dev-tools-team"
    model: "anthropic:claude-3.5-sonnet"
    budget_constraints:
      # Override default rate limit for this feature
      rate_limit:
        max_requests: 500
        window_seconds: 3600
```

### Environment Variables

The configuration system supports environment variable substitution using `${VAR}` syntax:

- `${OPENAI_API_KEY}`: OpenAI API key
- `${ANTHROPIC_API_KEY}`: Anthropic API key
- `${LANGFUSE_PUBLIC_KEY}`: Langfuse public key
- `${LANGFUSE_SECRET_KEY}`: Langfuse secret key

## Testing

The project includes comprehensive tests covering:

- Core pipeline functionality
- Middleware behavior
- Error handling
- Redis integration
- OpenAI API compatibility
- Integration scenarios

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_middleware.py
uv run pytest tests/test_integration_scenarios.py

# Run with Redis (requires Docker)
uv run pytest -m redis

# Run with testcontainers (requires Docker)
uv run pytest -m containers
```

## Dependencies

Core dependencies:
- `starlette` - ASGI web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `any-llm-sdk` - LLM provider integrations
- `redis` - Redis client (optional)
- `langfuse` - Observability (optional)
- `opentelemetry-*` - Distributed tracing

Development dependencies:
- `pytest` - Testing framework
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `testcontainers` - Integration testing with Docker

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request
