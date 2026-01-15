# CLAUDE.md

## Project Overview

**SLLMP** (Simple LLM Proxy) is an OpenAI API-compatible LLM proxy server built with Python and Starlette. It provides a modular, signal-based middleware pipeline for composable request/response processing, monitoring, rate limiting, and other cross-cutting concerns.

## Tech Stack

- **Python 3.13+** (required)
- **Starlette** - ASGI web framework
- **Uvicorn** - ASGI server
- **any-llm-sdk** - Multi-provider LLM abstraction
- **Pydantic** - Data validation and configuration
- **Redis** - Rate limiting backend (optional)
- **Langfuse** - Observability and tracing
- **uv** - Package manager (recommended)

## Common Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run tests excluding container tests
uv run pytest -m "not containers"

# Run with coverage
uv run pytest --cov=sllmp --cov-report=html

# Linting
uv run ruff check src tests

# Format code
uv run ruff format src tests

# Type checking
uv run mypy src/

# Start dev server
uv run python src/sllmp/main.py
# Or: uvicorn src.sllmp.main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
src/sllmp/
├── main.py              # Example server entry point
├── context.py           # RequestContext, Pipeline, PipelineState
├── pipeline.py          # Core pipeline execution logic
├── error.py             # Error type definitions
├── pricing.py           # Token pricing models
├── server/
│   ├── app.py           # SimpleProxyServer class
│   └── handlers.py      # HTTP endpoint handlers
├── middleware/
│   ├── base.py          # Middleware base patterns
│   ├── logging.py       # Logging/observability middleware
│   ├── retry.py         # Retry logic with exponential backoff
│   ├── validation.py    # Request validation middleware
│   ├── limit/           # Rate/budget limiting (in-memory & Redis)
│   ├── experimental/    # WIP: auth, guardrails, routing
│   └── service/langfuse/# Langfuse integration
├── config/              # YAML-based configuration system
└── util/
    ├── signal.py        # Signal/Hooks pub/sub system
    └── stream.py        # Stream processing utilities
tests/                   # Test suite (pytest)
```

## Architecture: Signal-Based Middleware Pipeline

The core design uses a **signal/hooks pub/sub pattern** where middleware registers callbacks into pipeline signals:

**Pipeline Phases:**
1. `setup` - Middleware registration
2. `pre` - Validation, auth, rate limiting
3. `llm_call` - LLM provider call (with streaming signals)
4. `post` - Response processing, metrics
5. `error` - Error handling, retry logic
6. `response_complete` - Final cleanup

**RequestContext** is the central data structure flowing through the pipeline containing request/response data, pipeline state, and shared middleware state.

**Creating middleware:**
```python
def my_middleware(**config):
    def setup(ctx: RequestContext):
        async def pre_process(ctx: RequestContext):
            # Custom logic
            pass
        ctx.pipeline.pre.connect(pre_process)
    return setup
```

## Code Conventions

- **Type hints required** - Strict mypy enabled
- **Ruff** for linting/formatting (line length 88, double quotes)
- **Naming**: PascalCase for classes, snake_case for functions/variables
- **Async/await** patterns throughout
- **Pydantic models** for configuration and data validation

## Testing

- **pytest** with pytest-asyncio for async tests
- **testcontainers** for Redis integration tests
- **Markers**: `@pytest.mark.redis`, `@pytest.mark.containers`
- Test files follow `test_*.py` pattern

## API Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (streaming supported)

## Error Types

All errors extend `PipelineError`:
- `ValidationError`, `AuthenticationError`, `ProviderRateLimitError`
- `NetworkError`, `ServiceUnavailableError` (retryable)
- `ClientRateLimitError`, `ContentPolicyError`, `ModelNotFoundError`
