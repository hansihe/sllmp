# LLM Proxy Pipeline Architecture Design

## Overview

This document proposes a flexible, extensible pipeline architecture for the LLM proxy, inspired by Elixir's Plug system but adapted for LLM operations. The design enables composable middleware that can transform requests, responses, and control execution flow.

## Core Architecture

### 1. Request Context (The "Connection")

```python
@dataclass
class RequestContext:
    # Request lifecycle
    original_request: Dict[str, Any]      # Immutable original OpenAI request
    request: Dict[str, Any]               # Mutable current request state
    response: Optional[Dict[str, Any]]    # Response when available
    
    # Pipeline control
    action: PipelineAction                # CONTINUE, HALT, RETRY, FALLBACK
    halt_reason: Optional[str]            # Why pipeline was halted
    
    # Metadata and state
    metadata: Dict[str, Any]              # Request metadata (timing, costs, etc.)
    state: Dict[str, Any]                 # Shared state between middleware
    errors: List[Exception]               # Accumulated errors
    
    # Request characteristics
    user_id: Optional[str]                # For rate limiting, budgeting
    request_id: str                       # Unique request identifier
    is_streaming: bool                    # Streaming vs non-streaming
    
    # Provider selection
    selected_provider: Optional[str]      # Which LLM provider to use
    fallback_providers: List[str]         # Fallback chain
```

### 2. Middleware Interface

```python
class Middleware(ABC):
    """Base middleware class - similar to Elixir Plug behaviour"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def call(self, ctx: RequestContext) -> RequestContext:
        """Main middleware entry point"""
        pass
    
    # Optional lifecycle hooks
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        return ctx
    
    async def on_retry(self, ctx: RequestContext, attempt: int) -> RequestContext:
        return ctx
```

### 3. Pipeline Execution Engine

```python
class Pipeline:
    def __init__(self, middleware: List[Middleware], providers: Dict[str, LLMProvider]):
        self.middleware = middleware
        self.providers = providers
    
    async def execute(self, ctx: RequestContext) -> RequestContext:
        """Execute the full pipeline"""
        
        # Phase 1: Before LLM (request transformation)
        ctx = await self._run_before_phase(ctx)
        
        # Phase 2: LLM execution (with retries/fallbacks)
        if ctx.action == PipelineAction.CONTINUE:
            ctx = await self._run_llm_phase(ctx)
        
        # Phase 3: After LLM (response transformation)
        ctx = await self._run_after_phase(ctx)
        
        return ctx
```

## Pipeline Control Flow

### Actions
- **CONTINUE**: Proceed to next middleware
- **HALT**: Stop pipeline, return current response
- **RETRY**: Retry current operation (with backoff)
- **FALLBACK**: Switch to fallback provider and retry

### Execution Phases

1. **Before Phase**: Request preprocessing, routing, validation
2. **LLM Phase**: Provider selection, execution, error handling
3. **After Phase**: Response processing, logging, cleanup

## Middleware Categories & Examples

### 1. Request Transformation
```python
class PromptTemplateMiddleware(Middleware):
    """Apply prompt templates and transformations"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        template = self.config.get('templates', {}).get(ctx.request['model'])
        if template:
            ctx.request['messages'] = self._apply_template(ctx.request['messages'], template)
        return ctx

class InputSanitizationMiddleware(Middleware):
    """Sanitize and validate input"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        if self._contains_pii(ctx.request['messages']):
            ctx.request['messages'] = self._redact_pii(ctx.request['messages'])
            ctx.metadata['pii_detected'] = True
        return ctx
```

### 2. Routing & Load Balancing
```python
class ModelRoutingMiddleware(Middleware):
    """Route requests to appropriate models/providers"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        # Route based on request characteristics
        if self._needs_vision_model(ctx.request):
            ctx.selected_provider = 'openai'
            ctx.request['model'] = 'gpt-4-vision-preview'
        elif self._is_complex_reasoning(ctx.request):
            ctx.selected_provider = 'anthropic'
            ctx.request['model'] = 'claude-3-opus'
        else:
            ctx.selected_provider = 'openai'
            ctx.request['model'] = 'gpt-3.5-turbo'
        
        return ctx

class LoadBalancerMiddleware(Middleware):
    """Distribute load across providers"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        provider = self._select_least_loaded_provider()
        ctx.selected_provider = provider
        return ctx
```

### 3. Rate Limiting & Budget Control
```python
class RateLimitMiddleware(Middleware):
    """Per-user rate limiting"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        if not await self._check_rate_limit(ctx.user_id):
            ctx.action = PipelineAction.HALT
            ctx.response = self._rate_limit_response()
            ctx.halt_reason = "rate_limit_exceeded"
        return ctx

class BudgetTrackingMiddleware(Middleware):
    """Track and enforce spending limits"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        estimated_cost = self._estimate_cost(ctx.request)
        if not await self._check_budget(ctx.user_id, estimated_cost):
            ctx.action = PipelineAction.HALT
            ctx.response = self._budget_exceeded_response()
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        actual_cost = self._calculate_actual_cost(ctx.response)
        await self._record_usage(ctx.user_id, actual_cost)
        ctx.metadata['cost'] = actual_cost
        return ctx
```

### 4. Observability & Logging
```python
class ObservabilityMiddleware(Middleware):
    """Comprehensive logging and metrics"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        ctx.metadata['start_time'] = time.time()
        await self._log_request_start(ctx)
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        duration = time.time() - ctx.metadata['start_time']
        await self._log_request_complete(ctx, duration)
        await self._emit_metrics(ctx, duration)
        return ctx

class AuditMiddleware(Middleware):
    """Audit trail for compliance"""
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        await self._record_audit_event({
            'request_id': ctx.request_id,
            'user_id': ctx.user_id,
            'model': ctx.request.get('model'),
            'provider': ctx.selected_provider,
            'cost': ctx.metadata.get('cost'),
            'duration': ctx.metadata.get('duration'),
            'timestamp': time.time()
        })
        return ctx
```

### 5. Resilience & Reliability
```python
class RetryMiddleware(Middleware):
    """Exponential backoff retries"""
    
    async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        if self._is_retryable_error(error) and ctx.metadata.get('retry_count', 0) < self.config['max_retries']:
            ctx.action = PipelineAction.RETRY
            ctx.metadata['retry_count'] = ctx.metadata.get('retry_count', 0) + 1
            await self._wait_with_backoff(ctx.metadata['retry_count'])
        return ctx

class FallbackMiddleware(Middleware):
    """Provider fallback chain"""
    
    async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        if self._is_provider_error(error) and ctx.fallback_providers:
            next_provider = ctx.fallback_providers.pop(0)
            ctx.selected_provider = next_provider
            ctx.action = PipelineAction.FALLBACK
            ctx.metadata['fallback_reason'] = str(error)
        return ctx

class CircuitBreakerMiddleware(Middleware):
    """Circuit breaker pattern for provider health"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        if self._is_circuit_open(ctx.selected_provider):
            if ctx.fallback_providers:
                ctx.selected_provider = ctx.fallback_providers.pop(0)
                ctx.metadata['circuit_breaker_triggered'] = True
            else:
                ctx.action = PipelineAction.HALT
                ctx.response = self._service_unavailable_response()
        return ctx
```

## Configuration & Composition

### Pipeline Configuration
```python
# Example pipeline configuration
pipeline_config = {
    "middleware": [
        {
            "name": "input_sanitization",
            "class": "InputSanitizationMiddleware",
            "config": {"redact_pii": True}
        },
        {
            "name": "rate_limiting", 
            "class": "RateLimitMiddleware",
            "config": {"requests_per_minute": 100}
        },
        {
            "name": "budget_tracking",
            "class": "BudgetTrackingMiddleware", 
            "config": {"daily_limit": 50.00}
        },
        {
            "name": "model_routing",
            "class": "ModelRoutingMiddleware",
            "config": {"routing_strategy": "intelligent"}
        },
        {
            "name": "prompt_templates",
            "class": "PromptTemplateMiddleware",
            "config": {"template_dir": "./templates"}
        },
        {
            "name": "observability",
            "class": "ObservabilityMiddleware",
            "config": {"log_level": "INFO"}
        },
        {
            "name": "retries",
            "class": "RetryMiddleware", 
            "config": {"max_retries": 3, "backoff_factor": 2}
        },
        {
            "name": "fallbacks",
            "class": "FallbackMiddleware",
            "config": {"fallback_chain": ["anthropic", "openai"]}
        }
    ],
    "providers": {
        "openai": {"api_key": "...", "base_url": "..."},
        "anthropic": {"api_key": "...", "base_url": "..."}
    }
}
```

### Dynamic Pipeline Composition
```python
class PipelineBuilder:
    """Fluent interface for building pipelines"""
    
    def __init__(self):
        self.middleware = []
        self.providers = {}
    
    def add_middleware(self, middleware: Middleware) -> 'PipelineBuilder':
        self.middleware.append(middleware)
        return self
    
    def add_provider(self, name: str, provider: LLMProvider) -> 'PipelineBuilder':
        self.providers[name] = provider
        return self
    
    def build(self) -> Pipeline:
        return Pipeline(self.middleware, self.providers)

# Usage
pipeline = (PipelineBuilder()
    .add_middleware(RateLimitMiddleware({"rpm": 100}))
    .add_middleware(ModelRoutingMiddleware({"strategy": "cost_optimized"}))
    .add_middleware(RetryMiddleware({"max_retries": 3}))
    .add_provider("openai", OpenAIProvider(api_key="..."))
    .add_provider("anthropic", AnthropicProvider(api_key="..."))
    .build())
```

## Benefits of This Architecture

1. **Composability**: Mix and match middleware as needed
2. **Testability**: Each middleware can be tested independently
3. **Extensibility**: Easy to add new capabilities
4. **Configuration**: Middleware behavior controlled by configuration
5. **Performance**: Async throughout, minimal overhead
6. **Observability**: Built-in request tracing and metrics
7. **Reliability**: Comprehensive error handling and recovery

## Integration with Current Codebase

The pipeline would integrate at the main endpoint:

```python
async def chat_completions(request: Request):
    try:
        body = await request.json()
        
        # Create request context
        ctx = RequestContext(
            original_request=body,
            request=body.copy(),
            request_id=generate_request_id(),
            user_id=extract_user_id(request),
            is_streaming=body.get('stream', False)
        )
        
        # Execute pipeline
        ctx = await app.pipeline.execute(ctx)
        
        # Return response
        if ctx.is_streaming:
            return create_streaming_response(ctx.response)
        else:
            return JSONResponse(ctx.response)
            
    except Exception as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "internal_error"}},
            status_code=500
        )
```

This architecture provides the foundation for all the features you mentioned while remaining flexible and maintainable.