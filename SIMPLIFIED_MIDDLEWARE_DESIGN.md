# Simplified Middleware Design - Progressive Complexity

## Philosophy: Only Implement What You Need

Middleware should only need to implement the methods relevant to their use case. We provide a progressive disclosure API where simple use cases require minimal code, but complex scenarios are still possible.

## Middleware Complexity Levels

### Level 0: Request/Response Middleware (Simplest)
```python
class SimpleMiddleware(Middleware):
    """Only cares about before/after LLM - works for both streaming and non-streaming"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Transform request before LLM call"""
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        """Transform response after LLM call (gets complete response even for streaming)"""
        return ctx
```

### Level 1: Response Monitoring (Callback-Based)
```python
class GuardrailMiddleware(Middleware):
    """Monitor response content with simple callbacks"""
    
    # Option A: Incremental monitoring (called with accumulated content)
    async def on_response_update(self, ctx: RequestContext, accumulated_content: str) -> ResponseAction:
        """Called periodically with all content generated so far"""
        if self._contains_violation(accumulated_content):
            return ResponseAction.HALT_WITH_ERROR("Content violation detected")
        return ResponseAction.CONTINUE
    
    # Option B: Final validation (called once at completion)
    async def on_response_complete(self, ctx: RequestContext, final_content: str) -> ResponseAction:
        """Called once when response is fully generated"""
        if self._contains_violation(final_content):
            return ResponseAction.REJECT("Final content review failed")
        return ResponseAction.ACCEPT
```

### Level 2: Chunk Processing (Stream-Aware)
```python
class ContentTransformMiddleware(Middleware):
    """Transform individual chunks as they flow through"""
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        """Process each chunk individually"""
        # Simple case: transform and yield
        transformed_chunk = self._transform_chunk(chunk)
        yield transformed_chunk
```

### Level 3: Full Stream Control (Advanced)
```python
class AdvancedStreamMiddleware(Middleware):
    """Full control over streaming lifecycle"""
    
    async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        yield chunk
    
    async def on_stream_end(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def on_stream_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        return ctx
```

## Enhanced Base Middleware Class

```python
from enum import Enum
from typing import Union, Optional, Callable

class ResponseAction(Enum):
    CONTINUE = "continue"           # Keep processing
    HALT_WITH_ERROR = "halt_error" # Stop and return error
    REJECT = "reject"              # Mark response as rejected
    ACCEPT = "accept"              # Accept response
    MODIFY = "modify"              # Modify response content

class Middleware(ABC):
    """Base middleware with progressive complexity disclosure"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Auto-detect middleware capabilities
        self._has_response_monitoring = (
            hasattr(self, 'on_response_update') or 
            hasattr(self, 'on_response_complete')
        )
        self._monitoring_interval = config.get('monitoring_interval', 5)  # chunks
    
    # Level 0: Basic middleware (always available)
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    # Level 1: Response monitoring (optional - implement if needed)
    async def on_response_update(self, ctx: RequestContext, accumulated_content: str) -> ResponseAction:
        """Override this for incremental content monitoring"""
        return ResponseAction.CONTINUE
    
    async def on_response_complete(self, ctx: RequestContext, final_content: str) -> ResponseAction:
        """Override this for final content validation"""
        return ResponseAction.ACCEPT
    
    # Level 2: Chunk processing (optional)
    async def process_chunk(self, ctx: RequestContext, chunk: Dict) -> AsyncGenerator[Dict, None]:
        """Override this for per-chunk processing"""
        yield chunk
    
    # Level 3: Stream lifecycle (optional)
    async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def on_stream_end(self, ctx: RequestContext) -> RequestContext:
        return ctx
    
    async def on_stream_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        return ctx
    
    # Metadata about middleware behavior
    def needs_complete_response(self) -> bool:
        """Whether middleware needs to see complete response before processing"""
        return hasattr(self, 'on_response_complete') and callable(getattr(self, 'on_response_complete'))
    
    def monitors_response(self) -> bool:
        """Whether middleware monitors response content"""
        return self._has_response_monitoring
```

## Simple Middleware Examples

### 1. Content Guardrail (Incremental Monitoring)
```python
class ContentGuardrailMiddleware(Middleware):
    """Monitor content for policy violations - gets callback every N chunks"""
    
    async def on_response_update(self, ctx: RequestContext, accumulated_content: str) -> ResponseAction:
        # Check accumulated content so far
        violations = self._check_content_policy(accumulated_content)
        
        if violations:
            # Log the violation
            await self._log_violation(ctx.request_id, violations)
            
            # Halt the stream with error
            return ResponseAction.HALT_WITH_ERROR(f"Content policy violation: {violations}")
        
        return ResponseAction.CONTINUE
    
    def _check_content_policy(self, content: str) -> Optional[str]:
        """Check content against policies - simple example"""
        if "inappropriate content" in content.lower():
            return "inappropriate_content"
        return None
```

### 2. Final Response Validator
```python
class ResponseValidatorMiddleware(Middleware):
    """Validate complete response quality - only called at the end"""
    
    async def on_response_complete(self, ctx: RequestContext, final_content: str) -> ResponseAction:
        # Only runs once when response is complete
        quality_score = await self._assess_response_quality(final_content)
        
        if quality_score < self.config.get('min_quality_score', 0.7):
            # Mark as rejected - could trigger retry or fallback
            return ResponseAction.REJECT(f"Response quality too low: {quality_score}")
        
        return ResponseAction.ACCEPT
    
    async def _assess_response_quality(self, content: str) -> float:
        """Assess response quality - placeholder"""
        # Could call external service, run local model, etc.
        return 0.85
```

### 3. Simple Logging Middleware
```python
class LoggingMiddleware(Middleware):
    """Just log requests and responses - works for both streaming and non-streaming"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        await self._log_request(ctx)
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        # For streaming, this gets the complete assembled response
        # For non-streaming, this gets the direct response
        await self._log_response(ctx)
        return ctx
```

### 4. Budget Tracking (Both Modes)
```python
class BudgetMiddleware(Middleware):
    """Track costs - handles both streaming and non-streaming automatically"""
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        # Estimate cost
        estimated_cost = self._estimate_cost(ctx.request)
        
        if not await self._check_budget(ctx.user_id, estimated_cost):
            ctx.action = PipelineAction.HALT
            ctx.response = {"error": {"message": "Budget exceeded"}}
            ctx.halt_reason = "budget_exceeded"
        
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        # Record actual usage
        actual_cost = self._calculate_cost(ctx.response)
        await self._record_usage(ctx.user_id, actual_cost)
        ctx.metadata['cost'] = actual_cost
        return ctx
```

## Pipeline Engine Enhancements

```python
class EnhancedPipeline:
    async def execute_streaming(self, ctx: RequestContext) -> AsyncGenerator[Dict, None]:
        # Phase 1: Before LLM
        ctx = await self._run_before_phase(ctx)
        
        if ctx.action != PipelineAction.CONTINUE:
            if ctx.response:
                yield ctx.response
            return
        
        # Phase 2: Stream processing with automatic monitoring
        try:
            llm_stream = await self._get_llm_stream(ctx)
            
            # Initialize monitoring state
            accumulated_content = ""
            monitoring_middleware = [m for m in self.middleware if m.monitors_response()]
            
            async for chunk in llm_stream:
                # Update accumulated content for monitoring
                if monitoring_middleware:
                    chunk_content = self._extract_content(chunk)
                    if chunk_content:
                        accumulated_content += chunk_content
                
                # Process chunk through regular streaming middleware
                async for processed_chunk in self._process_chunk(ctx, chunk):
                    # Check if any monitoring middleware want to intervene
                    if monitoring_middleware and ctx.chunk_count % self._get_monitoring_interval() == 0:
                        action = await self._check_monitoring_middleware(ctx, accumulated_content)
                        if action != ResponseAction.CONTINUE:
                            # Monitoring middleware wants to halt
                            yield self._create_halt_chunk(action)
                            return
                    
                    yield processed_chunk
                    ctx.chunk_count += 1
            
            # Phase 3: Final validation for complete response
            await self._run_final_validation(ctx, accumulated_content)
            
        except Exception as e:
            ctx = await self._handle_stream_error(ctx, e)
    
    async def _check_monitoring_middleware(self, ctx: RequestContext, accumulated_content: str) -> ResponseAction:
        """Check monitoring middleware periodically"""
        for middleware in [m for m in self.middleware if hasattr(m, 'on_response_update')]:
            action = await middleware.on_response_update(ctx, accumulated_content)
            if action != ResponseAction.CONTINUE:
                return action
        return ResponseAction.CONTINUE
    
    async def _run_final_validation(self, ctx: RequestContext, final_content: str) -> None:
        """Run final validation middleware"""
        for middleware in [m for m in self.middleware if hasattr(m, 'on_response_complete')]:
            action = await middleware.on_response_complete(ctx, final_content)
            if action == ResponseAction.REJECT:
                # Could trigger retry/fallback logic here
                ctx.metadata['response_rejected'] = True
```

## Configuration Examples

### Simple Guardrail Configuration
```python
pipeline_config = {
    "middleware": [
        {
            "name": "content_guardrail",
            "class": "ContentGuardrailMiddleware",
            "config": {
                "monitoring_interval": 3,  # Check every 3 chunks
                "policies": ["inappropriate_content", "pii_detection"]
            }
        },
        {
            "name": "response_validator", 
            "class": "ResponseValidatorMiddleware",
            "config": {
                "min_quality_score": 0.8
            }
        }
    ]
}
```

## Key Benefits

1. **Progressive Complexity**: Implement only what you need
2. **Automatic Handling**: Pipeline automatically detects middleware capabilities
3. **Simple Callbacks**: Easy callback-based monitoring for common cases
4. **Flexible Timing**: Choose incremental vs. final validation
5. **Unified Interface**: Same middleware works for streaming and non-streaming
6. **Response Control**: Easy halt/reject mechanisms with clear semantics

This design lets you write a simple guardrail middleware in just a few lines while still supporting complex streaming transformations when needed.