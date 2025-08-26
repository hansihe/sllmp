# Configuration as Middleware - Redesigned Architecture

## Philosophy

Keep the **pipeline API as the primary interface** with configuration as just one possible layer on top. This allows:

1. **Generic middleware** - no dependencies on specific config formats
2. **Pluggable configuration** - organizations can implement their own config approaches  
3. **Dynamic pipeline extension** - middleware can add other middleware during execution
4. **Clean separation** - config concerns separate from business logic

## Simplified Configuration Structure

### 1. Clean Config Format (config.yaml)

```yaml
# Global defaults - applied to all features unless overridden
defaults:
  provider: openai
  model: gpt-3.5-turbo
  openai_api_key: "${GLOBAL_OPENAI_KEY}"
  daily_budget: 100.00
  requests_per_minute: 60
  
  # Observability defaults
  langfuse:
    base_url: https://cloud.langfuse.com
    enabled: true
  
  # Guardrail defaults
  guardrails:
    content_safety: basic
    max_tokens: null

# Feature-specific configurations
features:
  search_autocomplete:
    description: "Search query autocomplete"
    owner_contacts: ["alice@company.com", "bob@company.com"]
    
    # Override defaults as needed
    model: gpt-3.5-turbo
    daily_budget: 50.00
    requests_per_minute: 200  # Higher rate for autocomplete
    
    # Feature-specific observability
    langfuse:
      project: "search-features"
      secret_key: "${SEARCH_LANGFUSE_SECRET}"
    
    # Multi-dimensional budget constraints
    budget_constraints:
      - dimensions: ["feature"]
        limit: 50.00
        window: "1d"
        description: "Daily feature budget"
        
      - dimensions: ["user_id"] 
        limit: 5.00
        window: "1h"
        description: "Per user hourly budget"
    
    guardrails:
      content_safety: disabled  # Search queries don't need filtering
      max_tokens: 50
  
  chat_assistant:
    description: "Customer support chat"
    owner_contacts: ["carol@company.com"]
    
    # Use different provider
    provider: anthropic
    model: claude-3-sonnet
    anthropic_api_key: "${CHAT_ANTHROPIC_KEY}"
    
    # Higher budgets for chat
    daily_budget: 1000.00
    
    langfuse:
      project: "chat-features"  
      secret_key: "${CHAT_LANGFUSE_SECRET}"
    
    # Complex budget constraints
    budget_constraints:
      - dimensions: ["feature", "user_id"]
        limit: 25.00
        window: "1d"
        description: "Per user daily budget for chat"
        
      - dimensions: ["feature"]
        limit: 2000.00
        window: "1d" 
        description: "Total daily chat budget"
        
      - dimensions: ["user_id"]  # Cross-feature constraint
        limit: 100.00
        window: "7d"
        description: "User weekly budget across all features"
    
    guardrails:
      content_safety: strict
      pii_detection: true
      max_tokens: 2000
    
    # Custom prompt template
    prompt_template: |
      You are a helpful customer support assistant.
      Please be professional and concise.
      
      Customer message: {user_message}
  
  document_analysis:
    description: "Document analysis for ML team"
    owner_contacts: ["dave@company.com", "eve@company.com"]
    
    model: gpt-4
    openai_api_key: "${ML_TEAM_OPENAI_KEY}"  # Team-specific key
    
    daily_budget: 500.00
    
    langfuse:
      project: "ml-experiments"
      secret_key: "${ML_LANGFUSE_SECRET}"
    
    budget_constraints:
      - dimensions: ["feature"]
        limit: 500.00
        window: "7d"
        description: "Weekly experiment budget"
    
    guardrails:
      content_safety: permissive
      max_tokens: 4000
    
    # Feature-specific configuration
    custom:
      max_document_size: 50000
      supported_formats: ["pdf", "txt", "docx"]
```

## Generic Middleware (Config-Agnostic)

All middleware remains completely generic with no config dependencies:

```python
# Generic budget middleware - no config knowledge
class BudgetEnforcementMiddleware(Middleware):
    def __init__(self, constraints: List[BudgetConstraint], backend: BudgetBackend):
        self.constraints = constraints  # Injected, not loaded from config
        self.backend = backend
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        # Generic budget enforcement logic
        pass

# Generic rate limiting - no config knowledge  
class RateLimitMiddleware(Middleware):
    def __init__(self, rate_limit: int, backend: RateLimitBackend):
        self.rate_limit = rate_limit
        self.backend = backend
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        # Generic rate limiting logic
        pass

# Generic observability - no config knowledge
class LangfuseMiddleware(Middleware):
    def __init__(self, project: str, secret_key: str, base_url: str):
        self.project = project
        self.secret_key = secret_key
        self.base_url = base_url
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        # Generic Langfuse integration
        pass
```

## Configuration Middleware

Single middleware that handles all configuration concerns:

```python
class ConfigurationMiddleware(Middleware):
    """
    Configuration middleware that dynamically extends the pipeline based on feature config.
    
    This is the ONLY middleware that knows about the config format.
    Other organizations can replace this with their own config approach.
    """
    
    def __init__(self, config_file: str, budget_backend, rate_limit_backend, **kwargs):
        super().__init__(**kwargs)
        self.config = self._load_config(config_file)
        self.budget_backend = budget_backend
        self.rate_limit_backend = rate_limit_backend
        
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Detect feature and dynamically extend pipeline"""
        
        # 1. Feature detection
        feature_name = self._detect_feature(ctx)
        if not feature_name:
            self.halt_with_error(ctx, "Unable to determine feature")
            return ctx
        
        # 2. Load feature config
        try:
            feature_config = self._resolve_feature_config(feature_name)
        except Exception as e:
            self.halt_with_error(ctx, f"Configuration error: {e}")
            return ctx
        
        # 3. Store config in context
        ctx.state['feature'] = {
            'name': feature_name,
            'config': feature_config
        }
        
        # 4. Apply config to current request
        self._apply_request_config(ctx, feature_config)
        
        # 5. Dynamically extend pipeline with feature-specific middleware
        await self._extend_pipeline(ctx, feature_config)
        
        return ctx
    
    def _detect_feature(self, ctx: RequestContext) -> str:
        """Detect feature from request (same logic as before)"""
        # Check metadata
        feature = ctx.client_metadata.get('feature')
        if feature:
            return feature
            
        # Check custom header
        feature = ctx.client_metadata.get('x-ai-feature') 
        if feature:
            return feature
            
        # Fallback logic
        return 'chat_assistant'  # Default
    
    def _resolve_feature_config(self, feature_name: str) -> Dict:
        """Resolve feature config with defaults inheritance"""
        if feature_name not in self.config['features']:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        # Start with defaults
        resolved = self.config['defaults'].copy()
        
        # Override with feature config
        feature_config = self.config['features'][feature_name]
        for key, value in feature_config.items():
            if value is not None:
                resolved[key] = value
        
        return resolved
    
    def _apply_request_config(self, ctx: RequestContext, config: Dict):
        """Apply config to current request"""
        # Override model if specified
        if 'model' in config:
            ctx.request['model'] = config['model']
            
        # Apply prompt template
        if 'prompt_template' in config:
            self._apply_prompt_template(ctx, config['prompt_template'])
    
    async def _extend_pipeline(self, ctx: RequestContext, config: Dict):
        """Dynamically add middleware to pipeline based on config"""
        
        # Add rate limiting if configured
        if 'requests_per_minute' in config:
            rate_middleware = RateLimitMiddleware(
                rate_limit=config['requests_per_minute'],
                backend=self.rate_limit_backend
            )
            ctx.extend_pipeline(rate_middleware)
        
        # Add budget constraints if configured
        if 'budget_constraints' in config:
            constraints = [BudgetConstraint(**c) for c in config['budget_constraints']]
            budget_middleware = BudgetEnforcementMiddleware(
                constraints=constraints,
                backend=self.budget_backend
            )
            ctx.extend_pipeline(budget_middleware)
        
        # Add Langfuse if configured
        if 'langfuse' in config and config['langfuse'].get('enabled', True):
            langfuse_config = config['langfuse']
            langfuse_middleware = LangfuseMiddleware(
                project=langfuse_config['project'],
                secret_key=langfuse_config['secret_key'],
                base_url=langfuse_config.get('base_url', 'https://cloud.langfuse.com')
            )
            ctx.extend_pipeline(langfuse_middleware)
        
        # Add content guardrails if configured
        if 'guardrails' in config:
            guardrail_config = config['guardrails']
            if guardrail_config.get('content_safety') != 'disabled':
                guardrail_middleware = ContentGuardrailMiddleware(
                    safety_level=guardrail_config['content_safety'],
                    pii_detection=guardrail_config.get('pii_detection', False),
                    max_tokens=guardrail_config.get('max_tokens')
                )
                ctx.extend_pipeline(guardrail_middleware)
    
    def _apply_prompt_template(self, ctx: RequestContext, template: str):
        """Apply prompt template to messages"""
        # Simple template substitution
        if '{user_message}' in template:
            # Get last user message
            messages = ctx.request.get('messages', [])
            if messages:
                last_message = messages[-1]
                if last_message.get('role') == 'user':
                    user_content = last_message['content']
                    new_content = template.format(user_message=user_content)
                    last_message['content'] = new_content
```

## Pipeline Extension API

Enhanced RequestContext and Pipeline to support dynamic extension:

```python
@dataclass 
class RequestContext:
    # ... existing fields ...
    
    # Dynamic pipeline extension
    _extended_middleware: List[Middleware] = field(default_factory=list)
    
    def extend_pipeline(self, middleware: Middleware):
        """Add middleware to be executed for this request"""
        self._extended_middleware.append(middleware)


class Pipeline:
    async def execute(self, ctx: RequestContext) -> RequestContext:
        """Execute pipeline with support for dynamic extension"""
        
        # Phase 1: Run initial middleware (including config middleware)
        ctx = await self._run_before_phase(ctx, self.middleware)
        
        if ctx.action != PipelineAction.CONTINUE:
            return ctx
        
        # Phase 2: Run dynamically added middleware  
        if ctx._extended_middleware:
            ctx = await self._run_before_phase(ctx, ctx._extended_middleware)
            
            if ctx.action != PipelineAction.CONTINUE:
                return ctx
        
        # Phase 3: LLM execution
        ctx = await self._execute_llm_stub(ctx)
        
        # Phase 4: After phases (reverse order)
        if ctx._extended_middleware:
            ctx = await self._run_after_phase(ctx, reversed(ctx._extended_middleware))
        
        ctx = await self._run_after_phase(ctx, reversed(self.middleware))
        
        return ctx
    
    async def execute_streaming(self, ctx: RequestContext) -> AsyncGenerator[Dict, None]:
        """Streaming execution with dynamic extension support"""
        
        # Phase 1: Initial before middleware
        ctx = await self._run_before_phase(ctx, self.middleware)
        
        if ctx.action != PipelineAction.CONTINUE:
            if ctx.response:
                yield ctx.response
            return
        
        # Phase 2: Extended before middleware
        if ctx._extended_middleware:
            ctx = await self._run_before_phase(ctx, ctx._extended_middleware)
            
            if ctx.action != PipelineAction.CONTINUE:
                if ctx.response:
                    yield ctx.response
                return
        
        # Phase 3: Streaming execution with all middleware
        all_middleware = self.middleware + ctx._extended_middleware
        monitoring_middleware = [m for m in all_middleware if m.monitors_response()]
        chunk_middleware = [m for m in all_middleware if m.processes_chunks()]
        
        async for chunk in self._execute_streaming_with_middleware(
            ctx, monitoring_middleware, chunk_middleware
        ):
            yield chunk
```

## Usage Examples

### Minimal Pipeline Setup
```python
# Only configuration middleware in the main pipeline
pipeline = (PipelineBuilder()
    .add(ConfigurationMiddleware(
        config_file="config.yaml",
        budget_backend=RedisbudgetBackend(),
        rate_limit_backend=RedisRateLimitBackend()
    ))
    .build())
```

### Alternative Configuration Approach
Organizations can implement their own config middleware:

```python
class DatabaseConfigMiddleware(Middleware):
    """Alternative config approach using database instead of YAML"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def before_llm(self, ctx: RequestContext):
        # Load config from database
        feature_config = await self.db.get_feature_config(feature_name)
        
        # Extend pipeline based on DB config
        ctx.extend_pipeline(BudgetMiddleware(...))
        ctx.extend_pipeline(LangfuseMiddleware(...))
        
        return ctx

# Use alternative config system
pipeline = (PipelineBuilder()
    .add(DatabaseConfigMiddleware(db_connection))
    .build())
```

### Adding New Features (PR-Friendly)
```yaml
# Just add to features section
new_email_classifier:
  description: "Email classification"
  owner_contacts: ["team@company.com"]
  
  model: gpt-3.5-turbo
  daily_budget: 25.00
  
  langfuse:
    project: "email-features"
    secret_key: "${EMAIL_LANGFUSE_SECRET}"
  
  budget_constraints:
    - dimensions: ["feature"]
      limit: 25.00
      window: "1d"
      description: "Daily email classification budget"
```

## Key Benefits

✅ **Clean Separation**: Generic middleware with no config dependencies
✅ **Pluggable Configuration**: Organizations can implement their own config approaches
✅ **Dynamic Extension**: Pipeline extended at runtime based on feature needs  
✅ **PR-Friendly**: Simple YAML changes with no repetition of team info
✅ **Flexible**: Same pipeline API supports config-driven and programmatic setups
✅ **Maintainable**: Configuration concerns isolated to single middleware

The configuration becomes just one possible layer on top of the core pipeline, maintaining the pipeline API as the primary interface while enabling flexible config-driven setups.