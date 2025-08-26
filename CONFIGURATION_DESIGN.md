# Multi-Team Configuration Design

## Overview

Design a **feature-centric configuration system** that enables multiple teams to independently configure AI features with different requirements while maintaining simplicity and preventing misconfigurations.

## Core Design Principles

1. **Feature-Centric**: Configuration organized around AI features/use-cases
2. **Hierarchical Inheritance**: Global → Team → Feature configuration layers
3. **Type Safety**: Strong validation to prevent misconfigurations
4. **Composable Constraints**: Flexible budget limit combinations
5. **PR-Friendly**: Simple YAML configs that are easy to review
6. **Runtime Validation**: Fail fast on invalid configurations

## Configuration Structure

### 1. Hierarchical Configuration (config.yaml)

```yaml
# Global defaults - inherited by all teams/features
defaults:
  provider: openai
  model: gpt-3.5-turbo
  budget_limits:
    daily_limit: 100.00
    requests_per_minute: 60
  observability:
    langfuse:
      base_url: https://cloud.langfuse.com
      enabled: true
  guardrails:
    content_safety: basic

# Team-level configuration - overrides global defaults
teams:
  search_team:
    description: "Product search and discovery features"
    contacts: ["alice@company.com", "bob@company.com"]
    langfuse_project: "search-ai-features"
    langfuse_secret_key: "${SEARCH_TEAM_LANGFUSE_SECRET}"
    openai_api_key: "${SEARCH_TEAM_OPENAI_KEY}"  # Team-specific key
    budget_limits:
      daily_limit: 500.00  # Higher limit for this team
    
  chat_team:
    description: "Customer chat and support features" 
    contacts: ["carol@company.com"]
    langfuse_project: "chat-ai-features"
    langfuse_secret_key: "${CHAT_TEAM_LANGFUSE_SECRET}"
    # Uses global OpenAI key (not specified)
    model: gpt-4  # Default to better model for chat
    budget_limits:
      daily_limit: 1000.00
    guardrails:
      content_safety: strict  # Higher safety requirements
      
  ml_team:
    description: "ML research and experimentation"
    contacts: ["dave@company.com", "eve@company.com"] 
    langfuse_project: "ml-experiments"
    langfuse_secret_key: "${ML_TEAM_LANGFUSE_SECRET}"
    openai_api_key: "${ML_TEAM_OPENAI_KEY}"
    budget_limits:
      daily_limit: 2000.00
    guardrails:
      content_safety: permissive  # Research needs flexibility

# Feature-level configuration - most specific
features:
  # Simple feature - cheap autocomplete
  search_autocomplete:
    team: search_team
    description: "Search query autocomplete suggestions"
    model: gpt-3.5-turbo  # Cheap model is sufficient
    
    # Multiple budget constraints - ANY violation blocks request
    budget_constraints:
      - name: "per_user_hourly"
        dimensions: ["user_id"]
        limit: 5.00
        window: "1h"
        description: "Max $5 per user per hour"
        
      - name: "feature_daily"
        dimensions: ["feature"]
        limit: 50.00
        window: "1d" 
        description: "Max $50 per day for entire feature"
    
    request_limits:
      requests_per_minute: 100  # Higher rate limit for autocomplete
    
    guardrails:
      content_safety: disabled  # Search queries don't need content filtering
      max_tokens: 50  # Keep responses short
  
  # Complex feature - sophisticated chat
  chat_assistant:
    team: chat_team
    description: "Customer support chat assistant"
    model: gpt-4
    
    budget_constraints:
      # Combined dimension - budget per (feature, user) pair
      - name: "per_user_per_feature_daily"
        dimensions: ["feature", "user_id"]
        limit: 25.00
        window: "1d"
        description: "Max $25 per user per day for chat"
        
      # Feature-wide constraint  
      - name: "feature_daily"
        dimensions: ["feature"]
        limit: 2000.00
        window: "1d"
        description: "Max $2000 per day for entire chat feature"
        
      # User-wide constraint (across all features)
      - name: "user_weekly" 
        dimensions: ["user_id"]
        limit: 100.00
        window: "7d"
        description: "Max $100 per user per week across all features"
    
    guardrails:
      content_safety: strict
      pii_detection: enabled
      response_validation: enabled
    
    prompt_template: |
      You are a helpful customer support assistant.
      Please be professional and concise.
      
      Customer message: {user_message}
  
  # Experimental feature - flexible configuration
  document_analysis:
    team: ml_team
    description: "Document analysis and summarization"
    model: gpt-4
    provider: anthropic  # Use different provider
    
    budget_constraints:
      - name: "experiment_weekly"
        dimensions: ["feature"]
        limit: 500.00
        window: "7d"
        description: "Weekly budget for document analysis experiments"
    
    # Custom configuration for this feature
    custom_config:
      max_document_size: 50000
      supported_formats: ["pdf", "txt", "docx"]
      
    guardrails:
      content_safety: permissive  # Research use case
```

### 2. Configuration Schema Validation

```python
# config_schema.py - Pydantic models for type safety and validation

from pydantic import BaseModel, validator, Field
from typing import List, Dict, Optional, Union, Literal
from enum import Enum

class BudgetConstraint(BaseModel):
    """Individual budget constraint definition"""
    name: str = Field(..., description="Human-readable constraint name")
    dimensions: List[Literal["feature", "user_id", "organization", "team"]] = Field(..., description="Constraint dimensions")
    limit: float = Field(..., gt=0, description="Budget limit in USD")
    window: Literal["1h", "1d", "7d", "30d"] = Field(..., description="Time window")
    description: str = Field(..., description="Human-readable description")
    
    @validator('dimensions')
    def validate_dimensions(cls, v):
        if not v:
            raise ValueError("At least one dimension required")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate dimensions not allowed")
        return v

class GuardrailConfig(BaseModel):
    """Content safety and guardrail configuration"""
    content_safety: Literal["disabled", "basic", "strict", "permissive"] = "basic"
    pii_detection: bool = False
    response_validation: bool = False
    max_tokens: Optional[int] = None

class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration"""
    langfuse: Dict[str, Union[str, bool]] = Field(default_factory=dict)
    
class BudgetLimits(BaseModel):
    """Simple budget limits (legacy support)"""
    daily_limit: float = Field(..., gt=0)
    requests_per_minute: int = Field(default=60, gt=0)

class TeamConfig(BaseModel):
    """Team-level configuration"""
    description: str
    contacts: List[str] = Field(..., min_items=1)
    langfuse_project: str
    langfuse_secret_key: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    budget_limits: Optional[BudgetLimits] = None
    guardrails: Optional[GuardrailConfig] = None
    custom_config: Dict[str, any] = Field(default_factory=dict)

class FeatureConfig(BaseModel):
    """Feature-specific configuration"""
    team: str = Field(..., description="Team that owns this feature")
    description: str
    model: Optional[str] = None
    provider: Optional[str] = None
    
    # Advanced budget constraints
    budget_constraints: List[BudgetConstraint] = Field(default_factory=list)
    
    # Simple limits (for backward compatibility)
    request_limits: Optional[Dict[str, int]] = None
    
    guardrails: Optional[GuardrailConfig] = None
    prompt_template: Optional[str] = None
    custom_config: Dict[str, any] = Field(default_factory=dict)
    
    @validator('team')
    def team_must_exist(cls, v, values, config, **kwargs):
        # This would be validated at runtime against available teams
        return v

class GlobalDefaults(BaseModel):
    """Global default configuration"""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    budget_limits: BudgetLimits
    observability: ObservabilityConfig
    guardrails: GuardrailConfig

class ConfigFile(BaseModel):
    """Complete configuration file structure"""
    defaults: GlobalDefaults
    teams: Dict[str, TeamConfig]
    features: Dict[str, FeatureConfig]
    
    @validator('features')
    def features_reference_valid_teams(cls, v, values):
        """Ensure all features reference valid teams"""
        teams = values.get('teams', {})
        invalid_refs = [
            feature_name for feature_name, feature in v.items() 
            if feature.team not in teams
        ]
        if invalid_refs:
            raise ValueError(f"Features reference invalid teams: {invalid_refs}")
        return v
```

## 3. Configuration Loading & Resolution

```python
# config_loader.py - Load and resolve hierarchical configuration

import os
import yaml
from typing import Dict, Any
from pydantic import ValidationError

class ConfigurationError(Exception):
    """Configuration validation or loading error"""
    pass

class ConfigResolver:
    """Resolves hierarchical configuration with inheritance"""
    
    def __init__(self, config_file: str):
        self.config = self._load_and_validate(config_file)
        self._resolved_cache = {}
    
    def _load_and_validate(self, config_file: str) -> ConfigFile:
        """Load and validate configuration file"""
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
                
            # Substitute environment variables
            raw_config = self._substitute_env_vars(raw_config)
            
            # Validate with Pydantic
            return ConfigFile(**raw_config)
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def resolve_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """Resolve complete configuration for a feature with inheritance"""
        
        if feature_name in self._resolved_cache:
            return self._resolved_cache[feature_name]
            
        if feature_name not in self.config.features:
            raise ConfigurationError(f"Unknown feature: {feature_name}")
        
        feature = self.config.features[feature_name]
        team = self.config.teams[feature.team]
        defaults = self.config.defaults
        
        # Resolve configuration with inheritance: defaults <- team <- feature
        resolved = {}
        
        # Start with defaults
        resolved.update(defaults.dict())
        
        # Override with team config
        team_overrides = {k: v for k, v in team.dict().items() 
                         if v is not None and k not in ['description', 'contacts']}
        resolved.update(team_overrides)
        
        # Override with feature config
        feature_overrides = {k: v for k, v in feature.dict().items() 
                           if v is not None and k not in ['team', 'description']}
        resolved.update(feature_overrides)
        
        # Add computed fields
        resolved['feature_name'] = feature_name
        resolved['team_name'] = feature.team
        resolved['team_contacts'] = team.contacts
        
        self._resolved_cache[feature_name] = resolved
        return resolved
    
    def get_budget_constraints(self, feature_name: str) -> List[BudgetConstraint]:
        """Get all budget constraints for a feature"""
        config = self.resolve_feature_config(feature_name)
        return config.get('budget_constraints', [])
    
    def get_langfuse_config(self, feature_name: str) -> Dict[str, str]:
        """Get Langfuse configuration for a feature"""
        config = self.resolve_feature_config(feature_name)
        return {
            'project': config['langfuse_project'],
            'secret_key': config['langfuse_secret_key'],
            'base_url': config['observability']['langfuse']['base_url']
        }
    
    def _substitute_env_vars(self, obj):
        """Recursively substitute ${VAR} with environment variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)  # Return original if env var not found
        else:
            return obj
```

## 4. Feature Detection Middleware

```python
# feature_detection.py - Detect which feature is being used

class FeatureDetectionMiddleware(Middleware):
    """Detect which feature/use-case the request belongs to"""
    
    def __init__(self, config_resolver: ConfigResolver, **kwargs):
        super().__init__(**kwargs)
        self.config_resolver = config_resolver
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Detect feature and load its configuration"""
        
        # Method 1: Explicit feature in metadata
        feature_name = ctx.client_metadata.get('feature')
        
        # Method 2: Feature detection via API path or headers
        if not feature_name:
            feature_name = self._detect_feature_from_request(ctx)
        
        # Method 3: Default feature for the user/org
        if not feature_name:
            feature_name = self._get_default_feature(ctx)
        
        if not feature_name:
            self.halt_with_error(ctx, "Unable to determine feature", "configuration_error")
            return ctx
        
        try:
            # Load and resolve feature configuration
            feature_config = self.config_resolver.resolve_feature_config(feature_name)
            
            # Store in context for other middleware
            ctx.state['feature'] = {
                'name': feature_name,
                'config': feature_config,
                'team': feature_config['team_name']
            }
            
            # Update request with feature-specific defaults
            if 'model' in feature_config:
                ctx.request['model'] = feature_config['model']
                
        except ConfigurationError as e:
            self.halt_with_error(ctx, str(e), "configuration_error")
        
        return ctx
    
    def _detect_feature_from_request(self, ctx: RequestContext) -> Optional[str]:
        """Detect feature from request characteristics"""
        # Example: Use custom header
        if 'x-ai-feature' in ctx.client_metadata:
            return ctx.client_metadata['x-ai-feature']
        
        # Example: Detect from model requested
        model = ctx.request.get('model', '')
        if 'autocomplete' in model:
            return 'search_autocomplete'
        elif 'chat' in model:
            return 'chat_assistant'
            
        return None
    
    def _get_default_feature(self, ctx: RequestContext) -> Optional[str]:
        """Get default feature for user/organization"""
        # Could use user_id, organization, etc. to determine default
        return 'chat_assistant'  # Fallback default
```

## 5. Budget Enforcement Middleware

```python
# budget_enforcement.py - Multi-dimensional budget enforcement

from typing import List, Tuple
from datetime import datetime, timedelta

class BudgetEnforcementMiddleware(Middleware):
    """Enforce budget constraints with multiple dimensions"""
    
    def __init__(self, budget_backend: 'BudgetBackend', **kwargs):
        super().__init__(**kwargs)
        self.budget_backend = budget_backend
    
    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Check all budget constraints before LLM execution"""
        
        feature_info = ctx.state.get('feature')
        if not feature_info:
            # Feature detection middleware must run first
            return ctx
        
        budget_constraints = feature_info['config'].get('budget_constraints', [])
        if not budget_constraints:
            return ctx
        
        # Estimate request cost
        estimated_cost = self._estimate_request_cost(ctx)
        
        # Check each constraint
        for constraint in budget_constraints:
            constraint_key = self._build_constraint_key(constraint, ctx)
            
            if await self._would_violate_constraint(constraint_key, constraint, estimated_cost):
                self.halt_with_error(
                    ctx, 
                    f"Budget constraint violated: {constraint['description']}", 
                    "budget_limit_error"
                )
                return ctx
        
        # Pre-reserve budget (optimistic locking)
        ctx.metadata['budget_reservations'] = []
        for constraint in budget_constraints:
            constraint_key = self._build_constraint_key(constraint, ctx)
            reservation_id = await self.budget_backend.reserve_budget(
                constraint_key, estimated_cost, constraint['window']
            )
            ctx.metadata['budget_reservations'].append({
                'constraint_key': constraint_key,
                'reservation_id': reservation_id,
                'estimated_cost': estimated_cost
            })
        
        return ctx
    
    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        """Finalize budget usage with actual costs"""
        
        # Calculate actual cost
        actual_cost = self._calculate_actual_cost(ctx.response)
        
        # Finalize budget reservations with actual cost
        for reservation in ctx.metadata.get('budget_reservations', []):
            await self.budget_backend.finalize_reservation(
                reservation['reservation_id'],
                actual_cost
            )
        
        # Store cost for observability
        ctx.metadata['actual_cost'] = actual_cost
        
        return ctx
    
    def _build_constraint_key(self, constraint: Dict, ctx: RequestContext) -> str:
        """Build unique key for budget constraint tracking"""
        key_parts = []
        
        for dimension in constraint['dimensions']:
            if dimension == 'feature':
                key_parts.append(f"feature:{ctx.state['feature']['name']}")
            elif dimension == 'user_id':
                user_id = ctx.client_metadata.get('user_id', 'anonymous')
                key_parts.append(f"user:{user_id}")
            elif dimension == 'team':
                key_parts.append(f"team:{ctx.state['feature']['team']}")
            elif dimension == 'organization':
                org = ctx.client_metadata.get('organization', 'default')
                key_parts.append(f"org:{org}")
        
        return "|".join(sorted(key_parts))  # Sort for consistency
    
    async def _would_violate_constraint(self, constraint_key: str, constraint: Dict, cost: float) -> bool:
        """Check if adding cost would violate the constraint"""
        current_usage = await self.budget_backend.get_usage(
            constraint_key, 
            constraint['window']
        )
        return (current_usage + cost) > constraint['limit']
    
    def _estimate_request_cost(self, ctx: RequestContext) -> float:
        """Estimate cost based on request characteristics"""
        # TODO: Implement proper cost estimation
        # This would consider model type, estimated tokens, etc.
        model = ctx.request.get('model', 'gpt-3.5-turbo')
        
        # Simple estimation based on model
        cost_per_1k_tokens = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.03,
            'claude-3-sonnet': 0.003
        }.get(model, 0.002)
        
        # Estimate tokens (very rough)
        estimated_tokens = len(str(ctx.request.get('messages', []))) * 0.75
        return (estimated_tokens / 1000) * cost_per_1k_tokens
    
    def _calculate_actual_cost(self, response: Dict) -> float:
        """Calculate actual cost from response"""
        # TODO: Implement based on actual token usage in response
        usage = response.get('usage', {})
        total_tokens = usage.get('total_tokens', 0)
        
        # This would need to be more sophisticated in practice
        return total_tokens * 0.000002  # Rough approximation
```

## 6. Usage Examples

### Simple Feature Addition (PR-friendly)
```yaml
# Just add to features section - inherits team defaults
new_email_classifier:
  team: ml_team
  description: "Email classification for customer support"
  model: gpt-3.5-turbo
  budget_constraints:
    - name: "feature_daily"
      dimensions: ["feature"]
      limit: 25.00
      window: "1d"
      description: "Daily budget for email classification"
```

### Complex Multi-Dimensional Budget
```yaml
advanced_chat_feature:
  team: chat_team
  description: "Advanced chat with multiple budget facets"
  budget_constraints:
    # User can't spend more than $10/day on this feature
    - name: "user_feature_daily"
      dimensions: ["user_id", "feature"] 
      limit: 10.00
      window: "1d"
      description: "Per user daily limit for advanced chat"
      
    # Feature can't exceed $1000/day total
    - name: "feature_daily"
      dimensions: ["feature"]
      limit: 1000.00
      window: "1d"
      description: "Total daily limit for advanced chat"
      
    # User can't exceed $50/week across ALL features
    - name: "user_weekly_all_features"
      dimensions: ["user_id"]
      limit: 50.00
      window: "7d"
      description: "User weekly limit across all features"
```

This design provides:

✅ **Team Independence**: Each team configures their own features
✅ **Flexible API Keys**: Team-level or global key configuration  
✅ **Langfuse Integration**: Per-team project routing
✅ **Model Flexibility**: Per-feature model selection
✅ **Complex Budget Constraints**: Multi-dimensional with arbitrary combinations
✅ **PR-Friendly**: Simple YAML additions with strong validation
✅ **Hard to Misconfigure**: Pydantic validation + runtime checks
✅ **Hierarchical Inheritance**: Reduces duplication via defaults