"""
Configuration system with data models and validation.

This module provides Pydantic models for type-safe configuration loading
and validation. It supports the feature-centric configuration approach
with defaults inheritance.
"""

import os
import yaml
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from ..middleware.limit import BudgetLimit, RateLimit, Constraint

class ConfigurationError(Exception):
    """Configuration validation or loading error."""
    pass

class GuardrailConfig(BaseModel):
    """Content safety and guardrail configuration."""
    content_safety: Literal["disabled", "basic", "strict", "permissive"] = "basic"
    pii_detection: bool = False
    response_validation: bool = False
    max_tokens: Optional[int] = None

    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration."""
    project: str = Field(..., description="Langfuse project name")
    secret_key: str = Field(..., description="Langfuse secret key (can use ${VAR} syntax)")
    base_url: str = Field(default="https://cloud.langfuse.com", description="Langfuse base URL")
    enabled: bool = Field(default=True, description="Whether Langfuse is enabled")

    @validator('secret_key')
    def validate_secret_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("secret_key cannot be empty")
        return v


class DefaultsConfig(BaseModel):
    """Global default configuration applied to all features."""
    provider: str = Field(default="openai", description="Default LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="Default model")
    openai_api_key: Optional[str] = Field(default=None, description="Global OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Global Anthropic API key")
    daily_budget: float = Field(default=100.0, gt=0, description="Default daily budget in USD")
    requests_per_minute: int = Field(default=60, gt=0, description="Default rate limit")

    # Observability defaults
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Default Langfuse config")

    # Guardrail defaults
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig, description="Default guardrail config")


class FeatureConfig(BaseModel):
    """Feature-specific configuration with optional overrides."""
    description: str = Field(..., description="Human-readable feature description")
    owner_contacts: List[str] = Field(..., min_length=1, description="Contact emails for feature owners")

    # Provider and model overrides
    provider: Optional[str] = Field(default=None, description="Override provider for this feature")
    model: Optional[str] = Field(default=None, description="Override model for this feature")
    openai_api_key: Optional[str] = Field(default=None, description="Feature-specific OpenAI key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Feature-specific Anthropic key")

    # Budget overrides
    daily_budget: Optional[float] = Field(default=None, gt=0, description="Override daily budget")
    requests_per_minute: Optional[int] = Field(default=None, gt=0, description="Override rate limit")

    # Advanced budget constraints
    budget_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Multi-dimensional budget constraints")

    # Observability overrides
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Feature-specific Langfuse config")

    # Guardrail overrides
    guardrails: Optional[GuardrailConfig] = Field(default=None, description="Feature-specific guardrail config")

    # Custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Feature-specific custom config")

    @validator('owner_contacts')
    def validate_owner_contacts(cls, v):
        """Validate that owner contacts are valid email formats."""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        for contact in v:
            if not email_pattern.match(contact):
                raise ValueError(f"Invalid email format: {contact}")
        return v

    @validator('budget_constraints')
    def validate_budget_constraints(cls, v):
        """Validate budget constraint format."""
        validated_constraints = []

        for constraint_data in v:
            constraint = validate_constraint(constraint_data)
            validated_constraints.append(constraint)

        return validated_constraints


class ConfigFile(BaseModel):
    """Complete configuration file structure."""
    defaults: DefaultsConfig = Field(..., description="Global default configuration")
    features: Dict[str, FeatureConfig] = Field(..., description="Feature-specific configurations")

    @validator('features')
    def validate_features(cls, v):
        """Validate feature configurations."""
        if not v:
            raise ValueError("At least one feature must be defined")

        # Validate feature names (no special characters)
        import re
        name_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')

        for feature_name in v.keys():
            if not name_pattern.match(feature_name):
                raise ValueError(f"Invalid feature name: {feature_name}. Only letters, numbers, _ and - allowed")

        return v


class ConfigResolver:
    """
    Resolves hierarchical configuration with inheritance and environment variable substitution.

    This class loads the configuration file, validates it, and provides methods
    to resolve feature-specific configuration with proper defaults inheritance.
    """

    def __init__(self, config_file: str):
        """
        Initialize configuration resolver.

        Args:
            config_file: Path to YAML configuration file

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        self.config_file = config_file
        self.config = self._load_and_validate(config_file)
        self._resolved_cache: Dict[str, Dict[str, Any]] = {}

    def _load_and_validate(self, config_file: str) -> ConfigFile:
        """Load and validate configuration file."""
        try:
            # Load YAML
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)

            if not raw_config:
                raise ConfigurationError("Configuration file is empty")

            # Substitute environment variables
            raw_config = self._substitute_env_vars(raw_config)

            # Validate with Pydantic
            return ConfigFile(**raw_config)

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def get_feature_names(self) -> List[str]:
        """Get list of all configured feature names."""
        return list(self.config.features.keys())

    def feature_exists(self, feature_name: str) -> bool:
        """Check if a feature is configured."""
        return feature_name in self.config.features

    def resolve_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """
        Resolve complete configuration for a feature with inheritance.

        Args:
            feature_name: Name of the feature to resolve configuration for

        Returns:
            Resolved configuration dictionary with all defaults applied

        Raises:
            ConfigurationError: If feature doesn't exist
        """
        # Check cache first
        if feature_name in self._resolved_cache:
            return self._resolved_cache[feature_name]

        if not self.feature_exists(feature_name):
            raise ConfigurationError(f"Unknown feature: {feature_name}")

        # Get configurations
        defaults = self.config.defaults
        feature = self.config.features[feature_name]

        # Start with defaults as base
        resolved = defaults.dict()

        # Override with feature-specific values (only if not None)
        feature_dict = feature.dict()
        for key, value in feature_dict.items():
            if value is not None and key not in ['description', 'owner_contacts']:
                # Handle nested configs specially
                if key == 'guardrails' and isinstance(value, dict):
                    # Merge guardrail configs
                    resolved_guardrails = resolved.get('guardrails', {})
                    resolved_guardrails.update(value)
                    resolved['guardrails'] = resolved_guardrails
                elif key == 'langfuse' and isinstance(value, dict):
                    # Override langfuse config completely
                    resolved['langfuse'] = value
                else:
                    resolved[key] = value

        # Add metadata
        resolved['feature_name'] = feature_name
        resolved['feature_description'] = feature.description
        resolved['feature_owners'] = feature.owner_contacts

        # Cache and return
        self._resolved_cache[feature_name] = resolved
        return resolved

    def get_limit_constraints(self, feature_name: str) -> List[Constraint]:
        """
        Get validated budget/rate limit constraints for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            List of validated Constraint objects
        """
        config = self.resolve_feature_config(feature_name)
        constraint_data = config.get('limit_constraints', [])

        constraints = []
        for data in constraint_data:
            constraint = validate_constraint(data)
            constraints.append(constraint)

        return constraints

    def get_langfuse_config(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get Langfuse configuration for a feature."""
        config = self.resolve_feature_config(feature_name)
        langfuse_config = config.get('langfuse')

        if not langfuse_config or not langfuse_config.get('enabled', True):
            return None

        return {
            'project': langfuse_config['project'],
            'public_key': langfuse_config['public_key'],
            'secret_key': langfuse_config['secret_key'],
            'base_url': langfuse_config.get('base_url', 'https://cloud.langfuse.com'),
            'enabled': langfuse_config.get('enabled', True)
        }

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR_NAME} syntax for environment variable substitution.
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ConfigurationError(f"Environment variable not found: {var_name}")
            return env_value
        else:
            return obj

    def validate_configuration(self) -> List[str]:
        """
        Validate the complete configuration and return any warnings.

        Returns:
            List of warning messages (empty if no warnings)
        """
        warnings = []

        # Check for missing environment variables in dry-run mode
        for feature_name, feature in self.config.features.items():
            try:
                resolved = self.resolve_feature_config(feature_name)

                # Check for required API keys
                provider = resolved.get('provider', 'openai')
                api_key_field = f"{provider}_api_key"

                if not resolved.get(api_key_field):
                    warnings.append(f"Feature '{feature_name}' using provider '{provider}' but no API key configured")

                # Check for Langfuse config
                langfuse_config = resolved.get('langfuse', {})
                if langfuse_config.get('enabled', True) and not langfuse_config.get('secret_key'):
                    warnings.append(f"Feature '{feature_name}' has Langfuse enabled but no secret key configured")

            except Exception as e:
                warnings.append(f"Configuration error for feature '{feature_name}': {e}")

        return warnings

def validate_constraint(constraint_data: dict):
    try:
        budget_limit = constraint_data.get('budget_limit', None)
        if budget_limit != None:
            budget_limit = BudgetLimit(
                limit=budget_limit.get('limit', 0),
                window=budget_limit.get('window', ''),
            )

        rate_limit = constraint_data.get('rate_limit', None)
        if rate_limit != None:
            rate_limit = RateLimit(
                per_minute=rate_limit['per_minute']
            )

        # Convert dict to BudgetConstraint for validation
        return Constraint(
            name=constraint_data.get('name', ''),
            dimensions=constraint_data.get('dimensions', []),
            budget_limit=budget_limit,
            rate_limit=rate_limit,
            description=constraint_data.get('description', '')
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid budget constraint: {e}")
