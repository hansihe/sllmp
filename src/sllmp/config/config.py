"""
Configuration system with data models and validation.

This module provides Pydantic models for type-safe configuration loading
and validation. It supports the feature-centric configuration approach
with defaults inheritance.
"""

import os
import yaml
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from ..middleware.limit import Constraint

class ConfigurationError(Exception):
    """Configuration validation or loading error."""
    pass

class GuardrailConfig(BaseModel):
    """Content safety and guardrail configuration."""
    content_safety: Literal["disabled", "basic", "strict", "permissive"] = "basic"
    pii_detection: bool = False
    response_validation: bool = False
    max_tokens: Optional[int] = None

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration."""
    public_key: str = Field(..., description="Langfuse public key (can use ${VAR} syntax)")
    secret_key: str = Field(..., description="Langfuse secret key (can use ${VAR} syntax)")
    base_url: str = Field(default="https://cloud.langfuse.com", description="Langfuse base URL")
    enabled: bool = Field(default=True, description="Whether Langfuse is enabled")

    @field_validator('public_key', 'secret_key')
    @classmethod
    def resolve_env_vars(cls, v):
        """Resolve environment variables in configuration values."""
        if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            var_name = v[2:-1]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable not found: {var_name}")
            return env_value
        return v

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("secret_key cannot be empty")
        return v


class DefaultsConfig(BaseModel):
    """Global default configuration applied to all features."""
    provider: str = Field(default="openai", description="Default LLM provider")
    model: str = Field(default="openai:gpt-5", description="Default model")
    openai_api_key: Optional[str] = Field(default=None, description="Global OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Global Anthropic API key")
    daily_budget: float = Field(default=100.0, gt=0, description="Default daily budget in USD")
    requests_per_minute: int = Field(default=60, gt=0, description="Default rate limit")

    # Observability defaults
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Default Langfuse config")

    # Guardrail defaults
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig, description="Default guardrail config")

    @field_validator('openai_api_key', 'anthropic_api_key')
    @classmethod
    def resolve_env_vars(cls, v):
        """Resolve environment variables in API keys."""
        if v and isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            var_name = v[2:-1]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable not found: {var_name}")
            return env_value
        return v


class FeatureConfig(BaseModel):
    """Feature-specific configuration with optional overrides."""
    description: str = Field(..., description="Human-readable feature description")
    owner_contacts: List[str] = Field(..., min_length=1, description="Contact emails for feature owners")

    # Provider and model overrides
    provider: Optional[str] = Field(default=None, description="Override provider for this feature")
    model: Optional[str] = Field(default=None, description="Override model for this feature")
    openai_api_key: Optional[str] = Field(default=None, description="Feature-specific OpenAI key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Feature-specific Anthropic key")

    # Advanced budget constraints
    budget_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Multi-dimensional budget constraints")

    # Observability overrides
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Feature-specific Langfuse config")

    # Guardrail overrides
    guardrails: Optional[GuardrailConfig] = Field(default=None, description="Feature-specific guardrail config")

    # Custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Feature-specific custom config")

    @field_validator('openai_api_key', 'anthropic_api_key')
    @classmethod
    def resolve_env_vars(cls, v):
        """Resolve environment variables in API keys."""
        if v and isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            var_name = v[2:-1]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable not found: {var_name}")
            return env_value
        return v

    @field_validator('owner_contacts')
    @classmethod
    def validate_owner_contacts(cls, v):
        """Validate that owner contacts are valid email formats."""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        for contact in v:
            if not email_pattern.match(contact):
                raise ValueError(f"Invalid email format: {contact}")
        return v

    @field_validator('budget_constraints')
    @classmethod
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

    @field_validator('features')
    @classmethod
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

            # Validate with Pydantic (env var substitution now handled by field validators)
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

        resolved = self._merge_configurations(
            self.config.defaults,
            self.config.features[feature_name],
            feature_name
        )

        # Cache and return
        self._resolved_cache[feature_name] = resolved
        return resolved

    def _merge_configurations(self, defaults: DefaultsConfig, feature: FeatureConfig, feature_name: str) -> Dict[str, Any]:
        """
        Merge default and feature configurations with proper inheritance rules.

        This method handles the complex logic of merging configurations while
        respecting None values and nested object merging.
        """
        # Start with defaults as base
        resolved = defaults.model_dump()

        # Override with feature-specific values (only if not None)
        feature_dict = feature.model_dump()
        for key, value in feature_dict.items():
            if value is not None and key not in ['description', 'owner_contacts']:
                if key == 'guardrails' and value:
                    # Merge guardrail configs (feature overrides specific fields)
                    resolved_guardrails = resolved.get('guardrails', {})
                    if resolved_guardrails:
                        resolved_guardrails.update(value)
                        resolved['guardrails'] = resolved_guardrails
                    else:
                        resolved['guardrails'] = value
                else:
                    # Direct override for other fields
                    resolved[key] = value

        # Add metadata
        resolved['feature_name'] = feature_name
        resolved['feature_description'] = feature.description
        resolved['feature_owners'] = feature.owner_contacts

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
        constraint_data = config.get('budget_constraints', [])

        # Constraints are already validated by Pydantic during config loading
        return constraint_data if isinstance(constraint_data, list) else []

    def get_langfuse_config(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get Langfuse configuration for a feature."""
        config = self.resolve_feature_config(feature_name)
        langfuse_config = config.get('langfuse')

        if not langfuse_config or not langfuse_config.get('enabled', True):
            return None

        # Return the config as-is since it's already validated by Pydantic
        return langfuse_config



    def validate_configuration(self) -> List[str]:
        """
        Validate the complete configuration and return any warnings.

        Returns:
            List of warning messages (empty if no warnings)
        """
        warnings = []

        # Check for missing environment variables in dry-run mode
        for feature_name, feature in self.config.features.items():
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

        return warnings

def validate_constraint(constraint_data: dict) -> Constraint:
    """
    Validate constraint data using Pydantic parsing.

    Args:
        constraint_data: Dictionary containing constraint configuration

    Returns:
        Validated Constraint object

    Raises:
        ValueError: If constraint data is invalid
    """
    try:
        return Constraint.model_validate(constraint_data)
    except Exception as e:
        raise ValueError(f"Invalid budget constraint: {e}")
