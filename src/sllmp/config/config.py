"""
Configuration system with data models and validation.

This module provides Pydantic models for type-safe configuration loading
and validation. It supports the feature-centric configuration approach
with defaults inheritance.
"""

import os
import re
import yaml
from typing import Dict, List, Any, Optional, Literal
import pydantic
from pydantic import BaseModel, Field, field_validator
from ..middleware.limit import Constraint

class ConfigurationError(Exception):
    """Configuration validation or loading error."""
    pass


class ProviderConfig(BaseModel):
    """
    Custom provider configuration.

    Allows defining a provider alias that maps to an underlying any_llm provider
    with additional options (api_base, etc.) passed to acompletion calls.
    """
    provider: str = Field(..., description="Underlying any_llm provider (e.g., openai, anthropic)")

    model_config = pydantic.ConfigDict(extra="allow")

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        if not v or v.strip() == "":
            raise ValueError("provider cannot be empty")
        return v


# class GuardrailConfig(BaseModel):
#     """Content safety and guardrail configuration."""
#     content_safety: Literal["disabled", "basic", "strict", "permissive"] = "basic"
#     pii_detection: bool = False
#     response_validation: bool = False
#     max_tokens: Optional[int] = None
#
#     @field_validator('max_tokens')
#     @classmethod
#     def validate_max_tokens(cls, v):
#         if v is not None and v <= 0:
#             raise ValueError("max_tokens must be positive")
#         return v


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration."""
    public_key: str = Field(..., description="Langfuse public key (can use ${VAR} syntax)")
    secret_key: str = Field(..., description="Langfuse secret key (can use ${VAR} syntax)")
    base_url: str = Field(default="https://cloud.langfuse.com", description="Langfuse base URL")
    enabled: bool = Field(default=True, description="Whether Langfuse is enabled")
    default_prompt_label: str = Field(default="latest", description="When using Langfuse prompt management, this label will be fetched by default. Can be specified by env variable.")

    @field_validator('public_key', 'secret_key', 'default_prompt_label')
    @classmethod
    def resolve_env_vars(cls, v):
        """Resolve environment variables in configuration values."""
        return _resolve_env_vars(v)

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("secret_key cannot be empty")
        return v


class DefaultsConfig(BaseModel):
    """Global default configuration applied to all features."""
    model: Optional[str] = Field(default=None, description="Override provider:model for this feature")
    provider_api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for LLM providers")
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict, description="Custom provider configurations")

    # Advanced budget constraints
    budget_constraints: Dict[str, Constraint] = Field(default_factory=dict, description="Multi-dimensional budget constraints by name")

    # Observability defaults
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Default Langfuse config")

    # # Guardrail defaults
    # guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig, description="Default guardrail config")

    @field_validator('provider_api_keys')
    @classmethod
    def resolve_env_vars(cls, kv):
        """Resolve environment variables in API keys."""
        return _resolve_env_vars(kv)

    @field_validator('providers', mode='before')
    @classmethod
    def validate_providers(cls, v):
        """Validate provider names and resolve env vars in provider configs."""
        if not v:
            return v
        name_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        for provider_name in v.keys():
            if not name_pattern.match(provider_name):
                raise ValueError(f"Invalid provider name: {provider_name}. Only letters, numbers, _ and - allowed")
            # Resolve env vars in provider config values
            if isinstance(v[provider_name], dict):
                v[provider_name] = _resolve_env_vars(v[provider_name])
        return v

    @field_validator('budget_constraints')
    @classmethod
    def set_constraint_names(cls, constraints_dict):
        """Set constraint names from dictionary keys."""
        for name, constraint in constraints_dict.items():
            # Set the name from the key
            constraint.name = name
        return constraints_dict


class FeatureConfig(BaseModel):
    """Feature-specific configuration with optional overrides."""
    description: str = Field("", description="Human-readable feature description")
    owner: str = Field("", description="Owner information")

    # Provider and model overrides
    model: Optional[str] = Field(default=None, description="Override provider:model for this feature")
    provider_api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for LLM providers")
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict, description="Feature-specific provider overrides")

    # Advanced budget constraints
    budget_constraints: Dict[str, Constraint] = Field(default_factory=dict, description="Multi-dimensional budget constraints by name")

    # Observability overrides
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Feature-specific Langfuse config")

    # # Guardrail overrides
    # guardrails: Optional[GuardrailConfig] = Field(default=None, description="Feature-specific guardrail config")

    # Custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Feature-specific custom config")

    @field_validator('provider_api_keys')
    @classmethod
    def resolve_env_vars(cls, kv):
        """Resolve environment variables in API keys."""
        return _resolve_env_vars(kv)

    @field_validator('providers', mode='before')
    @classmethod
    def validate_providers(cls, v):
        """Validate provider names and resolve env vars in provider configs."""
        if not v:
            return v
        name_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        for provider_name in v.keys():
            if not name_pattern.match(provider_name):
                raise ValueError(f"Invalid provider name: {provider_name}. Only letters, numbers, _ and - allowed")
            # Resolve env vars in provider config values
            if isinstance(v[provider_name], dict):
                v[provider_name] = _resolve_env_vars(v[provider_name])
        return v

    @field_validator('budget_constraints')
    @classmethod
    def set_constraint_names(cls, constraints_dict):
        """Set constraint names from dictionary keys."""
        for name, constraint in constraints_dict.items():
            # Set the name from the key
            constraint.name = name
        return constraints_dict

class ResolvedFeatureConfig(BaseModel):
    """Fully resolved feature configuration with all defaults applied."""
    # Core feature metadata
    feature_name: str = Field(..., description="Name of the feature")
    feature_description: str = Field("", description="Human-readable feature description")
    owner: str = Field("", description="Owner information")

    # Provider and model configuration (resolved)
    model: Optional[str] = Field(None, description="Resolved provider:model")
    provider_api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for LLM providers")
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict, description="Custom provider configurations")

    # Budget and rate limiting
    budget_constraints: Dict[str, Constraint] = Field(default_factory=dict, description="Multi-dimensional budget constraints by name")

    # Observability configuration
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Resolved Langfuse config")

    # # Guardrail configuration
    # guardrails: GuardrailConfig = Field(..., description="Resolved guardrail config")

    # Custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Feature-specific custom config")


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
        self._resolved_cache: Dict[str, ResolvedFeatureConfig] = {}

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

    def resolve_feature_config(self, feature_name: str) -> ResolvedFeatureConfig:
        """
        Resolve complete configuration for a feature with inheritance.

        Args:
            feature_name: Name of the feature to resolve configuration for

        Returns:
            Resolved configuration as a validated ResolvedFeatureConfig object

        Raises:
            ConfigurationError: If feature doesn't exist
        """
        # Check cache first
        if feature_name in self._resolved_cache:
            return self._resolved_cache[feature_name]

        if not self.feature_exists(feature_name):
            raise ConfigurationError(f"Unknown feature: {feature_name}")

        resolved_dict = self._merge_configurations(
            self.config.defaults,
            self.config.features[feature_name],
            feature_name
        )

        # Convert to typed ResolvedFeatureConfig
        resolved = ResolvedFeatureConfig(**resolved_dict)

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
            if value is not None and key not in ['description', 'owner']:
                if key == "provider_api_keys":
                    # Always merge provider_api_keys, don't replace with empty dict
                    if value:  # Only merge if feature has non-empty provider_api_keys
                        resolved_api_keys = resolved.get('provider_api_keys', {})
                        resolved_api_keys.update(value)
                        resolved['provider_api_keys'] = resolved_api_keys
                    # If value is empty dict, keep the defaults (don't override)
                elif key == "budget_constraints":
                    # Merge budget_constraints dicts, feature constraints override defaults by name
                    if value:  # Only merge if feature has non-empty budget_constraints
                        resolved_constraints = resolved.get('budget_constraints', {})
                        resolved_constraints.update(value)
                        resolved['budget_constraints'] = resolved_constraints
                    # If value is empty dict, keep the defaults (don't override)
                elif key == "providers":
                    # Merge providers dicts, feature providers override defaults by name
                    if value:  # Only merge if feature has non-empty providers
                        resolved_providers = resolved.get('providers', {})
                        resolved_providers.update(value)
                        resolved['providers'] = resolved_providers
                    # If value is empty dict, keep the defaults (don't override)
                # if key == 'guardrails' and value:
                #     # Merge guardrail configs (feature overrides specific fields)
                #     resolved_guardrails = resolved.get('guardrails', {})
                #     if resolved_guardrails:
                #         resolved_guardrails.update(value)
                #         resolved['guardrails'] = resolved_guardrails
                #     else:
                #         resolved['guardrails'] = value
                else:
                    # Direct override for other fields
                    resolved[key] = value

        # Add metadata
        resolved['feature_name'] = feature_name
        resolved['feature_description'] = feature.description
        resolved['owner'] = feature.owner

        return resolved

    def get_limit_constraints(self, feature_name: str) -> Dict[str, Constraint]:
        """
        Get validated budget/rate limit constraints for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dict of validated Constraint objects by name
        """
        config = self.resolve_feature_config(feature_name)
        return config.budget_constraints

    def get_langfuse_config(self, feature_name: str) -> Optional[LangfuseConfig]:
        """Get Langfuse configuration for a feature."""
        config = self.resolve_feature_config(feature_name)

        if not config.langfuse or not config.langfuse.enabled:
            return None

        return config.langfuse



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

            # # Check for required API keys
            # provider = resolved.provider
            # api_key = getattr(resolved, f"{provider}_api_key", None)

            # if not api_key:
            #     warnings.append(f"Feature '{feature_name}' using provider '{provider}' but no API key configured")

            # Check for Langfuse config
            if resolved.langfuse and resolved.langfuse.enabled and not resolved.langfuse.secret_key:
                warnings.append(f"Feature '{feature_name}' has Langfuse enabled but no secret key configured")

        return warnings

def _resolve_env_vars(v: Any) -> Any:
    if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
        var_name = v[2:-1]
        env_value = os.getenv(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable not found: {var_name}")
        return env_value
    if isinstance(v, dict):
        for key, val in v.items():
            v[key] = _resolve_env_vars(val)
        return v
    return v
