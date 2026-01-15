"""
Comprehensive tests for the configuration system.

Tests cover ConfigResolver, environment variable resolution, inheritance,
validation, and error handling scenarios.
"""

import os
import pytest
import tempfile
import yaml
import httpx
from pathlib import Path
from unittest.mock import patch, MagicMock

from sllmp.config.config import (
    ConfigResolver,
    ConfigFile,
    DefaultsConfig,
    FeatureConfig,
    LangfuseConfig,
    ResolvedFeatureConfig,
    ConfigurationError,
)
from sllmp.middleware.limit.limit import Constraint
from any_llm.exceptions import (
    RateLimitError as AnyLLMRateLimitError,
    AuthenticationError as AnyLLMAuthenticationError,
    ProviderError as AnyLLMProviderError,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def basic_config_data():
    """Basic valid configuration data."""
    return {
        "defaults": {
            "model": "openai:gpt-3.5-turbo",
            "provider_api_keys": {"openai": "sk-test-key"},
            "langfuse": {
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "enabled": True,
            },
        },
        "features": {
            "chat_completion": {
                "description": "Basic chat completion",
                "owner": "team-ai",
            },
            "code_generation": {
                "description": "Code generation feature",
                "owner": "team-dev",
                "model": "openai:gpt-4",
                "langfuse": {
                    "public_key": "pk-code",
                    "secret_key": "sk-code",
                    "enabled": True,
                },
            },
        },
    }


@pytest.fixture
def complex_config_data():
    """Complex configuration with constraints and environment variables."""
    return {
        "defaults": {
            "provider_api_keys": {
                "openai": "${OPENAI_API_KEY}",
                "anthropic": "${ANTHROPIC_API_KEY}",
            },
            "langfuse": {
                "public_key": "${LANGFUSE_PUBLIC_KEY}",
                "secret_key": "${LANGFUSE_SECRET_KEY}",
                "base_url": "https://custom.langfuse.com",
            },
        },
        "features": {
            "production_chat": {
                "description": "Production chat with limits",
                "owner": "prod-team",
                "budget_constraints": {
                    "daily-feature-cost-limit": {
                        "dimensions": ["feature"],
                        "budget_limit": {"limit": 1000.0, "window": "1d"},
                        "description": "Daily cost limit per feature",
                    },
                    "hourly-user-rate-limit": {
                        "dimensions": ["user_id"],
                        "rate_limit": {"per_minute": 60},
                        "description": "Rate limit per user",
                    },
                },
                "custom": {"priority": "high", "alerts_enabled": True},
            }
        },
    }


class TestConfigurationLoading:
    """Test configuration file loading and basic validation."""

    def test_load_valid_config(self, temp_config_file, basic_config_data):
        """Test loading a valid configuration file."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)

        assert resolver.config_file == temp_config_file
        assert isinstance(resolver.config, ConfigFile)
        assert len(resolver.config.features) == 2
        assert "chat_completion" in resolver.config.features
        assert "code_generation" in resolver.config.features

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent config file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            ConfigResolver("/nonexistent/config.yaml")

    def test_load_empty_file(self, temp_config_file):
        """Test error handling for empty config file."""
        # Create empty file
        open(temp_config_file, "w").close()

        with pytest.raises(ConfigurationError, match="Configuration file is empty"):
            ConfigResolver(temp_config_file)

    def test_load_invalid_yaml(self, temp_config_file):
        """Test error handling for invalid YAML syntax."""
        with open(temp_config_file, "w") as f:
            f.write("invalid: yaml: content: [\n")

        with pytest.raises(ConfigurationError, match="Invalid YAML syntax"):
            ConfigResolver(temp_config_file)

    def test_load_missing_required_fields(self, temp_config_file):
        """Test validation of required configuration fields."""
        incomplete_config = {
            "defaults": {},
            # Missing features section
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(incomplete_config, f)

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            ConfigResolver(temp_config_file)

    def test_invalid_feature_names(self, temp_config_file):
        """Test validation of feature names."""
        config_with_invalid_names = {
            "defaults": {},
            "features": {
                "invalid-feature!": {"description": "Invalid name", "owner": "test"}
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_with_invalid_names, f)

        with pytest.raises(ConfigurationError, match="Invalid feature name"):
            ConfigResolver(temp_config_file)


class TestEnvironmentVariableResolution:
    """Test environment variable substitution in configuration."""

    def test_env_var_resolution_success(self, temp_config_file):
        """Test successful environment variable resolution."""
        config_data = {
            "defaults": {"provider_api_keys": {"openai": "${TEST_API_KEY}"}},
            "features": {"test_feature": {"description": "Test", "owner": "test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.dict(os.environ, {"TEST_API_KEY": "resolved-key-123"}):
            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config("test_feature")

            # Check that environment variable was resolved and preserved
            assert "openai" in resolved.provider_api_keys
            assert resolved.provider_api_keys["openai"] == "resolved-key-123"

    def test_env_var_resolution_missing_var(self, temp_config_file):
        """Test error handling for missing environment variables."""
        config_data = {
            "defaults": {"provider_api_keys": {"openai": "${MISSING_API_KEY}"}},
            "features": {"test_feature": {"description": "Test", "owner": "test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        # Ensure the environment variable doesn't exist
        if "MISSING_API_KEY" in os.environ:
            del os.environ["MISSING_API_KEY"]

        with pytest.raises(
            ConfigurationError, match="Environment variable not found: MISSING_API_KEY"
        ):
            ConfigResolver(temp_config_file)

    def test_nested_env_var_resolution(self, temp_config_file):
        """Test environment variable resolution in nested structures."""
        config_data = {
            "defaults": {
                "langfuse": {
                    "public_key": "${LANGFUSE_PK}",
                    "secret_key": "${LANGFUSE_SK}",
                    "default_prompt_label": "${PROMPT_LABEL}",
                }
            },
            "features": {"test_feature": {"description": "Test", "owner": "test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        env_vars = {
            "LANGFUSE_PK": "pk-resolved",
            "LANGFUSE_SK": "sk-resolved",
            "PROMPT_LABEL": "production",
        }

        with patch.dict(os.environ, env_vars):
            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config("test_feature")

            assert resolved.langfuse.public_key == "pk-resolved"
            assert resolved.langfuse.secret_key == "sk-resolved"
            assert resolved.langfuse.default_prompt_label == "production"


class TestConfigurationInheritance:
    """Test configuration inheritance from defaults to features."""

    def test_basic_inheritance(self, temp_config_file, basic_config_data):
        """Test basic configuration inheritance."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("chat_completion")

        # Should inherit defaults
        assert resolved.model == "openai:gpt-3.5-turbo"
        # provider_api_keys should be inherited from defaults
        assert "openai" in resolved.provider_api_keys
        assert resolved.provider_api_keys["openai"] == "sk-test-key"
        assert resolved.langfuse.public_key == "pk-test"
        assert resolved.langfuse.enabled is True

        # Should have feature-specific metadata
        assert resolved.feature_name == "chat_completion"
        assert resolved.feature_description == "Basic chat completion"
        assert resolved.owner == "team-ai"

    def test_feature_overrides(self, temp_config_file, basic_config_data):
        """Test feature-specific overrides."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("code_generation")

        # Should override model
        assert resolved.model == "openai:gpt-4"

        # Should override langfuse config
        assert resolved.langfuse.public_key == "pk-code"
        assert resolved.langfuse.secret_key == "sk-code"

        # Should still inherit other defaults including api_keys
        assert "openai" in resolved.provider_api_keys
        assert resolved.provider_api_keys["openai"] == "sk-test-key"

    def test_api_keys_merging(self, temp_config_file):
        """Test API keys are merged, not replaced."""
        config_data = {
            "defaults": {
                "provider_api_keys": {
                    "openai": "default-openai-key",
                    "anthropic": "default-anthropic-key",
                }
            },
            "features": {
                "test_feature": {
                    "description": "Test merging",
                    "owner": "test",
                    "provider_api_keys": {
                        "openai": "feature-openai-key",
                        "google": "feature-google-key",
                    },
                }
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("test_feature")

        # Should merge API keys
        assert resolved.provider_api_keys["openai"] == "feature-openai-key"  # Override
        assert (
            resolved.provider_api_keys["anthropic"] == "default-anthropic-key"
        )  # Inherited
        assert resolved.provider_api_keys["google"] == "feature-google-key"  # New

    def test_none_values_dont_override(self, temp_config_file):
        """Test that None values in features don't override defaults."""
        config_data = {
            "defaults": {
                "model": "openai:gpt-3.5-turbo",
                "langfuse": {
                    "public_key": "pk-default",
                    "secret_key": "sk-default",
                    "enabled": True,
                },
            },
            "features": {
                "test_feature": {
                    "description": "Test",
                    "owner": "test",
                    "model": None,  # Explicit None should not override
                    "langfuse": None,  # Explicit None should not override
                }
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("test_feature")

        # Should inherit defaults despite explicit None
        assert resolved.model == "openai:gpt-3.5-turbo"
        assert resolved.langfuse.public_key == "pk-default"


class TestBudgetConstraints:
    """Test budget constraint handling."""

    def test_budget_constraints_parsing(self, temp_config_file, complex_config_data):
        """Test parsing of budget constraints."""
        # Set up environment variables
        env_vars = {
            "OPENAI_API_KEY": "sk-test-openai",
            "ANTHROPIC_API_KEY": "sk-test-anthropic",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }

        with patch.dict(os.environ, env_vars):
            with open(temp_config_file, "w") as f:
                yaml.dump(complex_config_data, f)

            resolver = ConfigResolver(temp_config_file)
            constraints = resolver.get_limit_constraints("production_chat")

            assert len(constraints) == 2

            # Check cost constraint
            assert "daily-feature-cost-limit" in constraints
            cost_constraint = constraints["daily-feature-cost-limit"]
            assert cost_constraint.name == "daily-feature-cost-limit"
            assert cost_constraint.dimensions == ["feature"]
            assert cost_constraint.budget_limit.limit == 1000.0
            assert cost_constraint.budget_limit.window == "1d"

            # Check rate constraint
            assert "hourly-user-rate-limit" in constraints
            rate_constraint = constraints["hourly-user-rate-limit"]
            assert rate_constraint.name == "hourly-user-rate-limit"
            assert rate_constraint.dimensions == ["user_id"]
            assert rate_constraint.rate_limit.per_minute == 60

    def test_empty_constraints(self, temp_config_file, basic_config_data):
        """Test features with no budget constraints."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)
        constraints = resolver.get_limit_constraints("chat_completion")

        assert constraints == {}

    def test_budget_constraints_merging(self, temp_config_file):
        """Test budget constraints are merged by name, with feature overriding defaults."""
        config_data = {
            "defaults": {
                "budget_constraints": {
                    "global-daily-limit": {
                        "dimensions": ["feature"],
                        "budget_limit": {"limit": 100.0, "window": "1d"},
                        "description": "Global daily limit",
                    },
                    "shared-rate-limit": {
                        "dimensions": ["user_id"],
                        "rate_limit": {"per_minute": 30},
                    },
                }
            },
            "features": {
                "premium_feature": {
                    "description": "Premium feature with custom limits",
                    "owner": "premium-team",
                    "budget_constraints": {
                        "shared-rate-limit": {  # Override default rate limit
                            "dimensions": ["user_id"],
                            "rate_limit": {
                                "per_minute": 100  # Higher limit for premium
                            },
                        },
                        "premium-specific": {  # New constraint only for this feature
                            "dimensions": ["organization"],
                            "budget_limit": {"limit": 500.0, "window": "1d"},
                        },
                    },
                }
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        constraints = resolver.get_limit_constraints("premium_feature")

        assert len(constraints) == 3  # 2 from defaults + 1 new, with 1 overridden

        # Should inherit global-daily-limit from defaults
        assert "global-daily-limit" in constraints
        global_limit = constraints["global-daily-limit"]
        assert global_limit.budget_limit.limit == 100.0

        # Should override shared-rate-limit with feature-specific value
        assert "shared-rate-limit" in constraints
        rate_limit = constraints["shared-rate-limit"]
        assert (
            rate_limit.rate_limit.per_minute == 100
        )  # Feature override, not default 30

        # Should have feature-specific constraint
        assert "premium-specific" in constraints
        premium_limit = constraints["premium-specific"]
        assert premium_limit.dimensions == ["organization"]
        assert premium_limit.budget_limit.limit == 500.0


class TestLangfuseConfiguration:
    """Test Langfuse-specific configuration handling."""

    def test_langfuse_config_enabled(self, temp_config_file, basic_config_data):
        """Test retrieval of enabled Langfuse configuration."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)
        langfuse_config = resolver.get_langfuse_config("chat_completion")

        assert langfuse_config is not None
        assert langfuse_config.public_key == "pk-test"
        assert langfuse_config.secret_key == "sk-test"
        assert langfuse_config.enabled is True
        assert langfuse_config.base_url == "https://cloud.langfuse.com"

    def test_langfuse_config_disabled(self, temp_config_file):
        """Test handling of disabled Langfuse configuration."""
        config_data = {
            "defaults": {
                "langfuse": {
                    "public_key": "pk-test",
                    "secret_key": "sk-test",
                    "enabled": False,
                }
            },
            "features": {"test_feature": {"description": "Test", "owner": "test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        langfuse_config = resolver.get_langfuse_config("test_feature")

        assert langfuse_config is None

    def test_langfuse_config_missing(self, temp_config_file):
        """Test handling when Langfuse config is not provided."""
        config_data = {
            "defaults": {},
            "features": {"test_feature": {"description": "Test", "owner": "test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        langfuse_config = resolver.get_langfuse_config("test_feature")

        assert langfuse_config is None

    def test_langfuse_secret_key_validation(self):
        """Test validation of Langfuse secret key."""
        # Test empty secret key
        with pytest.raises(ValueError, match="secret_key cannot be empty"):
            LangfuseConfig(public_key="pk-test", secret_key="", enabled=True)

        # Test whitespace-only secret key
        with pytest.raises(ValueError, match="secret_key cannot be empty"):
            LangfuseConfig(public_key="pk-test", secret_key="   ", enabled=True)


class TestResolverUtilities:
    """Test utility methods of ConfigResolver."""

    def test_get_feature_names(self, temp_config_file, basic_config_data):
        """Test retrieval of feature names."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)
        feature_names = resolver.get_feature_names()

        assert set(feature_names) == {"chat_completion", "code_generation"}

    def test_feature_exists(self, temp_config_file, basic_config_data):
        """Test feature existence checking."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)

        assert resolver.feature_exists("chat_completion") is True
        assert resolver.feature_exists("code_generation") is True
        assert resolver.feature_exists("nonexistent_feature") is False

    def test_resolve_nonexistent_feature(self, temp_config_file, basic_config_data):
        """Test error handling when resolving nonexistent feature."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)

        with pytest.raises(ConfigurationError, match="Unknown feature: nonexistent"):
            resolver.resolve_feature_config("nonexistent")

    def test_configuration_caching(self, temp_config_file, basic_config_data):
        """Test that resolved configurations are cached."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)

        # First resolution
        resolved1 = resolver.resolve_feature_config("chat_completion")

        # Second resolution should return cached result
        resolved2 = resolver.resolve_feature_config("chat_completion")

        assert resolved1 is resolved2  # Same object reference


class TestConfigurationValidation:
    """Test configuration validation and warning generation."""

    def test_validate_configuration_no_warnings(
        self, temp_config_file, basic_config_data
    ):
        """Test validation with no warnings."""
        with open(temp_config_file, "w") as f:
            yaml.dump(basic_config_data, f)

        resolver = ConfigResolver(temp_config_file)
        warnings = resolver.validate_configuration()

        assert warnings == []

    def test_validate_missing_langfuse_secret(self, temp_config_file):
        """Test warning for missing Langfuse secret key."""
        config_data = {
            "defaults": {
                "langfuse": {
                    "public_key": "pk-test",
                    "secret_key": "${MISSING_SECRET}",
                    "enabled": True,
                }
            },
            "features": {"test_feature": {"description": "Test", "owner": "test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        # Don't set the environment variable
        if "MISSING_SECRET" in os.environ:
            del os.environ["MISSING_SECRET"]

        with pytest.raises(ConfigurationError):
            # Should fail during loading due to missing env var
            ConfigResolver(temp_config_file)


class TestCustomConfiguration:
    """Test custom configuration fields."""

    def test_custom_config_inheritance(self, temp_config_file, complex_config_data):
        """Test custom configuration field handling."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-openai",
            "ANTHROPIC_API_KEY": "sk-test-anthropic",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }

        with patch.dict(os.environ, env_vars):
            with open(temp_config_file, "w") as f:
                yaml.dump(complex_config_data, f)

            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config("production_chat")

            assert resolved.custom["priority"] == "high"
            assert resolved.custom["alerts_enabled"] is True


class TestProviderConfiguration:
    """Test custom provider configuration."""

    def test_basic_provider_config(self, temp_config_file):
        """Test basic provider configuration parsing."""
        config_data = {
            "defaults": {
                "providers": {
                    "my-custom-openai": {
                        "provider": "openai",
                        "api_base": "https://custom.example.com/v1",
                    }
                },
                "provider_api_keys": {"my-custom-openai": "sk-custom-key"},
            },
            "features": {
                "test": {"description": "Test", "model": "my-custom-openai:gpt-4"}
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("test")

        assert "my-custom-openai" in resolved.providers
        assert resolved.providers["my-custom-openai"].provider == "openai"
        assert (
            resolved.providers["my-custom-openai"].api_base
            == "https://custom.example.com/v1"
        )
        assert resolved.provider_api_keys["my-custom-openai"] == "sk-custom-key"

    def test_provider_inheritance(self, temp_config_file):
        """Test provider configuration inheritance from defaults to features."""
        config_data = {
            "defaults": {
                "providers": {
                    "default-provider": {
                        "provider": "openai",
                        "api_base": "https://default.example.com/v1",
                    }
                }
            },
            "features": {"test": {"description": "Test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("test")

        # Feature should inherit providers from defaults
        assert "default-provider" in resolved.providers
        assert resolved.providers["default-provider"].provider == "openai"

    def test_provider_merging(self, temp_config_file):
        """Test provider configuration merging (feature overrides/extends defaults)."""
        config_data = {
            "defaults": {
                "providers": {
                    "provider-a": {
                        "provider": "openai",
                        "api_base": "https://a.example.com/v1",
                    },
                    "provider-b": {
                        "provider": "anthropic",
                        "api_base": "https://b.example.com/v1",
                    },
                }
            },
            "features": {
                "test": {
                    "description": "Test",
                    "providers": {
                        "provider-b": {
                            "provider": "anthropic",
                            "api_base": "https://b-override.example.com/v1",
                        },
                        "provider-c": {
                            "provider": "ollama",
                            "api_base": "http://localhost:11434",
                        },
                    },
                }
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("test")

        # provider-a inherited from defaults
        assert resolved.providers["provider-a"].api_base == "https://a.example.com/v1"
        # provider-b overridden by feature
        assert (
            resolved.providers["provider-b"].api_base
            == "https://b-override.example.com/v1"
        )
        # provider-c added by feature
        assert resolved.providers["provider-c"].provider == "ollama"

    def test_provider_env_var_resolution(self, temp_config_file):
        """Test environment variable resolution in provider config."""
        config_data = {
            "defaults": {
                "providers": {
                    "my-provider": {
                        "provider": "openai",
                        "api_base": "${CUSTOM_API_BASE}",
                    }
                }
            },
            "features": {"test": {"description": "Test"}},
        }

        env_vars = {"CUSTOM_API_BASE": "https://resolved.example.com/v1"}

        with patch.dict(os.environ, env_vars):
            with open(temp_config_file, "w") as f:
                yaml.dump(config_data, f)

            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config("test")

            assert (
                resolved.providers["my-provider"].api_base
                == "https://resolved.example.com/v1"
            )

    def test_invalid_provider_name(self, temp_config_file):
        """Test that invalid provider names are rejected."""
        config_data = {
            "defaults": {
                "providers": {"invalid provider name!": {"provider": "openai"}}
            },
            "features": {"test": {"description": "Test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigurationError, match="Invalid provider name"):
            ConfigResolver(temp_config_file)

    def test_provider_extra_options(self, temp_config_file):
        """Test that extra options are preserved in provider config."""
        config_data = {
            "defaults": {
                "providers": {
                    "azure-openai": {
                        "provider": "azure",
                        "api_base": "https://myorg.openai.azure.com",
                        "api_version": "2024-02-01",
                        "organization": "my-org",
                    }
                }
            },
            "features": {"test": {"description": "Test"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config("test")

        provider = resolved.providers["azure-openai"]
        # The extra='allow' in ConfigDict should preserve these
        provider_dict = provider.model_dump()
        assert provider_dict["provider"] == "azure"
        assert provider_dict["api_base"] == "https://myorg.openai.azure.com"
        assert provider_dict["api_version"] == "2024-02-01"
        assert provider_dict["organization"] == "my-org"


class TestProviderResolution:
    """Test provider resolution in the pipeline - unit tests."""

    def test_standard_provider_resolution(self):
        """Test resolving a standard provider (not custom)."""
        from sllmp.pipeline import _resolve_provider

        resolved = _resolve_provider("openai:gpt-4", {})

        assert resolved.underlying_provider == "openai"
        assert resolved.model_suffix == "gpt-4"
        assert resolved.full_model_id == "openai:gpt-4"
        assert resolved.extra_options == {}
        assert resolved.api_key_lookup == "openai"

    def test_custom_provider_resolution(self):
        """Test resolving a custom provider."""
        from sllmp.pipeline import _resolve_provider

        providers = {
            "my-custom-openai": {
                "provider": "openai",
                "api_base": "https://custom.example.com/v1",
            }
        }

        resolved = _resolve_provider("my-custom-openai:gpt-4", providers)

        assert resolved.underlying_provider == "openai"
        assert resolved.model_suffix == "gpt-4"
        assert resolved.full_model_id == "openai:gpt-4"  # Maps to underlying provider
        assert resolved.extra_options == {"api_base": "https://custom.example.com/v1"}
        assert (
            resolved.api_key_lookup == "my-custom-openai"
        )  # Uses custom name for API key

    def test_no_colon_model_id(self):
        """Test handling model_id without provider prefix."""
        from sllmp.pipeline import _resolve_provider

        resolved = _resolve_provider("gpt-4", {})

        assert resolved.underlying_provider == "unknown"
        assert resolved.model_suffix == "gpt-4"
        assert resolved.full_model_id == "gpt-4"
        assert resolved.extra_options == {}
        assert resolved.api_key_lookup == "unknown"

    def test_custom_provider_all_options_passed(self):
        """Test that all custom provider options are passed through."""
        from sllmp.pipeline import _resolve_provider

        providers = {
            "azure-custom": {
                "provider": "azure",
                "api_base": "https://myorg.openai.azure.com",
                "api_version": "2024-02-01",
                "organization": "my-org",
                "timeout": 30,
            }
        }

        resolved = _resolve_provider("azure-custom:gpt-4", providers)

        assert resolved.extra_options == {
            "api_base": "https://myorg.openai.azure.com",
            "api_version": "2024-02-01",
            "organization": "my-org",
            "timeout": 30,
        }


class TestProviderConfigIntegration:
    """Integration tests for provider configuration - full flow from HTTP request to LLM call."""

    @pytest.fixture
    def mock_llm_completion(self):
        """Mock any_llm.acompletion and capture call arguments."""
        from any_llm.types.completion import ChatCompletion

        captured_calls = []

        def create_completion(**kwargs):
            # Capture all call arguments for verification
            captured_calls.append(kwargs)
            model_id = kwargs.get("model", "openai:gpt-4")
            return ChatCompletion(
                id="chatcmpl-test",
                object="chat.completion",
                created=1234567890,
                model=model_id,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        async def mock_acompletion(stream=False, **kwargs):
            return create_completion(**kwargs)

        with patch(
            "sllmp.pipeline.any_llm.acompletion", side_effect=mock_acompletion
        ) as mock:
            mock.captured_calls = captured_calls
            yield mock

    @pytest.fixture
    async def configured_client(self, temp_config_file, mock_llm_completion):
        """Create an HTTP client with configuration middleware and custom provider."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        # Write config with custom provider
        config_data = {
            "defaults": {
                "providers": {
                    "my-custom-openai": {
                        "provider": "openai",
                        "api_base": "https://custom.example.com/v1",
                        "organization": "test-org",
                    }
                },
                "provider_api_keys": {"my-custom-openai": "sk-custom-key-123"},
            },
            "features": {
                "default": {
                    "description": "Default feature",
                    "model": "my-custom-openai:gpt-4",
                }
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        # Create server with configuration middleware
        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            yield client, mock_llm_completion

    async def test_custom_provider_api_base_passed_to_llm(self, configured_client):
        """Test that custom provider's api_base is passed to any_llm.acompletion."""
        client, mock = configured_client

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "my-custom-openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200

        # Verify acompletion was called with correct arguments
        assert len(mock.captured_calls) == 1
        call_args = mock.captured_calls[0]

        # Model should be mapped to underlying provider
        assert call_args["model"] == "openai:gpt-4"

        # Custom provider options should be passed
        assert call_args["api_base"] == "https://custom.example.com/v1"
        assert call_args["organization"] == "test-org"

        # API key should be resolved from provider_api_keys using custom provider name
        assert call_args["api_key"] == "sk-custom-key-123"

    async def test_custom_provider_with_feature_override(
        self, temp_config_file, mock_llm_completion
    ):
        """Test that feature-level provider config overrides defaults."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        config_data = {
            "defaults": {
                "providers": {
                    "base-provider": {
                        "provider": "openai",
                        "api_base": "https://default.example.com/v1",
                    }
                },
                "provider_api_keys": {"base-provider": "sk-default-key"},
            },
            "features": {
                "default": {
                    "description": "Default feature",
                    "providers": {
                        "base-provider": {
                            "provider": "openai",
                            "api_base": "https://feature-override.example.com/v1",
                            "custom_header": "feature-value",
                        }
                    },
                    "model": "base-provider:gpt-4",
                }
            },
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "base-provider:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 200

        # Verify feature-level override was used
        call_args = mock_llm_completion.captured_calls[0]
        assert call_args["api_base"] == "https://feature-override.example.com/v1"
        assert call_args["custom_header"] == "feature-value"

    async def test_standard_provider_no_extra_options(
        self, temp_config_file, mock_llm_completion
    ):
        """Test that standard providers (not custom) work without extra options."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        config_data = {
            "defaults": {"provider_api_keys": {"openai": "sk-standard-key"}},
            "features": {"default": {"description": "Default feature"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 200

        # Verify standard provider behavior
        call_args = mock_llm_completion.captured_calls[0]
        assert call_args["model"] == "openai:gpt-4"  # Not remapped
        assert call_args["api_key"] == "sk-standard-key"

        # Should not have extra options (api_base, etc.)
        assert "api_base" not in call_args
        assert "organization" not in call_args

    async def test_multiple_custom_providers(
        self, temp_config_file, mock_llm_completion
    ):
        """Test that multiple custom providers can be configured and used."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        config_data = {
            "defaults": {
                "providers": {
                    "provider-a": {
                        "provider": "openai",
                        "api_base": "https://provider-a.example.com/v1",
                    },
                    "provider-b": {
                        "provider": "anthropic",
                        "api_base": "https://provider-b.example.com/v1",
                    },
                },
                "provider_api_keys": {
                    "provider-a": "sk-key-a",
                    "provider-b": "sk-key-b",
                },
            },
            "features": {"default": {"description": "Default feature"}},
        }

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # First request with provider-a
            response1 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "provider-a:gpt-4",
                    "messages": [{"role": "user", "content": "Hello from A"}],
                },
            )

            # Second request with provider-b
            response2 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "provider-b:claude-3",
                    "messages": [{"role": "user", "content": "Hello from B"}],
                },
            )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Verify both providers were used correctly
        call1 = mock_llm_completion.captured_calls[0]
        assert call1["model"] == "openai:gpt-4"
        assert call1["api_base"] == "https://provider-a.example.com/v1"
        assert call1["api_key"] == "sk-key-a"

        call2 = mock_llm_completion.captured_calls[1]
        assert call2["model"] == "anthropic:claude-3"
        assert call2["api_base"] == "https://provider-b.example.com/v1"
        assert call2["api_key"] == "sk-key-b"


class TestRetryIntegration:
    """Integration tests for retry middleware - full flow from HTTP request through retry logic."""

    @pytest.fixture
    def mock_llm_completion_with_failures(self):
        """Mock any_llm.acompletion that can be configured to fail N times before succeeding."""
        from any_llm.types.completion import ChatCompletion

        call_count = [0]  # Use list to allow mutation in nested function
        failures_before_success = [0]
        failure_exception = [None]
        captured_calls = []

        def create_completion(**kwargs):
            model_id = kwargs.get("model", "openai:gpt-4")
            return ChatCompletion(
                id="chatcmpl-test",
                object="chat.completion",
                created=1234567890,
                model=model_id,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Success after retries!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        async def mock_acompletion(stream=False, **kwargs):
            captured_calls.append(kwargs)
            call_count[0] += 1

            # Fail the first N calls if configured
            if call_count[0] <= failures_before_success[0] and failure_exception[0]:
                raise failure_exception[0]

            return create_completion(**kwargs)

        with patch(
            "sllmp.pipeline.any_llm.acompletion", side_effect=mock_acompletion
        ) as mock:
            mock.call_count_ref = call_count
            mock.failures_before_success = failures_before_success
            mock.failure_exception = failure_exception
            mock.captured_calls = captured_calls
            yield mock

    @pytest.fixture
    def basic_retry_config(self):
        """Basic configuration for retry tests."""
        return {
            "defaults": {"provider_api_keys": {"openai": "sk-test-key"}},
            "features": {
                "default": {
                    "description": "Default feature with retry",
                    "model": "openai:gpt-4",
                }
            },
        }

    async def test_retry_on_network_error_then_success(
        self, temp_config_file, basic_retry_config, mock_llm_completion_with_failures
    ):
        """Test that network errors are retried and eventually succeed."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        with open(temp_config_file, "w") as f:
            yaml.dump(basic_retry_config, f)

        # Configure mock to fail twice with network error, then succeed
        mock_llm_completion_with_failures.failures_before_success[0] = 2
        mock_llm_completion_with_failures.failure_exception[0] = Exception(
            "Connection timeout"
        )

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        # Should succeed after retries
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Success after retries!"

        # Verify that acompletion was called 3 times (2 failures + 1 success)
        assert len(mock_llm_completion_with_failures.captured_calls) == 3

    async def test_retry_on_rate_limit_error_then_success(
        self, temp_config_file, basic_retry_config, mock_llm_completion_with_failures
    ):
        """Test that rate limit errors are retried and eventually succeed."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        with open(temp_config_file, "w") as f:
            yaml.dump(basic_retry_config, f)

        # Configure mock to fail once with rate limit error, then succeed
        mock_llm_completion_with_failures.failures_before_success[0] = 1
        mock_llm_completion_with_failures.failure_exception[0] = AnyLLMRateLimitError(
            "Rate limit exceeded"
        )

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        # Should succeed after retry
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Success after retries!"

        # Verify that acompletion was called 2 times (1 failure + 1 success)
        assert len(mock_llm_completion_with_failures.captured_calls) == 2

    async def test_retry_exhaustion_returns_error(
        self, temp_config_file, basic_retry_config, mock_llm_completion_with_failures
    ):
        """Test that after exhausting retries, the error is returned to the client."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        with open(temp_config_file, "w") as f:
            yaml.dump(basic_retry_config, f)

        # Configure mock to always fail (more failures than max retries)
        mock_llm_completion_with_failures.failures_before_success[0] = 100
        mock_llm_completion_with_failures.failure_exception[0] = AnyLLMProviderError(
            "Service unavailable"
        )

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        # Should return error after exhausting retries
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "unavailable" in data["error"]["message"].lower()

        # Default retry middleware has max_attempts=3, which means 4 total calls
        # (1 original attempt + 3 retries)
        assert len(mock_llm_completion_with_failures.captured_calls) == 4

    async def test_non_retryable_error_not_retried(
        self, temp_config_file, basic_retry_config, mock_llm_completion_with_failures
    ):
        """Test that authentication errors are NOT retried."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        with open(temp_config_file, "w") as f:
            yaml.dump(basic_retry_config, f)

        # Configure mock to fail with auth error (not retryable)
        mock_llm_completion_with_failures.failures_before_success[0] = 100
        mock_llm_completion_with_failures.failure_exception[0] = (
            AnyLLMAuthenticationError("Invalid API key")
        )

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        # Should return error immediately without retrying
        assert response.status_code == 401
        data = response.json()
        assert "error" in data

        # Should only be called once (no retries for auth errors)
        assert len(mock_llm_completion_with_failures.captured_calls) == 1

    async def test_retry_on_internal_error_then_success(
        self, temp_config_file, basic_retry_config, mock_llm_completion_with_failures
    ):
        """Test that internal server errors (500) are retried."""
        from sllmp import SimpleProxyServer
        from sllmp.config.middleware import configuration_middleware

        with open(temp_config_file, "w") as f:
            yaml.dump(basic_retry_config, f)

        # Configure mock to fail once with internal error, then succeed
        mock_llm_completion_with_failures.failures_before_success[0] = 1
        mock_llm_completion_with_failures.failure_exception[0] = AnyLLMProviderError(
            "Internal server error"
        )

        def create_pipeline():
            from sllmp.context import Pipeline

            pipeline = Pipeline()
            pipeline.setup.connect(configuration_middleware(temp_config_file))
            return pipeline

        server = SimpleProxyServer(pipeline_factory=create_pipeline)
        app = server.create_asgi_app(debug=True)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai:gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        # Should succeed after retry
        assert response.status_code == 200

        # Verify that acompletion was called 2 times (1 failure + 1 success)
        assert len(mock_llm_completion_with_failures.captured_calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
