"""
Comprehensive tests for the configuration system.

Tests cover ConfigResolver, environment variable resolution, inheritance,
validation, and error handling scenarios.
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from sllmp.config.config import (
    ConfigResolver, ConfigFile, DefaultsConfig, FeatureConfig,
    LangfuseConfig, ResolvedFeatureConfig, ConfigurationError
)
from sllmp.middleware.limit.limit import Constraint


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def basic_config_data():
    """Basic valid configuration data."""
    return {
        'defaults': {
            'model': 'openai:gpt-3.5-turbo',
            'provider_api_keys': {
                'openai': 'sk-test-key'
            },
            'langfuse': {
                'public_key': 'pk-test',
                'secret_key': 'sk-test',
                'enabled': True
            }
        },
        'features': {
            'chat_completion': {
                'description': 'Basic chat completion',
                'owner': 'team-ai'
            },
            'code_generation': {
                'description': 'Code generation feature',
                'owner': 'team-dev',
                'model': 'openai:gpt-4',
                'langfuse': {
                    'public_key': 'pk-code',
                    'secret_key': 'sk-code',
                    'enabled': True
                }
            }
        }
    }


@pytest.fixture
def complex_config_data():
    """Complex configuration with constraints and environment variables."""
    return {
        'defaults': {
            'provider_api_keys': {
                'openai': '${OPENAI_API_KEY}',
                'anthropic': '${ANTHROPIC_API_KEY}'
            },
            'langfuse': {
                'public_key': '${LANGFUSE_PUBLIC_KEY}',
                'secret_key': '${LANGFUSE_SECRET_KEY}',
                'base_url': 'https://custom.langfuse.com'
            }
        },
        'features': {
            'production_chat': {
                'description': 'Production chat with limits',
                'owner': 'prod-team',
                'budget_constraints': [
                    {
                        'dimension': 'cost',
                        'limit': 1000.0,
                        'window_minutes': 1440,
                        'entity': 'feature'
                    },
                    {
                        'dimension': 'requests',
                        'limit': 10000,
                        'window_minutes': 60,
                        'entity': 'user'
                    }
                ],
                'custom': {
                    'priority': 'high',
                    'alerts_enabled': True
                }
            }
        }
    }


class TestConfigurationLoading:
    """Test configuration file loading and basic validation."""

    def test_load_valid_config(self, temp_config_file, basic_config_data):
        """Test loading a valid configuration file."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        
        assert resolver.config_file == temp_config_file
        assert isinstance(resolver.config, ConfigFile)
        assert len(resolver.config.features) == 2
        assert 'chat_completion' in resolver.config.features
        assert 'code_generation' in resolver.config.features

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent config file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            ConfigResolver('/nonexistent/config.yaml')

    def test_load_empty_file(self, temp_config_file):
        """Test error handling for empty config file."""
        # Create empty file
        open(temp_config_file, 'w').close()
        
        with pytest.raises(ConfigurationError, match="Configuration file is empty"):
            ConfigResolver(temp_config_file)

    def test_load_invalid_yaml(self, temp_config_file):
        """Test error handling for invalid YAML syntax."""
        with open(temp_config_file, 'w') as f:
            f.write("invalid: yaml: content: [\n")
        
        with pytest.raises(ConfigurationError, match="Invalid YAML syntax"):
            ConfigResolver(temp_config_file)

    def test_load_missing_required_fields(self, temp_config_file):
        """Test validation of required configuration fields."""
        incomplete_config = {
            'defaults': {},
            # Missing features section
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            ConfigResolver(temp_config_file)

    def test_invalid_feature_names(self, temp_config_file):
        """Test validation of feature names."""
        config_with_invalid_names = {
            'defaults': {},
            'features': {
                'invalid-feature!': {
                    'description': 'Invalid name',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_with_invalid_names, f)
        
        with pytest.raises(ConfigurationError, match="Invalid feature name"):
            ConfigResolver(temp_config_file)


class TestEnvironmentVariableResolution:
    """Test environment variable substitution in configuration."""

    def test_env_var_resolution_success(self, temp_config_file):
        """Test successful environment variable resolution."""
        config_data = {
            'defaults': {
                'provider_api_keys': {
                    'openai': '${TEST_API_KEY}'
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch.dict(os.environ, {'TEST_API_KEY': 'resolved-key-123'}):
            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config('test_feature')
            
            assert resolved.provider_api_keys['openai'] == 'resolved-key-123'

    def test_env_var_resolution_missing_var(self, temp_config_file):
        """Test error handling for missing environment variables."""
        config_data = {
            'defaults': {
                'provider_api_keys': {
                    'openai': '${MISSING_API_KEY}'
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Ensure the environment variable doesn't exist
        if 'MISSING_API_KEY' in os.environ:
            del os.environ['MISSING_API_KEY']
        
        with pytest.raises(ConfigurationError, match="Environment variable not found: MISSING_API_KEY"):
            ConfigResolver(temp_config_file)

    def test_nested_env_var_resolution(self, temp_config_file):
        """Test environment variable resolution in nested structures."""
        config_data = {
            'defaults': {
                'langfuse': {
                    'public_key': '${LANGFUSE_PK}',
                    'secret_key': '${LANGFUSE_SK}',
                    'default_prompt_label': '${PROMPT_LABEL}'
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        env_vars = {
            'LANGFUSE_PK': 'pk-resolved',
            'LANGFUSE_SK': 'sk-resolved', 
            'PROMPT_LABEL': 'production'
        }
        
        with patch.dict(os.environ, env_vars):
            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config('test_feature')
            
            assert resolved.langfuse.public_key == 'pk-resolved'
            assert resolved.langfuse.secret_key == 'sk-resolved'
            assert resolved.langfuse.default_prompt_label == 'production'


class TestConfigurationInheritance:
    """Test configuration inheritance from defaults to features."""

    def test_basic_inheritance(self, temp_config_file, basic_config_data):
        """Test basic configuration inheritance."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config('chat_completion')
        
        # Should inherit defaults
        assert resolved.model == 'openai:gpt-3.5-turbo'
        assert resolved.provider_api_keys['openai'] == 'sk-test-key'
        assert resolved.langfuse.public_key == 'pk-test'
        assert resolved.langfuse.enabled is True
        
        # Should have feature-specific metadata
        assert resolved.feature_name == 'chat_completion'
        assert resolved.feature_description == 'Basic chat completion'
        assert resolved.owner == 'team-ai'

    def test_feature_overrides(self, temp_config_file, basic_config_data):
        """Test feature-specific overrides."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config('code_generation')
        
        # Should override model
        assert resolved.model == 'openai:gpt-4'
        
        # Should override langfuse config
        assert resolved.langfuse.public_key == 'pk-code'
        assert resolved.langfuse.secret_key == 'sk-code'
        
        # Should still inherit other defaults
        assert resolved.provider_api_keys['openai'] == 'sk-test-key'

    def test_api_keys_merging(self, temp_config_file):
        """Test API keys are merged, not replaced."""
        config_data = {
            'defaults': {
                'provider_api_keys': {
                    'openai': 'default-openai-key',
                    'anthropic': 'default-anthropic-key'
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test merging',
                    'owner': 'test',
                    'provider_api_keys': {
                        'openai': 'feature-openai-key',
                        'google': 'feature-google-key'
                    }
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config('test_feature')
        
        # Should merge API keys
        assert resolved.provider_api_keys['openai'] == 'feature-openai-key'  # Override
        assert resolved.provider_api_keys['anthropic'] == 'default-anthropic-key'  # Inherited
        assert resolved.provider_api_keys['google'] == 'feature-google-key'  # New

    def test_none_values_dont_override(self, temp_config_file):
        """Test that None values in features don't override defaults."""
        config_data = {
            'defaults': {
                'model': 'openai:gpt-3.5-turbo',
                'langfuse': {
                    'public_key': 'pk-default',
                    'secret_key': 'sk-default',
                    'enabled': True
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test',
                    'model': None,  # Explicit None should not override
                    'langfuse': None  # Explicit None should not override
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        resolved = resolver.resolve_feature_config('test_feature')
        
        # Should inherit defaults despite explicit None
        assert resolved.model == 'openai:gpt-3.5-turbo'
        assert resolved.langfuse.public_key == 'pk-default'


class TestBudgetConstraints:
    """Test budget constraint handling."""

    def test_budget_constraints_parsing(self, temp_config_file, complex_config_data):
        """Test parsing of budget constraints."""
        # Set up environment variables
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-openai',
            'ANTHROPIC_API_KEY': 'sk-test-anthropic',
            'LANGFUSE_PUBLIC_KEY': 'pk-test',
            'LANGFUSE_SECRET_KEY': 'sk-test'
        }
        
        with patch.dict(os.environ, env_vars):
            with open(temp_config_file, 'w') as f:
                yaml.dump(complex_config_data, f)
            
            resolver = ConfigResolver(temp_config_file)
            constraints = resolver.get_limit_constraints('production_chat')
            
            assert len(constraints) == 2
            
            # Check cost constraint
            cost_constraint = constraints[0]
            assert cost_constraint.dimension == 'cost'
            assert cost_constraint.limit == 1000.0
            assert cost_constraint.window_minutes == 1440
            assert cost_constraint.entity == 'feature'
            
            # Check requests constraint
            req_constraint = constraints[1]
            assert req_constraint.dimension == 'requests'
            assert req_constraint.limit == 10000
            assert req_constraint.window_minutes == 60
            assert req_constraint.entity == 'user'

    def test_empty_constraints(self, temp_config_file, basic_config_data):
        """Test features with no budget constraints."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        constraints = resolver.get_limit_constraints('chat_completion')
        
        assert constraints == []


class TestLangfuseConfiguration:
    """Test Langfuse-specific configuration handling."""

    def test_langfuse_config_enabled(self, temp_config_file, basic_config_data):
        """Test retrieval of enabled Langfuse configuration."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        langfuse_config = resolver.get_langfuse_config('chat_completion')
        
        assert langfuse_config is not None
        assert langfuse_config.public_key == 'pk-test'
        assert langfuse_config.secret_key == 'sk-test'
        assert langfuse_config.enabled is True
        assert langfuse_config.base_url == 'https://cloud.langfuse.com'

    def test_langfuse_config_disabled(self, temp_config_file):
        """Test handling of disabled Langfuse configuration."""
        config_data = {
            'defaults': {
                'langfuse': {
                    'public_key': 'pk-test',
                    'secret_key': 'sk-test',
                    'enabled': False
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        langfuse_config = resolver.get_langfuse_config('test_feature')
        
        assert langfuse_config is None

    def test_langfuse_config_missing(self, temp_config_file):
        """Test handling when Langfuse config is not provided."""
        config_data = {
            'defaults': {},
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        langfuse_config = resolver.get_langfuse_config('test_feature')
        
        assert langfuse_config is None

    def test_langfuse_secret_key_validation(self):
        """Test validation of Langfuse secret key."""
        # Test empty secret key
        with pytest.raises(ValueError, match="secret_key cannot be empty"):
            LangfuseConfig(
                public_key='pk-test',
                secret_key='',
                enabled=True
            )
        
        # Test whitespace-only secret key
        with pytest.raises(ValueError, match="secret_key cannot be empty"):
            LangfuseConfig(
                public_key='pk-test',
                secret_key='   ',
                enabled=True
            )


class TestResolverUtilities:
    """Test utility methods of ConfigResolver."""

    def test_get_feature_names(self, temp_config_file, basic_config_data):
        """Test retrieval of feature names."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        feature_names = resolver.get_feature_names()
        
        assert set(feature_names) == {'chat_completion', 'code_generation'}

    def test_feature_exists(self, temp_config_file, basic_config_data):
        """Test feature existence checking."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        
        assert resolver.feature_exists('chat_completion') is True
        assert resolver.feature_exists('code_generation') is True
        assert resolver.feature_exists('nonexistent_feature') is False

    def test_resolve_nonexistent_feature(self, temp_config_file, basic_config_data):
        """Test error handling when resolving nonexistent feature."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        
        with pytest.raises(ConfigurationError, match="Unknown feature: nonexistent"):
            resolver.resolve_feature_config('nonexistent')

    def test_configuration_caching(self, temp_config_file, basic_config_data):
        """Test that resolved configurations are cached."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        
        # First resolution
        resolved1 = resolver.resolve_feature_config('chat_completion')
        
        # Second resolution should return cached result
        resolved2 = resolver.resolve_feature_config('chat_completion')
        
        assert resolved1 is resolved2  # Same object reference


class TestConfigurationValidation:
    """Test configuration validation and warning generation."""

    def test_validate_configuration_no_warnings(self, temp_config_file, basic_config_data):
        """Test validation with no warnings."""
        with open(temp_config_file, 'w') as f:
            yaml.dump(basic_config_data, f)
        
        resolver = ConfigResolver(temp_config_file)
        warnings = resolver.validate_configuration()
        
        assert warnings == []

    def test_validate_missing_langfuse_secret(self, temp_config_file):
        """Test warning for missing Langfuse secret key."""
        config_data = {
            'defaults': {
                'langfuse': {
                    'public_key': 'pk-test',
                    'secret_key': '${MISSING_SECRET}',
                    'enabled': True
                }
            },
            'features': {
                'test_feature': {
                    'description': 'Test',
                    'owner': 'test'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Don't set the environment variable
        if 'MISSING_SECRET' in os.environ:
            del os.environ['MISSING_SECRET']
        
        with pytest.raises(ConfigurationError):
            # Should fail during loading due to missing env var
            ConfigResolver(temp_config_file)


class TestCustomConfiguration:
    """Test custom configuration fields."""

    def test_custom_config_inheritance(self, temp_config_file, complex_config_data):
        """Test custom configuration field handling."""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-openai',
            'ANTHROPIC_API_KEY': 'sk-test-anthropic',
            'LANGFUSE_PUBLIC_KEY': 'pk-test',
            'LANGFUSE_SECRET_KEY': 'sk-test'
        }
        
        with patch.dict(os.environ, env_vars):
            with open(temp_config_file, 'w') as f:
                yaml.dump(complex_config_data, f)
            
            resolver = ConfigResolver(temp_config_file)
            resolved = resolver.resolve_feature_config('production_chat')
            
            assert resolved.custom['priority'] == 'high'
            assert resolved.custom['alerts_enabled'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])