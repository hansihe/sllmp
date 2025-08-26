"""
Configuration middleware for dynamic pipeline extension.

This middleware is the ONLY component that knows about the specific configuration
format. It detects features, loads their configuration, and dynamically extends
the pipeline with appropriate middleware based on the feature's needs.

Other organizations can replace this middleware with their own configuration
approach while keeping the same generic middleware components.
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

from simple_llm_proxy.util.signal import HaltExecution

from .. import logger
from ..pipeline import RequestContext
from .config import ConfigResolver, ConfigurationError

# Import generic middleware (these have no config dependencies)
from ..middleware.limit import limit_enforcement_middleware, BaseLimitBackend
#from ..middleware.guardrails import ContentGuardrailMiddleware, ResponseValidatorMiddleware
from ..middleware.logging import logging_middleware

if TYPE_CHECKING:
    # Import for type hints only to avoid circular imports
    pass

def configuration_middleware(
    config_file: str,
    limit_backend: Optional[BaseLimitBackend]=None,
):
    """
    Configuration middleware that dynamically extends the pipeline.

    This middleware:
    1. Detects which feature is being used from the request
    2. Loads the feature's configuration with defaults inheritance
    3. Applies configuration to the current request (model, etc.)
    4. Dynamically adds feature-specific middleware to the pipeline

    This is the ONLY middleware that understands the specific config format.
    """

    try:
        config_resolver = ConfigResolver(config_file)
    except ConfigurationError as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")

    limit_backend = limit_backend

    # Validate configuration on startup
    warnings = config_resolver.validate_configuration()
    if warnings:
        for warning in warnings:
            print(f"⚠️  Configuration warning: {warning}")

    async def setup(ctx: RequestContext):
        try:
            # Step 1: Feature detection
            feature_name = _detect_feature(ctx, config_resolver)
            if not feature_name:
                raise HaltExecution("Unable to determine feature from request")
                # self.halt_with_message(ctx, "Unable to determine feature from request")
                # return ctx

            # Step 2: Load and resolve feature configuration
            feature_config = config_resolver.resolve_feature_config(feature_name)

            # Step 3: Store configuration in context for other middleware
            ctx.state['feature'] = {
                'name': feature_name,
                'config': feature_config,
                'description': feature_config['feature_description'],
                'owners': feature_config['feature_owners']
            }

            # Step 4: Apply request-level configuration
            _apply_request_config(ctx, feature_config)

            # Step 5: Dynamically extend pipeline with feature-specific middleware
            await _extend_pipeline_for_feature(ctx, config_resolver, limit_backend, feature_config)

            # Log feature detection for debugging
            ctx.metadata['detected_feature'] = feature_name
            ctx.metadata['config_applied'] = True

        except ConfigurationError as e:
            raise HaltExecution(f"Configuration error: {e}")
            # self.halt_with_message(ctx, f"Configuration error: {e}")
        except Exception as e:
            raise HaltExecution(f"Unexpected configuration error: {e}")
            # self.halt_with_message(ctx, f"Unexpected configuration error: {e}")

        return ctx

    return setup

def _detect_feature(ctx: RequestContext, config: ConfigResolver) -> Optional[str]:
    """
    Detect which feature is being used from the request.

    Detection methods (in order of priority):
    1. Explicit 'feature' in client metadata (from OpenAI metadata field)
    2. Custom 'x-ai-feature' header
    3. Feature detection from request characteristics
    4. Default feature for the client
    """
    # Method 1: Explicit feature in metadata
    feature_name = ctx.client_metadata.get('feature')
    if feature_name and config.feature_exists(feature_name):
        return feature_name

    # Method 2: Custom header
    feature_name = ctx.client_metadata.get('x-ai-feature')
    if feature_name and config.feature_exists(feature_name):
        return feature_name

    # Method 3: Default feature
    if config.feature_exists("default"):
        return "default"

    # None return halts with error
    return None

def _apply_request_config(ctx: RequestContext, config: Dict[str, Any]) -> None:
    """Apply configuration to the current request."""

    # Override model if specified in config
    if config.get('model') and config['model'] != ctx.request.model_id:
        ctx.request.model_id = config['model']
        ctx.metadata['model_overridden'] = True

    # # Apply prompt template if specified
    # prompt_template = config.get('prompt_template')
    # if prompt_template:
    #     self._apply_prompt_template(ctx, prompt_template)

    # Store API key info for LLM execution (don't expose in logs)
    provider = config.get('provider', 'openai')
    api_key_field = f"{provider}_api_key"
    if config.get(api_key_field):
        ctx.metadata['api_key_configured'] = True
        # API key will be retrieved securely during LLM execution

async def _extend_pipeline_for_feature(
    ctx: RequestContext,
    config_resolver: ConfigResolver,
    limit_backend: Optional[BaseLimitBackend],
    feature_config: Dict[str, Any]
) -> None:
    """
    Dynamically add middleware to the pipeline based on feature configuration.

    This method creates and configures generic middleware instances based on
    the feature's configuration, then adds them to the pipeline for this request.
    """

    # Add budget enforcement middleware if budget constraints are configured
    limit_constraints = config_resolver.get_limit_constraints(feature_config['feature_name'])
    if limit_constraints:
        if limit_backend:
            ctx.add_middleware(limit_enforcement_middleware(
                constraints=limit_constraints,
                backend=limit_backend
            ))
        else:
            logger.warning("Budget constraints provided but no limit backend provided!")

    # Add Langfuse observability middleware if configured
    langfuse_config = config_resolver.get_langfuse_config(feature_config['feature_name'])
    if langfuse_config:
        # Import here to avoid circular imports
        from ..middleware.service.langfuse import langfuse_middleware

        ctx.add_middleware(langfuse_middleware(
            project=langfuse_config['project'],
            public_key=langfuse_config['public_key'],
            secret_key=langfuse_config['secret_key'],
            base_url=langfuse_config['base_url']
        ))

    # Add content guardrails middleware if configured
    # guardrails_config = config.get('guardrails', {})
    # content_safety = guardrails_config.get('content_safety', 'basic')

    # if content_safety != 'disabled':
    #     # Determine policies based on safety level
    #     policies = self._get_guardrail_policies(content_safety)

    #     if policies:
    #         guardrail_middleware = ContentGuardrailMiddleware(
    #             policies=policies,
    #             check_interval=3  # Check every 3 chunks during streaming
    #         )
    #         ctx.extend_pipeline(guardrail_middleware)

    # # Add response validation middleware if configured
    # if guardrails_config.get('response_validation', False):
    #     validator_middleware = ResponseValidatorMiddleware(
    #         min_quality_score=0.7,  # TODO: Make configurable
    #         min_length=10
    #     )
    #     ctx.extend_pipeline(validator_middleware)

    # Add basic logging middleware (always enabled for debugging)
    ctx.add_middleware(logging_middleware(
        log_requests=True,
        log_responses=True,
        feature_name=feature_config['feature_name']
    ))

def _get_guardrail_policies(safety_level: str) -> List[str]:
    """Get guardrail policies based on safety level."""
    if safety_level == 'disabled':
        return []
    elif safety_level == 'basic':
        return ['inappropriate']
    elif safety_level == 'strict':
        return ['inappropriate', 'pii']
    elif safety_level == 'permissive':
        return []  # Very minimal filtering for research use cases
    else:
        return ['inappropriate']  # Default

async def after_llm(ctx: RequestContext) -> RequestContext:
    """Log configuration metadata after LLM execution."""

    feature_info = ctx.state.get('feature', {})
    if feature_info:
        ctx.metadata['configuration_summary'] = {
            'feature_used': feature_info['name'],
            'feature_description': feature_info['description'],
            'owners': feature_info['owners'],
            'model_overridden': ctx.metadata.get('model_overridden', False)
        }

    return ctx
