"""
Built-in middleware implementations for common use cases.

This package provides ready-to-use middleware for:
- Authentication and rate limiting
- Content guardrails and validation
- Logging and observability
- Request routing and provider selection
- Error handling and retries
"""

# from .auth import AuthMiddleware, RateLimitMiddleware
# from .guardrails import ContentGuardrailMiddleware, ResponseValidatorMiddleware
from .logging import logging_middleware, observability_middleware
from .retry import retry_middleware
# from .routing import RoutingMiddleware
from .limit import limit_enforcement_middleware, BudgetLimit, RateLimit, Constraint, InMemoryLimitBackend, RedisClusterLimitBackend, RedisLimitBackend

__all__ = [
    # Authentication & Rate Limiting
    # 'AuthMiddleware',
    # 'RateLimitMiddleware',

    # Budget & Rate Limiting
    'limit_enforcement_middleware',
    'BudgetLimit',
    'RateLimit',
    'Constraint',
    'InMemoryLimitBackend',
    'RedisLimitBackend',
    'RedisClusterLimitBackend',

    # Content Safety
    # 'ContentGuardrailMiddleware',
    # 'ResponseValidatorMiddleware',

    # Observability
    'logging_middleware',
    'observability_middleware',

    # Error Handling
    'retry_middleware',

    # Routing
    # 'RoutingMiddleware',
]
