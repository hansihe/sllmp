"""
Simplified budget enforcement middleware.

This middleware enforces budget constraints using a simple check-before,
increment-after pattern without reservations.
"""

from typing import List, Dict, Any, Protocol, Optional
from ...context import RequestContext
from ...error import PipelineError
from ... import logger
from ...pricing import calculate_usage_pricing

from dataclasses import dataclass, field
import time

@dataclass(frozen=True)
class LimitError(PipelineError):
    """Error due to budget or rate limit exceeded."""
    limit_type: str  # "budget_limit_exceeded" or "rate_limit_exceeded"
    constraint_description: str
    current_usage: str
    limit_value: str
    error_type: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'error_type', self.limit_type)

    def _extra_fields(self) -> Dict[str, Any]:
        return {
            "constraint": self.constraint_description,
            "current_usage": self.current_usage,
            "limit": self.limit_value
        }

@dataclass
class BudgetLimit:
    limit: float                  # Budget limit in USD
    window: str                   # Time window (1h, 1d, 7d, 30d)

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("Budget limit must be positive")

        valid_windows = ["1h", "1d", "7d", "30d"]
        if self.window not in valid_windows:
            raise ValueError(f"Invalid window: {self.window}. Must be one of {valid_windows}")

@dataclass
class RateLimit:
    per_minute: int

    def __post_init__(self) -> None:
        if self.per_minute <= 0:
            raise ValueError("Rate limit must be positive")

@dataclass
class Constraint:
    """
    Budget constraint specification.

    Defines a multi-dimensional budget constraint that can combine
    different dimensions like feature, user_id, organization, etc.
    """
    name: str                           # Human-readable constraint name
    dimensions: List[str]               # Constraint dimensions (feature, user_id, etc.)
    budget_limit: Optional[BudgetLimit]
    rate_limit: Optional[RateLimit]
    description: str                    # Human-readable description

    def __post_init__(self) -> None:
        """Validate constraint after creation."""

        valid_dimensions = ["feature", "user_id", "organization", "team"]
        invalid_dims = [d for d in self.dimensions if d not in valid_dimensions]
        if invalid_dims:
            raise ValueError(f"Invalid dimensions: {invalid_dims}. Must be from {valid_dimensions}")

        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("Duplicate dimensions not allowed")

        if self.budget_limit is None and self.rate_limit is None:
            raise ValueError("Either budget or rate limit must be specified")


class BaseLimitBackend(Protocol):
    """
    Simplified protocol for limit tracking backends.

    Just get usage and increment usage - no reservations.
    """

    async def get_usage(self, constraint_key: str, window: str) -> float:
        """Get current usage for a constraint within the time window."""
        ...

    async def increment_usage(self, constraint_key: str, amount: float, window: str) -> None:
        """Increment usage by the specified amount."""
        ...

    async def get_rate_usage(self, constraint_key: str) -> int:
        """Get current rate limit usage (requests in current minute)."""
        ...

    async def increment_rate_usage(self, constraint_key: str) -> None:
        """Increment rate limit usage by 1 request."""
        ...


class InMemoryLimitBackend(BaseLimitBackend):
    """
    Simple in-memory limit backend for development/testing.

    This is not suitable for production with multiple instances.
    Use Redis backend for production deployments.
    """

    def __init__(self) -> None:
        self._usage: Dict[str, float] = {}
        self._rate_usage: Dict[str, Dict[int, int]] = {}  # {key: {minute_timestamp: count}}

    async def get_usage(self, constraint_key: str, window: str) -> float:
        """Get current usage (simplified - doesn't handle time windows properly)."""
        # TODO: Implement proper time window handling with expiring keys
        return self._usage.get(constraint_key, 0.0)

    async def increment_usage(self, constraint_key: str, amount: float, window: str) -> None:
        """Increment usage by the specified amount."""
        self._usage[constraint_key] = self._usage.get(constraint_key, 0.0) + amount

    async def get_rate_usage(self, constraint_key: str) -> int:
        """Get current rate limit usage (requests in current minute)."""
        current_minute = int(time.time() // 60)

        if constraint_key not in self._rate_usage:
            self._rate_usage[constraint_key] = {}

        # Clean up old entries (keep only current minute)
        key_usage = self._rate_usage[constraint_key]
        self._rate_usage[constraint_key] = {
            minute: count for minute, count in key_usage.items()
            if minute >= current_minute
        }

        current_usage = self._rate_usage[constraint_key].get(current_minute, 0)
        return current_usage if current_usage is not None else 0

    async def increment_rate_usage(self, constraint_key: str) -> None:
        """Increment rate limit usage by 1 request."""
        current_minute = int(time.time() // 60)

        if constraint_key not in self._rate_usage:
            self._rate_usage[constraint_key] = {}

        self._rate_usage[constraint_key][current_minute] = (
            self._rate_usage[constraint_key].get(current_minute, 0) + 1
        )

def limit_enforcement_middleware(
    constraints: List[Constraint],
    backend: BaseLimitBackend,
    **kwargs,
):
    """
    Simplified budget enforcement middleware.

    Uses a simple check-before, increment-after pattern without reservations.
    This avoids complexity while still providing effective rate limiting.
    """

    def setup(ctx: RequestContext):
        ctx.pipeline.pre.connect(_check_limits)
        ctx.pipeline.post.connect(_update_usage)

    async def _check_limits(ctx: RequestContext) -> None:
        """Check all budget constraints before LLM execution."""

        if not constraints:
            return

        # Check each constraint
        for constraint in constraints:
            constraint_key = _build_constraint_key(constraint, ctx)

            # Check budget limits
            if constraint.budget_limit:
                estimated_cost = _estimate_request_cost(ctx)
                if estimated_cost > 0:
                    try:
                        current_usage = await backend.get_usage(
                            constraint_key, constraint.budget_limit.window
                        )

                        if (current_usage + estimated_cost) > constraint.budget_limit.limit:
                            budget_error = LimitError(
                                message=f"Budget limit exceeded: {constraint.description}. "
                                       f"Current usage: ${current_usage:.4f}, "
                                       f"Limit: ${constraint.budget_limit.limit:.4f}, "
                                       f"Estimated cost: ${estimated_cost:.4f}",
                                request_id=ctx.request_id,
                                limit_type="budget_limit_exceeded",
                                constraint_description=constraint.description,
                                current_usage=f"${current_usage:.4f}",
                                limit_value=f"${constraint.budget_limit.limit:.4f}"
                            )
                            ctx.set_error(budget_error)
                            return
                    except Exception:
                        # If we can't check usage, err on the side of caution and allow
                        logger.exception("Unable to check budget usage, allowing request")
                        pass

            # Check rate limits
            if constraint.rate_limit:
                current_rate_usage = await backend.get_rate_usage(constraint_key)

                if current_rate_usage >= constraint.rate_limit.per_minute:
                    rate_error = LimitError(
                        message=f"Rate limit exceeded: {constraint.description}. "
                               f"Current usage: {current_rate_usage} requests/minute, "
                               f"Limit: {constraint.rate_limit.per_minute} requests/minute",
                        request_id=ctx.request_id,
                        limit_type="rate_limit_exceeded",
                        constraint_description=constraint.description,
                        current_usage=f"{current_rate_usage} requests/minute",
                        limit_value=f"{constraint.rate_limit.per_minute} requests/minute"
                    )
                    ctx.set_error(rate_error)
                    return

        # Store estimated cost for later use
        ctx.state['limit_estimated_cost'] = _estimate_request_cost(ctx)

        # Increment rate limits immediately (before LLM call)
        for constraint in constraints:
            if constraint.rate_limit:
                constraint_key = _build_constraint_key(constraint, ctx)
                await backend.increment_rate_usage(constraint_key)

    async def _update_usage(ctx: RequestContext) -> None:
        """Increment budget usage with actual costs."""

        if not constraints:
            return

        try:
            # Calculate actual cost from response
            usage = ctx.response.usage if ctx.response is not None else None
            actual_cost = 0.0
            if usage is not None:
                actual_cost = calculate_usage_pricing(ctx.request.model_id, usage)

            estimated_cost = ctx.state.get('limit_estimated_cost', 0.0)

            if actual_cost > 0:
                # Increment usage for each budget constraint
                for constraint in constraints:
                    if constraint.budget_limit:
                        constraint_key = _build_constraint_key(constraint, ctx)
                        await backend.increment_usage(
                            constraint_key, actual_cost, constraint.budget_limit.window
                        )

            # Store tracking information for observability
            ctx.metadata['budget_tracking'] = {
                'estimated_cost': estimated_cost,
                'actual_cost': actual_cost,
                'constraints_checked': len(constraints)
            }

        except Exception as e:
            # Log error but don't halt (response already generated)
            logger.exception("Failed to update budget tracking")
            ctx.metadata['budget_error'] = f"Failed to update budget tracking: {e}"

    def _build_constraint_key(constraint: Constraint, ctx: RequestContext) -> str:
        """
        Build unique key for budget constraint tracking.

        The key combines all dimensions of the constraint to create a unique
        identifier for tracking usage.
        """
        key_parts = []

        for dimension in constraint.dimensions:
            if dimension == 'feature':
                feature_name = ctx.state.get('feature', {}).get('name', 'unknown')
                key_parts.append(f"feature:{feature_name}")
            elif dimension == 'user_id':
                user_id = ctx.client_metadata.get('user_id', 'anonymous')
                key_parts.append(f"user:{user_id}")
            elif dimension == 'organization':
                org = ctx.client_metadata.get('organization', 'unknown')
                key_parts.append(f"org:{org}")
            elif dimension == 'team':
                team = ctx.client_metadata.get('team', 'unknown')
                key_parts.append(f"team:{team}")
            else:
                # Handle meta: prefixed dimensions
                if dimension.startswith("meta:"):
                    meta_dimension = dimension[5:]
                    value = ctx.client_metadata.get(meta_dimension, 'unknown')
                    key_parts.append(f"meta:{meta_dimension}:{value}")
                else:
                    raise ValueError(f"Unknown dimension: {dimension}")

        # Sort key parts for consistency
        return "|".join(sorted(key_parts))

    def _estimate_request_cost(ctx: RequestContext) -> float:
        """
        Estimate cost based on request characteristics.
        """
        # Handle both CompletionParams object and dict for backward compatibility
        if hasattr(ctx.request, 'model_id'):
            # New CompletionParams object
            model = getattr(ctx.request, 'model_id', 'openai:gpt-3.5-turbo').lower()
            messages = getattr(ctx.request, 'messages', [])
            max_tokens = getattr(ctx.request, 'max_tokens', 150)
        else:
            # Fallback to dict access
            model = (ctx.request.model_id or 'openai:gpt-3.5-turbo').lower()
            messages = ctx.request.messages or []
            max_tokens = ctx.request.max_tokens or 150

        # Remove provider prefix if present
        if ':' in model:
            model = model.split(':', 1)[1]

        # Simple cost estimation per 1K tokens
        cost_per_1k_tokens = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.030,
            'gpt-4-turbo': 0.010,
            'claude-3-haiku': 0.0008,
            'claude-3-sonnet': 0.003,
            'claude-3-opus': 0.015
        }

        # Get cost rate for model (with fallback)
        rate = cost_per_1k_tokens.get(model, 0.002)

        # Estimate token count from message content
        estimated_tokens = _estimate_token_count(messages)

        # Add buffer for response tokens (rough estimate)
        if not isinstance(max_tokens, (int, float)):
            max_tokens = 150
        estimated_total_tokens = estimated_tokens + int(max_tokens)

        return (estimated_total_tokens / 1000) * rate

    def _estimate_token_count(messages: List[Dict[str, Any]]) -> int:
        """
        Rough token count estimation.
        """
        total_chars = 0

        for message in messages:
            content = message.get('content', '')

            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if item.get('type') == 'text':
                        total_chars += len(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        # Images cost more tokens (rough approximation)
                        total_chars += 1000  # Approximate cost for image processing

        # Rough conversion: ~4 characters per token for English text
        return max(1, int(total_chars / 4))

    return setup
