"""
Authentication and rate limiting middleware.
"""

import time

from ..pipeline import Middleware, RequestContext


class AuthMiddleware(Middleware):
    """
    Basic authentication middleware.

    Extracts user information from client metadata and validates access.
    """

    def __init__(self, require_user_id: bool = True, **config):
        super().__init__(**config)
        self.require_user_id = require_user_id

    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Validate authentication and extract user info."""

        # Extract user information from OpenAI metadata field
        user_id = ctx.client_metadata.get('user_id')

        if self.require_user_id and not user_id:
            self.halt_with_error(ctx, "Authentication required: user_id missing", "auth_error")
            return ctx

        # Store auth info in context state for other middleware
        if user_id:
            ctx.state['auth'] = {
                'user_id': user_id,
                'organization': ctx.client_metadata.get('organization'),
                'authenticated_at': time.time()
            }

        return ctx


class RateLimitMiddleware(Middleware):
    """
    Simple rate limiting middleware.

    TODO: Implement actual rate limiting with Redis/in-memory store.
    Currently just validates that auth middleware has run.
    """

    def __init__(self, requests_per_minute: int = 100, **config):
        super().__init__(**config)
        self.rpm = requests_per_minute
        # TODO: Initialize rate limiting store (Redis, in-memory, etc.)

    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Check rate limits for the authenticated user."""

        # Get user info from auth middleware
        auth_info = ctx.state.get('auth')
        if not auth_info:
            self.halt_with_error(ctx, "Rate limiting requires authentication", "auth_error")
            return ctx

        user_id = auth_info['user_id']

        # TODO: Implement actual rate limiting logic
        # For now, just log the attempt
        ctx.metadata['rate_limit_check'] = {
            'user_id': user_id,
            'limit': self.rpm,
            'checked_at': time.time()
        }

        # Example of how rate limiting would work:
        # current_count = await self.rate_limiter.get_count(user_id)
        # if current_count >= self.rpm:
        #     self.halt_with_error(ctx, "Rate limit exceeded", "rate_limit_error")
        #     return ctx
        #
        # await self.rate_limiter.increment(user_id)

        return ctx
