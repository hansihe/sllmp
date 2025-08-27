"""
Generic observability middleware including Langfuse integration.

This middleware is config-agnostic and receives observability parameters
through constructor injection.
"""

import time
from typing import Dict, Any, Optional

from ...pipeline import Middleware, RequestContext

class LangfuseMiddleware(Middleware):
    """
    Generic Langfuse observability middleware.

    This middleware integrates with Langfuse for LLM observability without
    any knowledge of configuration format. All parameters are injected.

    TODO: This is currently a stub implementation. In production, you would:
    1. Install the langfuse package: pip install langfuse
    2. Import and use the actual Langfuse client
    3. Handle async tracing properly
    4. Add error handling and retry logic
    """

    def __init__(self, project: str, secret_key: str, base_url: str = "https://cloud.langfuse.com", **kwargs):
        """
        Initialize Langfuse middleware.

        Args:
            project: Langfuse project name
            secret_key: Langfuse secret key
            base_url: Langfuse API base URL
        """
        super().__init__(**kwargs)
        self.project = project
        self.secret_key = secret_key
        self.base_url = base_url

        # TODO: Initialize actual Langfuse client
        # from langfuse import Langfuse
        # self.langfuse = Langfuse(
        #     secret_key=secret_key,
        #     public_key=public_key,  # Would need to be passed as parameter
        #     host=base_url
        # )

        # For now, use stub implementation
        self.langfuse = None
        print(f"🔍 Langfuse middleware initialized for project: {project}")

    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Start Langfuse trace for the request."""

        # Extract feature and user information
        feature_info = ctx.state.get('feature', {})
        feature_name = feature_info.get('name', 'unknown')
        user_id = ctx.client_metadata.get('user_id', 'anonymous')

        # Create trace metadata
        trace_metadata = {
            'feature': feature_name,
            'user_id': user_id,
            'model': ctx.request.get('model'),
            'is_streaming': ctx.is_streaming,
            'request_id': ctx.request_id
        }

        # TODO: Create actual Langfuse trace
        # trace = self.langfuse.trace(
        #     name=f"{feature_name}_request",
        #     user_id=user_id,
        #     metadata=trace_metadata
        # )

        # Store trace info in context for other middleware
        ctx.state['langfuse'] = {
            'trace_id': f"trace_{ctx.request_id}",  # Would be trace.id in real implementation
            'project': self.project,
            'trace_metadata': trace_metadata,
            'start_time': time.time()
        }

        # Log for debugging (remove in production)
        print(f"📊 [Langfuse] Started trace for feature '{feature_name}' (user: {user_id})")

        return ctx

    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        """Finalize Langfuse trace with response data."""

        langfuse_info = ctx.state.get('langfuse')
        if not langfuse_info:
            return ctx

        try:
            # Calculate duration
            duration = time.time() - langfuse_info['start_time']

            # Extract response information
            response_metadata = self._extract_response_metadata(ctx.response)

            # TODO: Update actual Langfuse trace
            # trace = self.langfuse.get_trace(langfuse_info['trace_id'])
            # trace.generation(
            #     name="llm_completion",
            #     model=ctx.request.get('model'),
            #     input=self._sanitize_input(ctx.request),
            #     output=self._sanitize_output(ctx.response),
            #     metadata={
            #         **response_metadata,
            #         'duration_ms': duration * 1000,
            #         'chunk_count': ctx.chunk_count if ctx.is_streaming else None
            #     },
            #     usage=response_metadata.get('usage')
            # )

            # Log completion (remove in production)
            feature_name = ctx.state.get('feature', {}).get('name', 'unknown')
            print(f"📊 [Langfuse] Completed trace for '{feature_name}' - Duration: {duration:.2f}s")

            # Store observability data in context
            ctx.metadata['langfuse_tracking'] = {
                'trace_id': langfuse_info['trace_id'],
                'project': self.project,
                'duration_ms': duration * 1000,
                'tokens_used': response_metadata.get('total_tokens', 0),
                'model': ctx.request.get('model')
            }

        except Exception as e:
            # Don't halt request on observability errors
            ctx.metadata['langfuse_error'] = f"Failed to track in Langfuse: {e}"
            print(f"⚠️  [Langfuse] Tracking error: {e}")

        return ctx

    async def on_stream_start(self, ctx: RequestContext) -> RequestContext:
        """Handle streaming start event."""
        langfuse_info = ctx.state.get('langfuse')
        if langfuse_info:
            print(f"📊 [Langfuse] Streaming started for trace {langfuse_info['trace_id']}")
        return ctx

    async def on_stream_end(self, ctx: RequestContext) -> RequestContext:
        """Handle streaming completion event."""
        langfuse_info = ctx.state.get('langfuse')
        if langfuse_info:
            print(f"📊 [Langfuse] Streaming completed for trace {langfuse_info['trace_id']} - {ctx.chunk_count} chunks")
        return ctx

    async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        """Handle error event in Langfuse."""
        langfuse_info = ctx.state.get('langfuse')
        if langfuse_info:
            # TODO: Update trace with error information
            # trace = self.langfuse.get_trace(langfuse_info['trace_id'])
            # trace.update(
            #     metadata={
            #         **langfuse_info['trace_metadata'],
            #         'error': str(error),
            #         'error_type': type(error).__name__
            #     }
            # )

            print(f"❌ [Langfuse] Error logged for trace {langfuse_info['trace_id']}: {error}")

        return ctx

    def _extract_response_metadata(self, response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant metadata from LLM response."""
        if not response:
            return {}

        metadata = {}

        # Extract usage information
        usage = response.get('usage', {})
        if usage:
            metadata['usage'] = usage
            metadata['prompt_tokens'] = usage.get('prompt_tokens', 0)
            metadata['completion_tokens'] = usage.get('completion_tokens', 0)
            metadata['total_tokens'] = usage.get('total_tokens', 0)

        # Extract model and finish reason
        metadata['model'] = response.get('model')

        choices = response.get('choices', [])
        if choices:
            first_choice = choices[0]
            metadata['finish_reason'] = first_choice.get('finish_reason')

        return metadata

    def _sanitize_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize input data for Langfuse logging.

        Remove sensitive information while preserving useful data for analysis.
        """
        sanitized = request.copy()

        # Remove or mask sensitive fields
        sensitive_fields = ['api_key', 'authorization', 'metadata']
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = '[REDACTED]'

        # Optionally truncate very long content
        messages = sanitized.get('messages', [])
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str) and len(content) > 1000:
                message['content'] = content[:1000] + '... [TRUNCATED]'

        return sanitized

    def _sanitize_output(self, response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Sanitize output data for Langfuse logging.

        Remove sensitive information while preserving useful data for analysis.
        """
        if not response:
            return None

        sanitized = response.copy()

        # Keep important fields for analysis
        important_fields = ['id', 'object', 'model', 'choices', 'usage']
        sanitized = {k: v for k, v in sanitized.items() if k in important_fields}

        # Optionally truncate very long responses
        choices = sanitized.get('choices', [])
        for choice in choices:
            message = choice.get('message', {})
            content = message.get('content', '')
            if isinstance(content, str) and len(content) > 2000:
                message['content'] = content[:2000] + '... [TRUNCATED]'

        return sanitized
