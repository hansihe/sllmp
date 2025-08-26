import time
from ...pipeline import Middleware, RequestContext

class MetricsMiddleware(Middleware):
    """
    Generic metrics collection middleware.

    Collects and emits metrics about LLM usage without knowledge of
    configuration format. Metrics backend is injected via constructor.

    TODO: Implement actual metrics backend integration (Prometheus, DataDog, etc.)
    """

    def __init__(self, metrics_backend=None, emit_detailed_metrics: bool = True, **kwargs):
        """
        Initialize metrics middleware.

        Args:
            metrics_backend: Backend for metrics collection (TODO: define interface)
            emit_detailed_metrics: Whether to emit detailed per-request metrics
        """
        super().__init__(**kwargs)
        self.metrics_backend = metrics_backend
        self.emit_detailed_metrics = emit_detailed_metrics

    async def before_llm(self, ctx: RequestContext) -> RequestContext:
        """Record request start metrics."""

        feature_info = ctx.state.get('feature', {})
        feature_name = feature_info.get('name', 'unknown')
        model = ctx.request.get('model', 'unknown')

        # TODO: Emit actual metrics
        # self.metrics_backend.increment('llm_requests_total', {
        #     'feature': feature_name,
        #     'model': model,
        #     'streaming': ctx.is_streaming
        # })

        # Store timing info
        ctx.metadata['metrics_start_time'] = time.time()

        # Log for debugging
        print(f"📈 [Metrics] Request started - Feature: {feature_name}, Model: {model}")

        return ctx

    async def after_llm(self, ctx: RequestContext) -> RequestContext:
        """Record request completion metrics."""

        start_time = ctx.metadata.get('metrics_start_time')
        if not start_time:
            return ctx

        duration = time.time() - start_time
        feature_info = ctx.state.get('feature', {})
        feature_name = feature_info.get('name', 'unknown')
        model = ctx.request.get('model', 'unknown')

        # Extract metrics from response
        usage = {}
        if ctx.response:
            usage = ctx.response.get('usage', {})

        # TODO: Emit actual metrics
        # self.metrics_backend.histogram('llm_request_duration_seconds', duration, {
        #     'feature': feature_name,
        #     'model': model,
        #     'success': str(ctx.action.value == 'continue')
        # })
        #
        # if usage:
        #     self.metrics_backend.histogram('llm_tokens_total', usage.get('total_tokens', 0), {
        #         'feature': feature_name,
        #         'model': model,
        #         'token_type': 'total'
        #     })

        # Log completion
        total_tokens = usage.get('total_tokens', 0)
        print(f"📈 [Metrics] Request completed - Feature: {feature_name}, Duration: {duration:.2f}s, Tokens: {total_tokens}")

        # Store metrics data in context
        ctx.metadata['request_metrics'] = {
            'duration_seconds': duration,
            'total_tokens': total_tokens,
            'feature': feature_name,
            'model': model,
            'success': ctx.action.value == 'continue'
        }

        return ctx

    async def on_error(self, ctx: RequestContext, error: Exception) -> RequestContext:
        """Record error metrics."""

        feature_info = ctx.state.get('feature', {})
        feature_name = feature_info.get('name', 'unknown')

        # TODO: Emit error metrics
        # self.metrics_backend.increment('llm_errors_total', {
        #     'feature': feature_name,
        #     'error_type': type(error).__name__
        # })

        print(f"📈 [Metrics] Error recorded - Feature: {feature_name}, Error: {type(error).__name__}")

        return ctx
