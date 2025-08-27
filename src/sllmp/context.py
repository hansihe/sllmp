from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .util.signal import Signal, Hooks

import pydantic
from any_llm.types.completion import CompletionParams, ChatCompletion, ChatCompletionChunk

from .error import PipelineError
from .util.stream import MultiChoiceDeltaCollector

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:8]}"

@dataclass
class Pipeline:
    """
    A set of signals which are invoked at various points in pipeline execution.

    ## Call phases
    Concists of 3 main phases. These phases always execute in order.
    - Setup - The `setup` signal, configures the pipeline.
    - LLM call - See below.
    - Response - The `response_complete` signal.

    ## LLM call state machine
    The most complex phase is the LLM call phase. It consists of a small
    state machine:
    1. `prepare`- Alter request before calling to provider. Can halt.
    2. LLM call
        a. `llm_call` - Hooks wrapping LLM call itself
        b. For streams:
            - `llm_call_stream_process`
            - `llm_call_stream_update`
        c. `llm_call_complete` - Invoked at the end of the LLM call.
    3. If error: Error handling with `error` callback

    In the regular case these execute in order.

    An exception would be retries, where an `error` handler can put
    the pipeline back into either prepare or LLM call state.

    Any callback within the LLM call state for instance may raise an error,
    at which point the state will transition immediately into the error state.
    """

    setup: Signal[[RequestContext], None] = field(default_factory=Signal)
    """
    Invoked once at the beginning of the pipeline.

    Primarily used for registering more middleware, not retryable.
    """

    pre: Signal[[RequestContext], None] = field(default_factory=Signal)
    """
    Invoked before LLM call.

    Can be used for:
    - Request validation
    - Authentication/authorization
    - Rate limiting
    - Request routing/modification
    - Prompt templating
    - etc
    """

    llm_call: Hooks[[RequestContext], RequestContext] = field(default_factory=Hooks)
    """
    Hooks for pre/post LLM call.
    Most useful for observability.
    """

    # Maybe `stream_start`, `stream_end`, `stream_error`?

    llm_call_stream_process: Signal[[RequestContext, ChatCompletionChunk], None] = field(default_factory=Signal)
    """
    Process individual chunks during streaming.

    Callbacks can perform per-chunk transformations.
    """

    llm_call_stream_update: Signal[[RequestContext], None] = field(default_factory=Signal)
    """
    Called periodically during streaming with accumulated content so far.

    Override this for real-time content monitoring (e.g., guardrails).
    The pipeline calls this every N chunks based on monitoring_interval config.
    """

    post: Signal[[RequestContext], None] = field(default_factory=Signal)
    """
    Called after LLM execution for response postprocessing.

    For streaming responses, this receives the complete assembled response.
    This is where you typically do:
    - Response validation
    - Logging/metrics
    - Cost tracking
    - Response modification
    """

    error: Signal[[RequestContext], None] = field(default_factory=Signal)
    """
    Invoked when the pipeline has reached an error state, and we would
    normally next invoke `response_complete` and return error to the client.

    Callbacks could potentially apply a retry policy and kick the pipeline
    back into a valid state.
    """

    response_complete: Signal[[RequestContext], None] = field(default_factory=Signal)
    """
    Invoked right before the response is returned to the client.
    When this callback is invoked the pipeline is finalized, and no
    pipeline state changes will be accepted.
    """

    def close(self):
        self.setup.close()
        self.pre.close()
        self.llm_call_stream_process.close()
        self.llm_call_stream_update.close()
        self.post.close()
        self.error.close()
        self.response_complete.close()

class NCompletionParams(CompletionParams):
    model_config = pydantic.ConfigDict(extra="allow")
    metadata: Dict[str, Any] = pydantic.Field()

class PipelineAction(Enum):
    """Actions that control pipeline flow."""
    CONTINUE = "continue"  # Keep processing through pipeline
    HALT = "halt"         # Stop pipeline and return current response
    RETRY = "retry"       # Retry current operation (TODO: implement retry logic)

class PipelineState(Enum):
    SETUP = "setup"
    PRE = "pre"
    LLM_CALL = "llm_call"
    POST = "post"
    ERROR = "error"
    COMPLETE = "complete"

@dataclass
class RequestContext:
    """
    Context object that flows through the pipeline carrying request state.

    This is the core data structure that middleware operates on. It contains
    the request/response data and provides a place for middleware to communicate
    through shared state.
    """
    # Core request/response data
    original_request: CompletionParams                 # Immutable original OpenAI request
    request: CompletionParams                          # Mutable current request state
    response: Optional[ChatCompletion] = None          # Successful response from LLM
    error: Optional[PipelineError] = None              # Pipeline error if occurred

    # Pipeline state
    pipeline: Pipeline = field(default_factory=Pipeline)
    last_success_pipeline_state: Optional[PipelineState] = None
    pipeline_state: PipelineState = PipelineState.SETUP
    next_pipeline_state: Optional[PipelineState] = None

    # Request identification and metadata
    request_id: str = field(default_factory=generate_request_id)
    client_metadata: Dict[str, Any] = field(default_factory=dict)  # From OpenAI metadata + headers

    # Shared state between middleware
    provider_keys: Dict[str, str] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)      # Inter-middleware communication
    metadata: Dict[str, Any] = field(default_factory=dict)   # Pipeline execution metadata
    errors: List[Exception] = field(default_factory=list)    # Accumulated errors

    # Stream characteristics (auto-detected)
    chunk_count: int = 0
    stream_collector: MultiChoiceDeltaCollector = field(default_factory=MultiChoiceDeltaCollector)

    # Response/Error state management
    @property
    def has_response(self) -> bool:
        """Check if context has a successful response."""
        return self.response is not None

    @property
    def has_error(self) -> bool:
        """Check if context has an error."""
        return self.error is not None

    @property
    def is_complete(self) -> bool:
        """Check if context has either a response or error."""
        return self.has_response or self.has_error

    @property
    def is_streaming(self) -> bool:
        if self.original_request.stream:
            return True
        return False

    def add_middleware(self, middleware: Callable[["RequestContext"]]):
        self.pipeline.setup.connect(middleware)

    def set_response(self, response: ChatCompletion) -> None:
        """Set successful response and clear any error."""
        self.response = response
        self.error = None

    def set_error(self, error: PipelineError) -> None:
        """Set error and clear any response."""
        self.error = error
        self.response = None

        # Automatically transition to error state when error is set
        if self.next_pipeline_state is None:
            self.next_pipeline_state = PipelineState.ERROR
