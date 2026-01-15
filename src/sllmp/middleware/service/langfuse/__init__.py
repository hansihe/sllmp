try:
    import langfuse
    from langfuse.api import NotFoundError
except ImportError:
    _has_langfuse = False
    langfuse = None
    NotFoundError = Exception
else:
    _has_langfuse = True

import os
from typing import Dict, Optional, cast

from sllmp.error import MiddlewareError
from sllmp.pipeline import RequestContext

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

# from opentelemetry.trace import use_span
# from opentelemetry.trace.span import INVALID_SPAN
from opentelemetry import context as ctx_api, trace
from opentelemetry.trace import set_span_in_context


if _has_langfuse:
    import langfuse
    from langfuse import Langfuse
    from langfuse.model import PromptClient
    from langfuse import LangfuseOtelSpanAttributes

    # Environment variable constants
    LANGFUSE_RELEASE = "LANGFUSE_RELEASE"
    LANGFUSE_TRACING_ENVIRONMENT = "LANGFUSE_ENVIRONMENT"

    from .util import process_output, extract_chat_prompt

    LANGFUSE_CLIENTS: Dict[str, Langfuse] = {}

    def _make_langfuse_client(public_key, secret_key, base_url):
        environment = os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)
        release = os.environ.get(LANGFUSE_RELEASE)

        resource_attributes = {
            LangfuseOtelSpanAttributes.ENVIRONMENT: environment,
            LangfuseOtelSpanAttributes.RELEASE: release,
        }

        resource = Resource.create(
            {k: v for k, v in resource_attributes.items() if v is not None}
        )

        # Because OTEL tracing may also be configured on a global level,
        # we create a separate TracerProvider for LangFuse specifically.
        provider = TracerProvider(
            resource=resource,
            sampler=None,
        )

        return Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url,
            tracer_provider=provider,
        )

    def langfuse_middleware(
        public_key: str,
        secret_key: str,
        base_url: str = "https://cloud.langfuse.com",
        default_prompt_label: str = "latest",
    ):
        def setup(ctx: RequestContext):
            """
            Configures the Langfuse for the request.

            Supports prompt management and observability.
            """

            client = LANGFUSE_CLIENTS.get(public_key, None)
            if client is None:
                client = _make_langfuse_client(
                    public_key=public_key, secret_key=secret_key, base_url=base_url
                )
                LANGFUSE_CLIENTS[public_key] = client

            ctx.state["langfuse"] = {
                "client": client,
                "prompt_label": default_prompt_label,
                "used_prompt_client": None,
                "used_prompt_variables": {},
            }

            # Propmpt management
            ctx.pipeline.pre.connect(_prompt_management_pre_llm)

            # Observability
            _observability_setup(ctx, client)

        return setup

    def _prompt_management_pre_llm(ctx: RequestContext):
        prompt_id = (ctx.request.__pydantic_extra__ or {}).get("prompt_id", None)
        prompt_variables = (ctx.request.__pydantic_extra__ or {}).get(
            "prompt_variables", {}
        )

        if prompt_id is not None:
            langfuse_state = ctx.state["langfuse"]
            client = cast(Langfuse, langfuse_state["client"])

            prompt_label = langfuse_state["prompt_label"]

            try:
                prompt_client = client.get_prompt(prompt_id, label=prompt_label)
            except NotFoundError:
                raise MiddlewareError(
                    message=f"Langfuse prompt '{prompt_id}' not found with label '{prompt_label}'",
                    request_id=ctx.request_id,
                    middleware_name="langfuse_prompt",
                )

            prompt = prompt_client.compile(**prompt_variables)

            if isinstance(prompt, str):
                ctx.request.messages.insert(0, {"role": "system", "content": prompt})
            else:
                ctx.request.messages = prompt + ctx.request.messages

            langfuse_state["used_prompt_client"] = prompt_client
            langfuse_state["used_prompt_variables"] = prompt_variables

    def _observability_setup(ctx: RequestContext, client: Langfuse):
        langfuse_state = ctx.state["langfuse"]

        # Create new root span for LLM observability without infra observability parent
        token = ctx_api.attach(
            set_span_in_context(trace.INVALID_SPAN, ctx_api.Context())
        )
        root_span = client.start_span(
            name="chat-completion",
            input=extract_chat_prompt(ctx.request),
            metadata=ctx.client_metadata,
        )
        ctx_api.detach(token)

        langfuse_state["root_span"] = root_span

        generation = None

        def observability_pre_llm(ctx: RequestContext):
            used_prompt_client = cast(
                Optional[PromptClient], langfuse_state.get("used_prompt_client", None)
            )

            nonlocal generation
            generation = root_span.start_observation(
                name="llm-request",
                as_type="generation",
                input=extract_chat_prompt(ctx.request),
                metadata=ctx.client_metadata,
                model=ctx.request.model_id,
                model_parameters=ctx.request.model_dump(
                    exclude_none=True,
                    include={
                        "temperature",
                        "top_p",
                        "max_tokens",
                        "stream",
                        "n",
                        "stop",
                        "presence_penalty",
                        "frequency_penalty",
                        "seed",
                        "parallel_tool_calls",
                        "logprobs",
                        "top_logprobs",
                        "logit_bias",
                        "max_completion_tokens",
                        "reasoning_effort",
                    },
                ),
                prompt=used_prompt_client,
            )

        def observability_post_llm(ctx: RequestContext):
            nonlocal generation
            if generation is not None:
                if ctx.has_error:
                    generation.update(
                        level="ERROR", status_message=f"error: {ctx.error}"
                    )
                else:
                    response = ctx.response
                    assert response is not None

                    generation.update(
                        output=process_output(response),
                        # TODO map fields?
                        usage_details=response.usage.model_dump()
                        if response.usage is not None
                        else None,
                    )
                generation.end()

        ctx.pipeline.llm_call.add(
            observability_pre_llm,
            observability_post_llm,
        )

        @ctx.pipeline.response_complete.connect
        def _observability_response_complete(ctx: RequestContext):
            if ctx.has_error:
                root_span.update(level="ERROR", status_message=f"error: {ctx.error}")
            else:
                response = ctx.response
                assert response is not None

                root_span.update(output=process_output(response))
            root_span.end()
