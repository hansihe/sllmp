try:
    import langfuse
except ImportError:
    _has_langfuse = False
else:
    _has_langfuse = True

from langfuse.api import NotFoundError
from typing import Dict, Optional, cast

from sllmp.error import MiddlewareError
from sllmp.pipeline import RequestContext

from any_llm.types.completion import ChatCompletionMessage

if _has_langfuse:
    import langfuse.model

    from .util import process_output, extract_chat_prompt

    LANGFUSE_CLIENTS: Dict[str, langfuse.Langfuse] = {}

    def langfuse_middleware(public_key: str, secret_key: str, base_url: str = "https://cloud.langfuse.com"):
        def setup(ctx: RequestContext):
            """
            Configures the Langfuse for the request.

            Supports prompt management and observability.
            """

            client = LANGFUSE_CLIENTS.get(public_key, None)
            if client is None:
                client = langfuse.Langfuse(
                    secret_key=secret_key,
                    public_key=public_key,
                    host=base_url
                )
                LANGFUSE_CLIENTS[public_key] = client

            ctx.state['langfuse'] = {
                'client': client
            }

            # Propmpt management
            ctx.pipeline.pre.connect(_prompt_management_pre_llm)

            # Observability
            _observability_setup(ctx, client)

        return setup

    def _prompt_management_pre_llm(ctx: RequestContext):
        prompt_id = (ctx.request.__pydantic_extra__ or {}).get('prompt_id', None)
        prompt_variables = (ctx.request.__pydantic_extra__ or {}).get('prompt_variables', {})

        if prompt_id is not None:
            langfuse_state = ctx.state['langfuse']
            client = cast(langfuse.Langfuse, langfuse_state['client'])

            try:
                prompt_client = client.get_prompt(prompt_id)
            except NotFoundError:
                ctx.set_error(MiddlewareError(
                    message=f"Langfuse prompt {prompt_id} not found",
                    request_id=ctx.request_id,
                    middleware_name="langfuse_prompt",
                ))
                return

            prompt = prompt_client.compile(**prompt_variables)

            if isinstance(prompt, str):
                ctx.request.messages.insert(0, {
                    "role": "system",
                    "content": prompt
                })
            else:
                prompt.extend(ctx.request.messages)
                ctx.request.messages = prompt

            langfuse_state["used_prompt_client"] = prompt_client

    def _observability_setup(ctx: RequestContext, client: langfuse.Langfuse):
        langfuse_state = ctx.state['langfuse']
        root_span = client.start_span(name="chat-completion")
        langfuse_state['root_span'] = root_span

        generation = None

        def observability_pre_llm(ctx: RequestContext):
            used_prompt_client = cast(Optional[langfuse.model.PromptClient], langfuse_state['used_prompt_client'])

            nonlocal generation
            generation = root_span.start_generation(
                name="llm-response",
                input=extract_chat_prompt(ctx.request),
                metadata=ctx.client_metadata,
                prompt=used_prompt_client,
            )

        def observability_post_llm(ctx: RequestContext):
            if generation is not None:
                if ctx.has_error:
                    generation.update(
                        level="ERROR",
                        status_message=f"error: {ctx.error}"
                    )
                else:
                    response = ctx.response
                    assert response is not None

                    generation.update(
                        output=process_output(response),
                        # TODO map fields?
                        usage_details=response.usage.model_dump() if response.usage is not None else None,
                    )
                generation.end()

        ctx.pipeline.llm_call.add(
            observability_pre_llm,
            observability_post_llm,
        )

        @ctx.pipeline.response_complete.connect
        def _observability_response_complete(ctx: RequestContext):
            if ctx.has_error:
                root_span.update(
                    level="ERROR",
                    status_message=f"error: {ctx.error}"
                )
            else:
                response = ctx.response
                assert response is not None

                root_span.update(
                    output=process_output(response)
                )
            root_span.end()
