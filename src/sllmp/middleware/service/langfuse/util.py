from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    CompletionParams,
)
from typing import Dict, Any

from . import _has_langfuse

if _has_langfuse:
    import langfuse.media

    def process_output(response: ChatCompletion):
        completion = None
        if len(response.choices) == 1:
            completion = _extract_chat_response(response.choices[0].message)
        elif len(response.choices) > 1:
            completion = [
                _extract_chat_response(choice.message) for choice in response.choices
            ]
        return completion

    def _extract_chat_response(kwargs: ChatCompletionMessage) -> Any:
        """Extracts the llm output from the response."""
        response: Dict[str, Any] = {
            "role": kwargs.role,
        }

        audio = None

        if kwargs.function_call is not None:
            response.update({"function_call": kwargs.function_call.model_dump()})

        if kwargs.tool_calls is not None:
            response.update(
                {
                    "tool_calls": [
                        tool_call.model_dump() for tool_call in kwargs.tool_calls
                    ]
                }
            )

        if kwargs.audio is not None:
            audio = kwargs.audio.__dict__

            if "data" in audio and audio["data"] is not None:
                base64_data_uri = f"data:audio/{audio.get('format', 'wav')};base64,{audio.get('data', None)}"
                audio["data"] = langfuse.media.LangfuseMedia(
                    base64_data_uri=base64_data_uri
                )

        response.update(
            {
                "content": kwargs.content,
            }
        )

        if audio is not None:
            response.update({"audio": audio})

        return response

    def extract_chat_prompt(params: CompletionParams) -> Any:
        """Extracts the user input from prompts. Returns an array of messages or dict with messages and functions"""
        prompt = {}

        if params.tools is not None:
            prompt.update({"tools": params.tools})

        if prompt:
            # uf user provided functions, we need to send these together with messages to langfuse
            prompt.update(
                {
                    "messages": [
                        _process_message(message) for message in params.messages
                    ],
                }
            )
            return prompt
        else:
            # vanilla case, only send messages in openai format to langfuse
            return [_process_message(message) for message in params.messages]

    def _process_message(message: Any) -> Any:
        if not isinstance(message, dict):
            return message

        processed_message = {**message}

        content = processed_message.get("content", None)
        if not isinstance(content, list):
            return processed_message

        processed_content = []

        for content_part in content:
            if content_part.get("type") == "input_audio":
                audio_base64 = content_part.get("input_audio", {}).get("data", None)
                format = content_part.get("input_audio", {}).get("format", "wav")

                if audio_base64 is not None:
                    base64_data_uri = f"data:audio/{format};base64,{audio_base64}"

                    processed_content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": langfuse.media.LangfuseMedia(
                                    base64_data_uri=base64_data_uri
                                ),
                                "format": format,
                            },
                        }
                    )
            else:
                processed_content.append(content_part)

        processed_message["content"] = processed_content

        return processed_message
