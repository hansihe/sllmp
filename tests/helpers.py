"""
Shared test helper functions for the SLLMP test suite.

This module provides factory functions for creating mock LLM responses
that can be imported by both conftest.py and individual test files.
"""

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk


def create_chat_completion(
    model: str = "openai:gpt-3.5-turbo",
    content: str = "Hello! This is a test response.",
    completion_id: str = "chatcmpl-test123",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ChatCompletion:
    """Factory function to create a ChatCompletion response."""
    return ChatCompletion(
        id=completion_id,
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def create_stream_chunks(
    model: str = "openai:gpt-4",
    content_parts: list[str] | None = None,
    completion_id: str = "chatcmpl-test123",
) -> list[ChatCompletionChunk]:
    """Factory function to create a list of ChatCompletionChunk objects for streaming."""
    if content_parts is None:
        content_parts = ["Hello", " test"]

    chunks = [
        # Initial chunk with role
        ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=1234567890,
            model=model,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        )
    ]

    # Content chunks
    for part in content_parts:
        chunks.append(
            ChatCompletionChunk(
                id=completion_id,
                object="chat.completion.chunk",
                created=1234567890,
                model=model,
                choices=[
                    {"index": 0, "delta": {"content": part}, "finish_reason": None}
                ],
            )
        )

    # Final chunk
    chunks.append(
        ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=1234567890,
            model=model,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
    )

    return chunks
