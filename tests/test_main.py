import json
import pytest
from any_llm.types.completion import ChatCompletionChunk

from helpers import create_chat_completion, create_stream_chunks


# Note: client, mock_llm_completion, and basic_request fixtures are provided by conftest.py


class TestHealthEndpoints:
    async def test_root_health_check(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    async def test_health_endpoint(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestModelsEndpoint:
    async def test_list_models(self, client):
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 0

        # Check model structure
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model
            assert model["object"] == "model"


class TestChatCompletions:
    """Tests for chat completion endpoint."""

    # Note: basic_request fixture is provided by conftest.py

    @pytest.fixture
    def multimodal_request(self):
        return {
            "model": "openai:gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
                            },
                        },
                    ],
                }
            ],
        }

    async def test_basic_chat_completion(
        self, client, basic_request, mock_llm_completion
    ):
        response = await client.post("/v1/chat/completions", json=basic_request)
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["model"] == basic_request["model"]
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "usage" in data

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] == "stop"

    async def test_chat_completion_with_parameters(self, client, mock_llm_completion):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "user": "test-user",
        }

        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "gpt-4"

    async def test_multimodal_chat_completion(
        self, client, multimodal_request, mock_llm_completion
    ):
        # Override the mock to return multimodal-specific content
        mock_llm_completion.side_effect = (
            lambda stream=False, **kwargs: create_chat_completion(
                model=kwargs.get("model", "openai:gpt-4-vision-preview"),
                content="I can see a multimodal image in this request.",
            )
        )

        response = await client.post("/v1/chat/completions", json=multimodal_request)
        assert response.status_code == 200

        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Check that it recognizes multimodal content
        assert "multimodal" in content.lower() or "image" in content.lower()

    async def test_streaming_chat_completion(
        self, client, basic_request, mock_llm_completion
    ):
        basic_request["stream"] = True

        response = await client.post("/v1/chat/completions", json=basic_request)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        content = response.text
        lines = content.strip().split("\n")

        # Should have data lines and final [DONE]
        data_lines = [
            line
            for line in lines
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        done_lines = [line for line in lines if line.endswith("[DONE]")]

        assert len(data_lines) > 0
        assert len(done_lines) == 1

        # Check first chunk structure
        first_chunk_data = data_lines[0][6:]  # Remove 'data: ' prefix
        chunk = json.loads(first_chunk_data)

        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "created" in chunk
        assert "choices" in chunk
        assert len(chunk["choices"]) == 1

        choice = chunk["choices"][0]
        assert choice["index"] == 0
        assert "delta" in choice

    async def test_streaming_multimodal_completion(
        self, client, multimodal_request, mock_llm_completion
    ):
        multimodal_request["stream"] = True

        # Override mock for streaming multimodal content
        async def create_multimodal_stream(**kwargs):
            chunks = create_stream_chunks(
                model="openai:gpt-4-vision-preview",
                content_parts=["I can see this multimodal image"],
            )
            for chunk in chunks:
                yield chunk

        mock_llm_completion.side_effect = (
            lambda stream=False, **kwargs: create_multimodal_stream(**kwargs)
        )

        response = await client.post("/v1/chat/completions", json=multimodal_request)
        assert response.status_code == 200

        content = response.text
        # Should contain multimodal-specific content
        assert "multimodal" in content.lower() or "image" in content.lower()


class TestErrorHandling:
    async def test_invalid_json(self, client):
        response = await client.post(
            "/v1/chat/completions",
            content="invalid json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"

    async def test_missing_messages(self, client, mock_llm_completion):
        request_data = {"model": "openai:gpt-3.5-turbo"}

        response = await client.post("/v1/chat/completions", json=request_data)
        # Missing messages should return validation error (empty list is invalid)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "messages" in data["error"]["message"]

    async def test_empty_messages(self, client, mock_llm_completion):
        request_data = {"model": "openai:gpt-3.5-turbo", "messages": []}

        response = await client.post("/v1/chat/completions", json=request_data)
        # Empty messages should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "messages" in data["error"]["message"]

    async def test_missing_model(self, client):
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}

        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "model" in data["error"]["message"]

    async def test_invalid_message_structure(self, client):
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": ["invalid message structure"],
        }

        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        # Either our custom message or Pydantic's message is acceptable
        assert (
            "must be an object" in data["error"]["message"]
            or "should be a valid dictionary" in data["error"]["message"]
        )

    async def test_invalid_temperature(self, client, mock_llm_completion):
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 5.0,  # Too high
        }

        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "temperature" in data["error"]["message"]
