import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from any_llm.types.completion import ChatCompletionChunk
from simple_llm_proxy.main import app


@pytest.fixture
def mock_llm_completion():
    """Mock any_llm.acompletion to avoid needing real API keys."""
    def create_mock_completion(model="openai:gpt-3.5-turbo", content="Hello! This is a test response.", **kwargs):
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,  # Use the model from the request
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    def create_mock_stream(model="openai:gpt-3.5-turbo", **kwargs):
        """Create mock streaming response chunks."""
        chunk_data = [
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": model,  # Use the model from the request
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }
                ]
            },
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None
                    }
                ]
            },
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " test"},
                        "finish_reason": None
                    }
                ]
            },
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
        ]
        
        async def async_chunks():
            for chunk_dict in chunk_data:
                # Create actual ChatCompletionChunk objects
                chunk = ChatCompletionChunk(**chunk_dict)
                yield chunk
        
        return async_chunks()
    
    async def mock_acompletion(stream=False, **kwargs):
        if stream:
            return create_mock_stream(**kwargs)
        else:
            return create_mock_completion(**kwargs)
    
    with patch('simple_llm_proxy.pipeline.any_llm.acompletion') as mock_completion:
        mock_completion.side_effect = mock_acompletion
        yield mock_completion


@pytest.fixture
async def client():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


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
        assert len(data["data"]) >= 2
        
        # Check model structure
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model
            assert model["object"] == "model"


class TestChatCompletions:
    
    @pytest.fixture
    def basic_request(self):
        return {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    
    @pytest.fixture  
    def multimodal_request(self):
        return {
            "model": "openai:gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."}
                    }
                ]
            }]
        }

    async def test_basic_chat_completion(self, client, basic_request, mock_llm_completion):
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
            "user": "test-user"
        }
        
        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model"] == "gpt-4"

    async def test_multimodal_chat_completion(self, client, multimodal_request, mock_llm_completion):
        # Override the mock to return multimodal-specific content  
        def create_multimodal_response(**kwargs):
            return {
                "id": "chatcmpl-test123",
                "object": "chat.completion", 
                "created": 1234567890,
                "model": kwargs.get("model", "openai:gpt-4-vision-preview"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I can see a multimodal image in this request."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
        
        mock_llm_completion.side_effect = lambda stream=False, **kwargs: (
            create_multimodal_response(**kwargs)
        )
        
        response = await client.post("/v1/chat/completions", json=multimodal_request)
        assert response.status_code == 200
        
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        
        # Check that it recognizes multimodal content
        assert "multimodal" in content.lower() or "image" in content.lower()

    async def test_streaming_chat_completion(self, client, basic_request, mock_llm_completion):
        basic_request["stream"] = True
        
        response = await client.post("/v1/chat/completions", json=basic_request)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        content = response.text
        lines = content.strip().split('\n')
        
        # Should have data lines and final [DONE]
        data_lines = [line for line in lines if line.startswith('data: ') and not line.endswith('[DONE]')]
        done_lines = [line for line in lines if line.endswith('[DONE]')]
        
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

    async def test_streaming_multimodal_completion(self, client, multimodal_request, mock_llm_completion):
        multimodal_request["stream"] = True
        
        # Override mock for streaming multimodal content
        def create_multimodal_stream(**kwargs):
            chunk_data = [
                {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "openai:gpt-4-vision-preview",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
                },
                {
                    "id": "chatcmpl-test123", 
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "openai:gpt-4-vision-preview",
                    "choices": [{"index": 0, "delta": {"content": "I can see this multimodal image"}, "finish_reason": None}]
                },
                {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion.chunk", 
                    "created": 1234567890,
                    "model": "openai:gpt-4-vision-preview",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
            ]
            
            async def async_chunks():
                for chunk_dict in chunk_data:
                    chunk = ChatCompletionChunk(**chunk_dict)
                    yield chunk
            return async_chunks()
            
        mock_llm_completion.side_effect = lambda stream=False, **kwargs: (
            create_multimodal_stream(**kwargs) if stream else create_multimodal_stream(**kwargs)
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
            headers={"content-type": "application/json"}
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
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": []
        }
        
        response = await client.post("/v1/chat/completions", json=request_data)
        # Empty messages should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "messages" in data["error"]["message"]

    async def test_missing_model(self, client):
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "model" in data["error"]["message"]

    async def test_invalid_message_structure(self, client):
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": ["invalid message structure"]
        }
        
        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        # Either our custom message or Pydantic's message is acceptable
        assert ("must be an object" in data["error"]["message"] or 
                "should be a valid dictionary" in data["error"]["message"])

    async def test_invalid_temperature(self, client):
        request_data = {
            "model": "openai:gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 5.0  # Too high
        }
        
        response = await client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "validation_error"
        assert "temperature" in data["error"]["message"]


class TestMessageParsing:
    
    def test_text_message_parsing(self):
        from simple_llm_proxy.main import OpenAIMessage
        
        msg = OpenAIMessage(role="user", content="Hello world")
        assert not msg.has_images()
        assert msg.get_text_content() == "Hello world"
        assert msg.get_image_urls() == []

    def test_multimodal_message_parsing(self):
        from simple_llm_proxy.main import OpenAIMessage
        
        content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url", 
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        ]
        
        msg = OpenAIMessage(role="user", content=content)
        assert msg.has_images()
        assert msg.get_text_content() == "What's in this image?"
        assert msg.get_image_urls() == ["https://example.com/image.jpg"]

    def test_multiple_images_parsing(self):
        from simple_llm_proxy.main import OpenAIMessage
        
        content = [
            {"type": "text", "text": "Compare these images:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image1.jpg"}
            },
            {
                "type": "image_url", 
                "image_url": {"url": "https://example.com/image2.jpg"}
            }
        ]
        
        msg = OpenAIMessage(role="user", content=content)
        assert msg.has_images()
        assert msg.get_text_content() == "Compare these images:"
        assert len(msg.get_image_urls()) == 2
        assert "image1.jpg" in msg.get_image_urls()[0]
        assert "image2.jpg" in msg.get_image_urls()[1]


class TestRequestParsing:
    
    def test_request_with_all_parameters(self):
        from simple_llm_proxy.main import ChatCompletionRequest
        
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 150,
            "stream": True,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "user": "test-user",
            "n": 2,
            "stop": ["END"],
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "auto"
        }
        
        req = ChatCompletionRequest(data)
        assert req.model == "gpt-4"
        assert len(req.messages) == 1
        assert req.temperature == 0.8
        assert req.top_p == 0.95
        assert req.max_tokens == 150
        assert req.stream is True
        assert req.presence_penalty == 0.2
        assert req.frequency_penalty == 0.1
        assert req.user == "test-user"
        assert req.n == 2
        assert req.stop == ["END"]
        assert req.tools is not None
        assert req.tool_choice == "auto"

    def test_request_defaults(self):
        from simple_llm_proxy.main import ChatCompletionRequest
        
        data = {"messages": []}
        req = ChatCompletionRequest(data)
        
        assert req.model == "openai:gpt-3.5-turbo"
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.max_tokens is None
        assert req.stream is False
        assert req.presence_penalty == 0.0
        assert req.frequency_penalty == 0.0
        assert req.n == 1