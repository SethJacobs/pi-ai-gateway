from __future__ import annotations

import pytest
import respx
from httpx import Response

from ai_gateway.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDefinition,
    ToolDefinition,
)
from ai_gateway.services.cloud_provider import CloudProvider


@pytest.fixture
def provider() -> CloudProvider:
    return CloudProvider(api_key="test-key", base_url="https://openrouter.ai/api/v1")


MOCK_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "qwen/qwen3-coder:free",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
}


class TestCloudProvider:
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_chat(self, provider: CloudProvider) -> None:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=Response(200, json=MOCK_RESPONSE)
        )

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )
        resp = await provider.chat(req, "qwen/qwen3-coder:free")

        assert resp.id == "chatcmpl-test123"
        assert resp.model == "qwen/qwen3-coder:free"
        assert resp.choices[0].message.content == "Hello! How can I help?"
        assert resp.x_route == "cloud"
        assert resp.x_provider == "openrouter"
        assert resp.usage.total_tokens == 13

        await provider.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_raises(self, provider: CloudProvider) -> None:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=Response(429, json={"error": "rate limited"})
        )

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with pytest.raises(Exception):
            await provider.chat(req, "some/model:free")

        await provider.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_sends_auth_header(self, provider: CloudProvider) -> None:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=Response(200, json=MOCK_RESPONSE)
        )

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )
        await provider.chat(req, "model")

        assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"

        await provider.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_calling_request(self, provider: CloudProvider) -> None:
        """Test that tools are passed through to the API."""
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "id": "chatcmpl-tool123",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "anthropic/claude-3.5-sonnet",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_abc123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"location": "Boston"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                },
            )
        )

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What's the weather in Boston?")],
            tools=[
                ToolDefinition(
                    type="function",
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get current weather",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"}
                            },
                            "required": ["location"],
                        },
                    ),
                )
            ],
        )
        resp = await provider.chat(req, "anthropic/claude-3.5-sonnet")

        # Verify request included tools
        request_body = route.calls[0].request.content
        import json
        sent = json.loads(request_body)
        assert "tools" in sent
        assert len(sent["tools"]) == 1
        assert sent["tools"][0]["function"]["name"] == "get_weather"

        # Verify response includes tool_calls
        assert resp.choices[0].message.tool_calls is not None
        assert len(resp.choices[0].message.tool_calls) == 1
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert resp.choices[0].message.tool_calls[0].function.arguments == '{"location": "Boston"}'
        assert resp.choices[0].finish_reason == "tool_calls"

        await provider.close()
