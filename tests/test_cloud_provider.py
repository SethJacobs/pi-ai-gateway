from __future__ import annotations

import pytest
import respx
from httpx import Response

from ai_gateway.schemas import ChatCompletionRequest, ChatMessage
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
