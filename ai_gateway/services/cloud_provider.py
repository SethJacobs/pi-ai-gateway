from __future__ import annotations

import logging
import time
import uuid

import httpx

from ..schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)

logger = logging.getLogger(__name__)


class CloudProvider:
    """Calls OpenRouter API for cloud-based LLM inference."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://pi-ai-gateway.local",
                    "X-Title": "Pi AI Gateway",
                },
            )
        return self._client

    async def chat(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        payload: dict = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        logger.info("Cloud request: model=%s messages=%d", model, len(request.messages))

        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choices_raw = data.get("choices", [])
        first = choices_raw[0] if choices_raw else {}
        msg = first.get("message", {})

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content", ""),
                    ),
                    finish_reason=first.get("finish_reason", "stop"),
                )
            ],
            usage=Usage(**(data.get("usage") or {})),
            x_route="cloud",
            x_provider="openrouter",
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
