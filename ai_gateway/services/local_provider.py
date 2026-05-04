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


class LocalProvider:
    """Calls llama-server (via model-bridge host) for local inference."""

    def __init__(self, bridge_url: str) -> None:
        self.bridge_url = bridge_url
        # llama-server runs on the host, port configured in model-bridge
        # We derive the llama port from bridge_url (replace 9099 with 8081)
        self._llama_base = bridge_url.replace(":9099", ":8188")

    async def chat(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Send request to llama-server's OpenAI-compatible endpoint."""
        payload: dict = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        logger.info("Local request: model=%s messages=%d", model, len(request.messages))

        # llama-server exposes /v1/chat/completions (OpenAI-compatible)
        async with httpx.AsyncClient(
            base_url=self._llama_base,
            timeout=httpx.Timeout(120.0, connect=5.0),
        ) as client:
            resp = await client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices_raw = data.get("choices", [])
        first = choices_raw[0] if choices_raw else {}
        msg = first.get("message", {})

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-local-{uuid.uuid4().hex[:8]}"),
            created=data.get("created", int(time.time())),
            model=model,
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
            x_route="local",
            x_provider="llama-server",
        )
