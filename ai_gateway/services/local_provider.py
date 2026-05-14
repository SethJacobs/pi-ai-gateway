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
        self._llama_base = bridge_url.replace(":9099", ":8081")

    async def chat(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Send request to llama-server's OpenAI-compatible endpoint."""
        payload: dict = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools is not None:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        logger.info("Local request: model=%s messages=%d tools=%s",
                   model, len(request.messages),
                   len(request.tools) if request.tools else 0)

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

        # Parse message with tool calls support
        message = ChatMessage(
            role=msg.get("role", "assistant"),
            content=msg.get("content"),
        )

        # Handle tool_calls if present
        if "tool_calls" in msg and msg["tool_calls"]:
            from ..schemas import FunctionCall, ToolCall
            message.tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in msg["tool_calls"]
            ]

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-local-{uuid.uuid4().hex[:8]}"),
            created=data.get("created", int(time.time())),
            model=model,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    finish_reason=first.get("finish_reason", "stop"),
                )
            ],
            usage=Usage(**(data.get("usage") or {})),
            x_route="local",
            x_provider="llama-server",
        )
