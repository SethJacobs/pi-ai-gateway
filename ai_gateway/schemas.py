from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Request models ---


class FunctionCall(BaseModel):
    """Function call arguments (legacy format)."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call in OpenAI format."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    # Tool calling fields
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """Tool definition in OpenAI format."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    # Tool calling support
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    # Gateway extension: force a routing path
    route: Literal["auto", "cloud", "local"] | None = None


# --- Response models ---


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = []
    usage: Usage = Field(default_factory=Usage)
    # Gateway metadata extensions
    x_route: str = ""
    x_provider: str = ""
    x_intent: str = ""


# --- Model listing ---


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = ""
    source: str = ""
    context_length: int | None = None
    fit: str | None = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = []


# --- Health / Status ---


class ServiceStatus(BaseModel):
    name: str
    status: str  # "ok", "degraded", "unreachable"


class HealthResponse(BaseModel):
    status: str = "ok"


class StatusResponse(BaseModel):
    status: str = "ok"
    services: list[ServiceStatus] = []
    cloud_models: int = 0
    local_model: str | None = None
    free_ram_mb: int = 0
    routing_default: str = "cloud"
