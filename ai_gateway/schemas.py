from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


# --- Request models ---


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
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
