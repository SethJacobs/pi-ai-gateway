from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from ..schemas import ChatCompletionRequest, ChatCompletionResponse

if TYPE_CHECKING:
    from ..services.router_engine import RouterEngine

logger = logging.getLogger(__name__)

router = APIRouter()


def _engine(request: Request) -> RouterEngine:
    return request.app.state.router_engine  # type: ignore[no-any-return]


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@router.post("/chat", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:
    engine = _engine(request)
    decision = await engine.decide(body)

    logger.info(
        "Routing: intent=%s route=%s model=%s reason=%s",
        decision.intent.value,
        decision.route.value,
        decision.model,
        decision.reason,
    )

    cloud = request.app.state.cloud_provider
    local = request.app.state.local_provider

    # Primary attempt
    primary_err: Exception | None = None
    try:
        if decision.route.value == "cloud":
            resp = await cloud.chat(body, decision.model)
        else:
            resp = await local.chat(body, decision.model)
        resp.x_intent = decision.intent.value
        return resp
    except Exception as exc:
        primary_err = exc
        logger.warning("Primary route (%s) failed: %s", decision.route.value, exc)

    # Fallback attempt
    if decision.fallback_route is not None:
        try:
            if decision.fallback_route.value == "cloud":
                fallback_model = await request.app.state.model_registry.best_cloud_model()
                resp = await cloud.chat(body, fallback_model)
            else:
                resp = await local.chat(body, decision.model)
            resp.x_intent = decision.intent.value
            return resp
        except Exception as fallback_err:
            raise HTTPException(
                status_code=502,
                detail=f"Both routes failed. primary: {primary_err}, fallback: {fallback_err}",
            ) from fallback_err

    raise HTTPException(
        status_code=502,
        detail=f"Route failed with no fallback available: {primary_err}",
    )
