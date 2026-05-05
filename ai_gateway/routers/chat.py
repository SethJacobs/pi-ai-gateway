from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
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
    registry = request.app.state.model_registry

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

    # If cloud failed, cycle through all freeride fallback models before giving up
    if decision.route.value == "cloud":
        fallback_models = await registry.cloud_fallback_models()
        for model_id in fallback_models:
            if model_id == decision.model:
                continue  # already tried this one
            try:
                logger.info("Trying cloud fallback model: %s", model_id)
                resp = await cloud.chat(body, model_id)
                resp.x_intent = decision.intent.value
                return resp
            except Exception as exc:
                logger.warning("Cloud fallback model %s failed: %s", model_id, exc)

    # Explicit fallback route (e.g. local was preferred, cloud failed)
    if decision.fallback_route is not None:
        logger.info("Attempting fallback route: %s", decision.fallback_route.value)
        try:
            if decision.fallback_route.value == "cloud":
                fallback_model = await registry.best_cloud_model()
                resp = await cloud.chat(body, fallback_model)
            else:
                fallback_local_model = await registry.best_local_model()
                resp = await local.chat(body, fallback_local_model)
            resp.x_intent = decision.intent.value
            return resp
        except Exception as fallback_err:
            raise HTTPException(
                status_code=502,
                detail=f"Both routes failed. primary: {primary_err}, fallback: {fallback_err}",
            ) from fallback_err

    # No fallback configured — try to auto-load a local model and retry
    logger.info("No fallback route set; attempting local auto-load")
    try:
        registry = request.app.state.model_registry
        bridge_url = request.app.state.settings.model_bridge_url
        best_model = await registry.best_local_model()
        if best_model and best_model != "unknown":
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Download the model first (no-op if already present, fast if cached)
                logger.info("Downloading local model: %s", best_model)
                dl_resp = await client.post(
                    f"{bridge_url}/llmfit/download",
                    params={"model": best_model},
                )
                dl_data = dl_resp.json()
                logger.info("Download result: %s", dl_data)

                if dl_data.get("status") != "ok":
                    raise HTTPException(
                        status_code=502,
                        detail=f"Model download failed: {dl_data.get('error')}",
                    )

                # Use the absolute path returned by llmfit download
                model_path = dl_data.get("path", best_model)

                load_resp = await client.post(
                    f"{bridge_url}/local/load",
                    params={"model_path": model_path},
                )
                load_data = load_resp.json()
                logger.info("Auto-load result: %s", load_data)

            if load_data.get("status") == "loaded":
                resp = await local.chat(body, best_model)
                resp.x_intent = decision.intent.value
                return resp
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Cloud failed and local auto-load failed: {load_data}",
                )
    except HTTPException:
        raise
    except Exception as load_err:
        logger.warning("Local auto-load attempt failed: %s", load_err)

    raise HTTPException(
        status_code=502,
        detail=f"Route failed with no fallback available: {primary_err}",
    )
