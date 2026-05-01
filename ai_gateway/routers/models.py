from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import ModelInfo, ModelListResponse

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    registry = request.app.state.model_registry

    models: list[ModelInfo] = []

    # Local models from llmfit
    local = await registry.local_models()
    for m in local:
        if isinstance(m, dict):
            models.append(ModelInfo(
                id=m.get("name", "unknown"),
                owned_by="local",
                source="local-llmfit",
                context_length=m.get("context_length"),
                fit=m.get("fit_level"),
            ))

    # Cloud default
    default_cloud = await registry.best_cloud_model()
    models.append(ModelInfo(
        id=default_cloud,
        owned_by="openrouter",
        source="openrouter-free",
    ))

    return ModelListResponse(data=models)
