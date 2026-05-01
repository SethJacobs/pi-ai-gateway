from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request

from ..schemas import HealthResponse, ServiceStatus, StatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    settings = request.app.state.settings
    monitor = request.app.state.system_monitor
    registry = request.app.state.model_registry

    # Check service connectivity
    services: list[ServiceStatus] = []

    # model-bridge
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.model_bridge_url}/health")
            bridge_ok = resp.status_code == 200
    except (httpx.HTTPError, Exception):
        bridge_ok = False
    services.append(ServiceStatus(
        name="model-bridge",
        status="ok" if bridge_ok else "unreachable",
    ))

    # System info
    system_info = await monitor.get_system_info()
    free_ram_mb = system_info.get("available_ram_mb", 0)

    # Local model
    local_loaded = await monitor.is_local_model_loaded()
    local_model = await monitor.get_local_model_name() if local_loaded else None

    # Cloud models count
    cloud = await registry.cloud_models()
    cloud_count = len(cloud) if isinstance(cloud, list) else 0

    return StatusResponse(
        status="ok",
        services=services,
        cloud_models=cloud_count,
        local_model=local_model,
        free_ram_mb=free_ram_mb,
        routing_default="cloud",
    )
