from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import GatewaySettings
from .routers import chat, health, models
from .services.cloud_provider import CloudProvider
from .services.local_provider import LocalProvider
from .services.model_registry import ModelRegistry
from .services.router_engine import RouterEngine
from .services.system_monitor import SystemMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = GatewaySettings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Wire up services
    app.state.settings = settings
    app.state.cloud_provider = CloudProvider(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    app.state.local_provider = LocalProvider(
        bridge_url=settings.model_bridge_url,
    )
    app.state.model_registry = ModelRegistry(
        bridge_url=settings.model_bridge_url,
        default_cloud_model=settings.default_cloud_model,
    )
    app.state.system_monitor = SystemMonitor(
        bridge_url=settings.model_bridge_url,
    )
    app.state.router_engine = RouterEngine(
        config=settings,
        system_monitor=app.state.system_monitor,
        model_registry=app.state.model_registry,
    )

    logger.info("AI Gateway started — bridge=%s", settings.model_bridge_url)
    yield

    await app.state.cloud_provider.close()
    logger.info("AI Gateway stopped")


app = FastAPI(
    title="Pi AI Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Optional bearer-token auth. Skip for /health."""
    if settings.gateway_api_key and request.url.path != "/health":
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != settings.gateway_api_key:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


app.include_router(chat.router)
app.include_router(models.router)
app.include_router(health.router)
