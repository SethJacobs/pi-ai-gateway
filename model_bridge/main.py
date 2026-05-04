from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from .config import BridgeSettings
from .freeride_wrapper import freeride_auto, freeride_fallbacks, freeride_list, freeride_status
from .llama_manager import LlamaManager
from .llmfit_wrapper import llmfit_download, llmfit_recommend, llmfit_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

settings = BridgeSettings()

llama = LlamaManager(
    server_bin=settings.llama_server_bin,
    port=settings.llama_server_port,
    model_dir=settings.model_dir,
    max_context=settings.max_context_size,
    threads=settings.llama_threads,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await llama.stop()


app = FastAPI(title="Model Bridge", version="0.1.0", lifespan=lifespan)


# --- FreeRide endpoints ---


@app.get("/freeride/list")
async def get_freeride_list() -> dict:
    return await freeride_list(settings.freeride_bin)


@app.get("/freeride/status")
async def get_freeride_status() -> dict:
    return await freeride_status(settings.freeride_bin)


@app.post("/freeride/auto")
async def post_freeride_auto(fallback_count: int = 5) -> dict:
    return await freeride_auto(settings.freeride_bin, fallback_count)


@app.get("/freeride/fallbacks")
async def get_freeride_fallbacks() -> dict:
    return await freeride_fallbacks(settings.freeride_bin)


# --- LLMFit endpoints ---


@app.get("/llmfit/recommend")
async def get_llmfit_recommend(limit: int = 5) -> dict:
    return await llmfit_recommend(settings.llmfit_bin, limit)


@app.get("/llmfit/system")
async def get_llmfit_system() -> dict:
    return await llmfit_system(settings.llmfit_bin)


@app.post("/llmfit/download")
async def post_llmfit_download(model: str) -> dict:
    """Download a GGUF model via llmfit. Returns the local file path on success."""
    return await llmfit_download(settings.llmfit_bin, model)


# --- Local model management ---


@app.post("/local/load")
async def post_local_load(model_path: str, context_size: int = 2048) -> dict:
    llama.max_context = context_size
    return await llama.load_model(model_path)


@app.post("/local/stop")
async def post_local_stop() -> dict:
    return await llama.stop()


@app.get("/local/status")
async def get_local_status() -> dict:
    return await llama.status()


# --- Health ---


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "model-bridge"}
