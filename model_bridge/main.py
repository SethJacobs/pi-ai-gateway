from __future__ import annotations

import asyncio
import glob
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from .config import BridgeSettings
from .freeride_wrapper import freeride_auto, freeride_fallbacks, freeride_list, freeride_list_all, freeride_refresh, freeride_status
from .llama_manager import LlamaManager
from .llmfit_wrapper import llmfit_download, llmfit_recommend, llmfit_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = BridgeSettings()

llama = LlamaManager(
    server_bin=settings.llama_server_bin,
    port=settings.llama_server_port,
    model_dir=settings.model_dir,
    max_context=settings.max_context_size,
    threads=settings.llama_threads,
)


async def _freeride_refresh_loop(bin_path: str) -> None:
    """Periodically refresh freeride model list and auto-select best fallbacks."""
    interval_hours = settings.freeride_refresh_interval_hours
    while True:
        try:
            logger.info("Running freeride refresh + auto...")
            await freeride_refresh(bin_path)
            result = await freeride_auto(bin_path, fallback_count=5)
            logger.info("freeride auto complete: %s", result.get("raw", "")[:200])
        except Exception as e:
            logger.warning("freeride refresh/auto failed: %s", e)
        await asyncio.sleep(interval_hours * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Start background freeride refresh task
    refresh_task = asyncio.create_task(
        _freeride_refresh_loop(settings.freeride_bin)
    )
    yield
    refresh_task.cancel()
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


@app.get("/freeride/models")
async def get_freeride_models() -> dict:
    """Return ordered list of all cloud model IDs.
    
    Configured fallbacks come first (already ranked by freeride auto),
    followed by remaining free models in score order.
    """
    # Get configured fallbacks (primary + numbered fallbacks)
    fallback_data = await freeride_fallbacks(settings.freeride_bin)
    raw = fallback_data.get("raw", "")
    configured: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Current primary:"):
            primary = line.split("Current primary:", 1)[1].strip()
            if primary:
                configured.append(primary)
        else:
            m = re.match(r"^\d+\.\s+(\S+)", line)
            if m:
                configured.append(m.group(1))

    # Get full ranked list and append any not already in configured
    all_models = await freeride_list_all(settings.freeride_bin, limit=50)
    seen = set(configured)
    for model_id in all_models:
        if model_id not in seen:
            configured.append(model_id)
            seen.add(model_id)

    return {"models": configured or ["openrouter/auto"]}


# --- LLMFit endpoints ---


@app.get("/llmfit/recommend")
async def get_llmfit_recommend(limit: int = 5) -> dict:
    return await llmfit_recommend(settings.llmfit_bin, limit)


@app.get("/llmfit/system")
async def get_llmfit_system() -> dict:
    return await llmfit_system(settings.llmfit_bin)


@app.post("/llmfit/download")
async def post_llmfit_download(model: str) -> dict:
    """Download a GGUF model via llmfit. Returns the local file path on success.

    If the model is already downloaded, returns the existing path immediately.
    """
    model_basename = model.split("/")[-1].replace("-GGUF", "")
    pattern = os.path.join(settings.model_dir, f"{model_basename}*.gguf")
    existing = glob.glob(pattern)
    if existing:
        path = existing[0]
        logger.info("Model already downloaded: %s", path)
        return {"status": "ok", "path": path}
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
