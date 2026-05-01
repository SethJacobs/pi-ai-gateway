from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # 5 minutes


class ModelRegistry:
    """Aggregates model info from FreeRide (cloud) and LLMFit (local) via model-bridge."""

    def __init__(self, bridge_url: str, default_cloud_model: str) -> None:
        self.bridge_url = bridge_url
        self.default_cloud_model = default_cloud_model

        self._cloud_cache: list[dict] = []
        self._cloud_cache_time: float = 0
        self._local_cache: list[dict] = []
        self._local_cache_time: float = 0

    async def cloud_models(self) -> list[dict]:
        """Get available cloud models from FreeRide via model-bridge."""
        now = time.time()
        if now - self._cloud_cache_time < _CACHE_TTL and self._cloud_cache:
            return self._cloud_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.bridge_url}/freeride/list")
                data = resp.json()
                # freeride list returns raw text; cache it as-is
                self._cloud_cache = [data] if isinstance(data, dict) else data
                self._cloud_cache_time = now
                return self._cloud_cache
        except (httpx.HTTPError, Exception) as e:
            logger.debug("Failed to fetch cloud models: %s", e)
            return self._cloud_cache

    async def local_models(self) -> list[dict]:
        """Get recommended local models from LLMFit via model-bridge."""
        now = time.time()
        if now - self._local_cache_time < _CACHE_TTL and self._local_cache:
            return self._local_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.bridge_url}/llmfit/recommend")
                data = resp.json()
                models = data.get("models", [])
                self._local_cache = models
                self._local_cache_time = now
                return models
        except (httpx.HTTPError, Exception) as e:
            logger.debug("Failed to fetch local models: %s", e)
            return self._local_cache

    async def best_cloud_model(self) -> str:
        """Return the best cloud model ID. Falls back to default."""
        # FreeRide manages this via openclaw config; we use the default
        # which can be overridden by calling POST /freeride/auto first
        return self.default_cloud_model

    async def best_local_model(self) -> str:
        """Return the top-ranked local model name from LLMFit."""
        models = await self.local_models()
        if models and isinstance(models[0], dict):
            return models[0].get("name", "unknown")
        return "unknown"

    async def current_local_model(self) -> str:
        """Return the currently loaded local model name."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.bridge_url}/local/status")
                data = resp.json()
                return data.get("model", "unknown")
        except (httpx.HTTPError, Exception):
            return "unknown"
