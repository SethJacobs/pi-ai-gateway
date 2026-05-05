from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # 5 minutes
_CLOUD_CACHE_TTL = 300  # re-read freeride config every 5 min


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
        if now - self._cloud_cache_time < _CLOUD_CACHE_TTL and self._cloud_cache:
            return self._cloud_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.bridge_url}/freeride/models")
                data = resp.json()
                models = [{"id": m} for m in data.get("models", [])]
                if models:
                    self._cloud_cache = models
                    self._cloud_cache_time = now
                return self._cloud_cache or [{"id": self.default_cloud_model}]        except (httpx.HTTPError, Exception) as e:
            logger.debug("Failed to fetch cloud models: %s", e)
            return self._cloud_cache or [{"id": self.default_cloud_model}]

    async def cloud_fallback_models(self) -> list[str]:
        """Return ordered list of cloud model IDs to try, starting with default."""
        models = await self.cloud_models()
        ids = [m["id"] for m in models if m.get("id")]
        if not ids:
            return [self.default_cloud_model]
        # Ensure default is first
        if self.default_cloud_model not in ids:
            ids.insert(0, self.default_cloud_model)
        return ids

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
        """Return the GGUF repo of the top-ranked local model that has GGUF sources."""
        models = await self.local_models()
        for m in models:
            if not isinstance(m, dict):
                continue
            sources = m.get("gguf_sources", [])
            if sources:
                # Prefer bartowski, fall back to first available provider
                repo = next(
                    (s["repo"] for s in sources if s.get("provider") == "bartowski"),
                    sources[0]["repo"],
                )
                return repo
        # Fall back to model name if no GGUF sources found
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
