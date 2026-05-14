from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL = 30  # seconds


class SystemMonitor:
    """Queries model-bridge for system state and local model status."""

    def __init__(self, bridge_url: str) -> None:
        self.bridge_url = bridge_url
        self._cache: dict = {}
        self._cache_time: float = 0
        self._local_cache: dict = {}
        self._local_cache_time: float = 0

    async def get_system_info(self) -> dict:
        """Return system RAM/CPU info, cached for 30s."""
        now = time.time()
        if now - self._cache_time < _CACHE_TTL and self._cache:
            return self._cache

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.bridge_url}/llmfit/system")
                raw = resp.json()

                # Parse llmfit system output for RAM info
                # llmfit system returns text; extract what we can
                text = raw.get("raw", "")
                info = self._parse_system_text(text)
                self._cache = info
                self._cache_time = now
                return info
        except (httpx.HTTPError, Exception) as e:
            logger.debug("Failed to get system info: %s", e)
            # Return safe defaults
            return self._cache or {"total_ram_mb": 8192, "available_ram_mb": 2560, "cpu_cores": 4}

    async def is_local_model_loaded(self) -> bool:
        """Check if a local model is currently loaded in llama-server."""
        now = time.time()
        if now - self._local_cache_time < 10 and self._local_cache:
            return self._local_cache.get("status") == "loaded"

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.bridge_url}/local/status")
                data = resp.json()
                self._local_cache = data
                self._local_cache_time = now
                logger.debug("Local model status response: %s", data)
                return data.get("status") == "loaded"
        except (httpx.HTTPError, Exception):
            return False

    async def get_local_model_name(self) -> str | None:
        """Return the name of the currently loaded local model."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.bridge_url}/local/status")
                data = resp.json()
                return data.get("model")
        except (httpx.HTTPError, Exception):
            return None

    @staticmethod
    def _parse_system_text(text: str) -> dict:
        """Best-effort parse of llmfit system text output."""
        info: dict = {"total_ram_mb": 8192, "available_ram_mb": 2560, "cpu_cores": 4}

        for line in text.splitlines():
            lower = line.lower()
            if "total ram" in lower:
                try:
                    gb = float(line.split(":")[-1].strip().split()[0])
                    info["total_ram_mb"] = int(gb * 1024)
                except (ValueError, IndexError):
                    pass
            elif "available ram" in lower:
                try:
                    gb = float(line.split(":")[-1].strip().split()[0])
                    info["available_ram_mb"] = int(gb * 1024)
                except (ValueError, IndexError):
                    pass
            elif "cpu" in lower and "core" in lower:
                try:
                    cores = int(line.split(":")[-1].strip().split()[0])
                    info["cpu_cores"] = cores
                except (ValueError, IndexError):
                    pass

        return info
