from __future__ import annotations

import asyncio
import logging
import os
import signal

import httpx

logger = logging.getLogger(__name__)


class LlamaManager:
    """Manages a single llama-server process for local inference."""

    def __init__(
        self,
        server_bin: str,
        port: int,
        model_dir: str,
        max_context: int = 2048,
        threads: int = 4,
    ) -> None:
        self.server_bin = server_bin
        self.port = port
        self.model_dir = model_dir
        self.max_context = max_context
        self.threads = threads

        self._process: asyncio.subprocess.Process | None = None
        self._model_name: str | None = None
        self._model_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def load_model(self, model_path: str) -> dict:
        """Stop any running server, then start with the given model."""
        await self.stop()

        full_path = model_path if os.path.isabs(model_path) else os.path.join(self.model_dir, model_path)
        if not os.path.exists(full_path):
            return {"status": "error", "error": f"Model not found: {full_path}"}

        logger.info("Starting llama-server with model: %s", full_path)

        self._process = await asyncio.create_subprocess_exec(
            self.server_bin,
            "--model", full_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(self.max_context),
            "--threads", str(self.threads),
            "--batch-size", "256",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._model_path = full_path
        self._model_name = os.path.basename(model_path)

        # Wait for server to become ready (Pi can be slow loading large models)
        for _ in range(60):
            await asyncio.sleep(1)
            if await self._health_check():
                logger.info("llama-server ready: %s", self._model_name)
                return {"status": "loaded", "model": self._model_name}

        # Timed out
        await self.stop()
        return {"status": "error", "error": "Server failed to start within 30s"}

    async def stop(self) -> dict:
        """Stop the running llama-server process."""
        if self._process is not None:
            try:
                self._process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError, OSError):
                try:
                    self._process.kill()
                except (ProcessLookupError, OSError):
                    pass
            self._process = None
            name = self._model_name
            self._model_name = None
            self._model_path = None
            logger.info("llama-server stopped (was: %s)", name)
            return {"status": "stopped"}
        return {"status": "not_running"}

    async def status(self) -> dict:
        """Return current server status."""
        if self._process is None or self._process.returncode is not None:
            return {"status": "not_running", "model": None}

        healthy = await self._health_check()
        return {
            "status": "loaded" if healthy else "starting",
            "model": self._model_name,
            "port": self.port,
        }

    async def _health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                # llama-server health endpoint varies by version;
                # /health returns 200 when ready, 503 when loading, 404 on older builds
                # Fall back to /v1/models which is always present
                resp = await client.get(f"{self.base_url}/health")
                if resp.status_code in (200, 503):
                    # 503 means loading, 200 means ready
                    return resp.status_code == 200
                # Older builds: try /v1/models
                resp = await client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except (httpx.HTTPError, ConnectionError):
            return False
