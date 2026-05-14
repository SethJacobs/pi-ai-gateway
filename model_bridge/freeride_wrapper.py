from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


async def _run_cli(bin_path: str, *args: str) -> dict:
    """Run a CLI command and return stdout/stderr/returncode."""
    proc = await asyncio.create_subprocess_exec(
        bin_path,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "stdout": stdout.decode().strip(),
        "stderr": stderr.decode().strip(),
        "returncode": proc.returncode,
    }


async def freeride_list(bin_path: str) -> dict:
    """Run ``freeride list`` and return raw output."""
    result = await _run_cli(bin_path, "list")
    if result["returncode"] != 0:
        logger.warning("freeride list failed: %s", result["stderr"])
        return {"error": result["stderr"], "models": []}
    return {"raw": result["stdout"], "returncode": result["returncode"]}


async def freeride_list_all(bin_path: str, limit: int = 50) -> list[str]:
    """Return all free model IDs from ``freeride list``, ordered by score."""
    result = await _run_cli(bin_path, "list", "--limit", str(limit))
    if result["returncode"] != 0:
        logger.warning("freeride list failed: %s", result["stderr"])
        return []

    models = []
    for line in result["stdout"].splitlines():
        line = line.strip()
        # Rows look like: "1   openrouter/owl-alpha   1M tokens   0.897   [FALLBACK]"
        # Split on whitespace, second token is the model ID (contains a slash)
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            model_id = parts[1]
            if "/" in model_id:
                models.append(model_id)
    return models


async def freeride_status(bin_path: str) -> dict:
    """Run ``freeride status`` and return raw output."""
    result = await _run_cli(bin_path, "status")
    return {"raw": result["stdout"], "returncode": result["returncode"]}


async def freeride_refresh(bin_path: str) -> dict:
    """Run ``freeride refresh`` to update available free models from API."""
    result = await _run_cli(bin_path, "refresh")
    return {"raw": result["stdout"], "returncode": result["returncode"]}


async def freeride_auto(bin_path: str, fallback_count: int = 5) -> dict:
    """Run ``freeride auto`` to auto-select best free model + fallbacks."""
    result = await _run_cli(bin_path, "auto", "-c", str(fallback_count))
    return {"raw": result["stdout"], "returncode": result["returncode"]}


async def freeride_fallbacks(bin_path: str) -> dict:
    """Run ``freeride fallbacks`` to show current fallback config."""
    result = await _run_cli(bin_path, "fallbacks", "--json")
    if result["returncode"] != 0:
        # try without --json flag
        result = await _run_cli(bin_path, "fallbacks")
    return {"raw": result["stdout"], "returncode": result["returncode"]}
