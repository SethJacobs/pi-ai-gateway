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


async def freeride_status(bin_path: str) -> dict:
    """Run ``freeride status`` and return raw output."""
    result = await _run_cli(bin_path, "status")
    return {"raw": result["stdout"], "returncode": result["returncode"]}


async def freeride_auto(bin_path: str, fallback_count: int = 5) -> dict:
    """Run ``freeride auto`` to auto-select best free model + fallbacks."""
    result = await _run_cli(bin_path, "auto", "-c", str(fallback_count))
    return {"raw": result["stdout"], "returncode": result["returncode"]}


async def freeride_fallbacks(bin_path: str) -> dict:
    """Run ``freeride fallbacks`` to show current fallback config."""
    result = await _run_cli(bin_path, "fallbacks")
    return {"raw": result["stdout"], "returncode": result["returncode"]}
