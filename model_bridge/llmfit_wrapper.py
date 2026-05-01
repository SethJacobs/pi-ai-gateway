from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


async def _run_cli(bin_path: str, *args: str) -> dict:
    """Run a CLI command and return parsed output."""
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


async def llmfit_recommend(bin_path: str, limit: int = 5) -> dict:
    """Run ``llmfit recommend`` and parse JSON output."""
    result = await _run_cli(bin_path, "recommend")
    if result["returncode"] != 0:
        logger.warning("llmfit recommend failed: %s", result["stderr"])
        return {"error": result["stderr"], "models": []}

    # llmfit outputs JSON by default for recommend
    try:
        data = json.loads(result["stdout"])
        models = data.get("models", [])[:limit]
        return {"models": models, "system": data.get("system", {})}
    except json.JSONDecodeError:
        return {"raw": result["stdout"], "models": []}


async def llmfit_system(bin_path: str) -> dict:
    """Run ``llmfit system`` and return system info."""
    result = await _run_cli(bin_path, "system")
    if result["returncode"] != 0:
        return {"error": result["stderr"]}
    # llmfit system outputs text, not JSON — return raw
    return {"raw": result["stdout"], "returncode": result["returncode"]}
