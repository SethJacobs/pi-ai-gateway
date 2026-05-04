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
    result = await _run_cli(bin_path, "recommend", "--json")
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


async def llmfit_download(bin_path: str, model: str) -> dict:
    """Run ``llmfit download <model>`` and stream stdout until complete.

    llmfit download does not support --json, so we parse stdout for the
    downloaded file path which it prints on the last line as an absolute path.
    """
    logger.info("Downloading model via llmfit: %s", model)
    proc = await asyncio.create_subprocess_exec(
        bin_path, "download", model,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = stdout.decode().strip()
    err = stderr.decode().strip()

    if proc.returncode != 0:
        logger.warning("llmfit download failed: %s", err or out)
        return {"status": "error", "error": err or out}

    # Extract path from "Saved to: /path/to/file.gguf" line
    file_path = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Saved to:"):
            file_path = line.split("Saved to:", 1)[1].strip()
            break

    if not file_path:
        logger.warning("Could not parse file path from llmfit output: %s", out)
        return {"status": "error", "error": "Could not determine downloaded file path"}

    logger.info("llmfit download complete: %s", file_path)
    return {"status": "ok", "path": file_path, "output": out}
