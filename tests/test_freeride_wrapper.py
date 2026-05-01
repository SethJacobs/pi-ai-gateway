from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from model_bridge.freeride_wrapper import freeride_auto, freeride_list, freeride_status


class MockProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.returncode = returncode
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_freeride_list_success() -> None:
    mock_proc = MockProcess(stdout="1. model/name:free (score: 95)\n2. other/model:free (score: 90)")

    with patch("model_bridge.freeride_wrapper.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await freeride_list("freeride")

    assert result["returncode"] == 0
    assert "model/name:free" in result["raw"]


@pytest.mark.asyncio
async def test_freeride_list_failure() -> None:
    mock_proc = MockProcess(stdout="", stderr="command not found", returncode=1)

    with patch("model_bridge.freeride_wrapper.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await freeride_list("freeride")

    assert "error" in result


@pytest.mark.asyncio
async def test_freeride_status() -> None:
    mock_proc = MockProcess(stdout="Primary: qwen/qwen3-coder:free\nFallbacks: 5 configured")

    with patch("model_bridge.freeride_wrapper.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await freeride_status("freeride")

    assert result["returncode"] == 0
    assert "Primary" in result["raw"]


@pytest.mark.asyncio
async def test_freeride_auto() -> None:
    mock_proc = MockProcess(stdout="Auto-configured: qwen/qwen3-coder:free with 5 fallbacks")

    with patch("model_bridge.freeride_wrapper.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await freeride_auto("freeride", fallback_count=5)

    assert result["returncode"] == 0
    assert "Auto-configured" in result["raw"]
