from __future__ import annotations

import pytest

from ai_gateway.config import GatewaySettings


@pytest.fixture
def settings() -> GatewaySettings:
    return GatewaySettings(
        openrouter_api_key="test-key",
        model_bridge_url="http://localhost:9099",
        default_cloud_model="openrouter/auto",
    )
