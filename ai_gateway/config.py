from __future__ import annotations

from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080

    # External services
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_bridge_url: str = "http://host.docker.internal:9099"

    # Default cloud model (overridden by freeride discovery)
    default_cloud_model: str = "openrouter/auto"

    # Routing thresholds
    local_ram_threshold_mb: int = 2048

    # API security — optional bearer token
    gateway_api_key: str = ""

    model_config = {"env_prefix": "GATEWAY_"}
