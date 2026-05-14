from __future__ import annotations

from pydantic_settings import BaseSettings


class BridgeSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9099

    freeride_bin: str = "freeride"
    llmfit_bin: str = "llmfit"
    llama_server_bin: str = "llama-server"
    llama_server_port: int = 8188
    model_dir: str = "/home/pi/.cache/llmfit/models"
    max_context_size: int = 2048
    llama_threads: int = 4
    freeride_refresh_interval_hours: int = 24

    model_config = {"env_prefix": "BRIDGE_"}
