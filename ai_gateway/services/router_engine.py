from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import GatewaySettings
    from ..schemas import ChatCompletionRequest, ChatMessage
    from .model_registry import ModelRegistry
    from .system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class Route(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"


class Intent(str, Enum):
    CODING = "coding"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    QUICK_QA = "quick_qa"
    TRANSLATION = "translation"
    GENERAL = "general"


# Intent -> preferred route
INTENT_ROUTES: dict[Intent, Route] = {
    Intent.CODING: Route.CLOUD,
    Intent.ANALYSIS: Route.CLOUD,
    Intent.CREATIVE: Route.CLOUD,
    Intent.QUICK_QA: Route.LOCAL,
    Intent.TRANSLATION: Route.LOCAL,
    Intent.GENERAL: Route.CLOUD,
}

# Compiled keyword patterns for intent classification
_PATTERNS: dict[Intent, re.Pattern[str]] = {
    Intent.CODING: re.compile(
        r"\b(code|program|function|class|debug|refactor|implement|api|sql|"
        r"python|javascript|typescript|rust|golang|java|html|css|regex|git)\b",
        re.IGNORECASE,
    ),
    Intent.ANALYSIS: re.compile(
        r"\b(analyze|explain|compare|review|evaluate|assess|critique|"
        r"how does|why does|what causes|difference between)\b",
        re.IGNORECASE,
    ),
    Intent.CREATIVE: re.compile(
        r"\b(write|essay|story|poem|article|blog|draft|compose|rewrite)\b",
        re.IGNORECASE,
    ),
    Intent.QUICK_QA: re.compile(
        r"\b(what is|who is|when did|how many|define|meaning of|"
        r"quick|brief|tldr|summarize|summary)\b",
        re.IGNORECASE,
    ),
    Intent.TRANSLATION: re.compile(
        r"\b(translate|translation|in spanish|in french|in german|"
        r"in japanese|in chinese|to english)\b",
        re.IGNORECASE,
    ),
}


@dataclass
class RoutingDecision:
    route: Route
    intent: Intent
    model: str
    reason: str
    fallback_route: Route | None = None


class RouterEngine:
    def __init__(
        self,
        config: GatewaySettings,
        system_monitor: SystemMonitor,
        model_registry: ModelRegistry,
    ) -> None:
        self.config = config
        self.monitor = system_monitor
        self.registry = model_registry

    def classify_intent(self, messages: list[ChatMessage]) -> Intent:
        """Classify intent from the last user message."""
        last_msg = ""
        for m in reversed(messages):
            if m.role == "user":
                last_msg = m.content
                break

        if not last_msg:
            return Intent.GENERAL

        # Score each intent by keyword matches
        scores: dict[Intent, int] = {}
        for intent, pattern in _PATTERNS.items():
            matches = pattern.findall(last_msg)
            if matches:
                scores[intent] = len(matches)

        if not scores:
            # Short messages default to quick Q&A
            if len(last_msg.split()) < 15:
                return Intent.QUICK_QA
            return Intent.GENERAL

        return max(scores, key=lambda k: scores[k])

    async def decide(self, request: ChatCompletionRequest) -> RoutingDecision:
        """Make a routing decision based on intent + system state."""

        # Explicit route override from request
        if request.route == "cloud":
            model = request.model or await self.registry.best_cloud_model()
            return RoutingDecision(Route.CLOUD, Intent.GENERAL, model, "explicit_cloud")

        if request.route == "local":
            model = request.model or await self.registry.best_local_model()
            return RoutingDecision(Route.LOCAL, Intent.GENERAL, model, "explicit_local")

        # Auto routing: classify intent
        intent = self.classify_intent(request.messages)
        preferred = INTENT_ROUTES.get(intent, Route.CLOUD)

        # Check local viability
        system_info = await self.monitor.get_system_info()
        free_ram_mb = system_info.get("available_ram_mb", 0)
        local_loaded = await self.monitor.is_local_model_loaded()
        local_viable = free_ram_mb > self.config.local_ram_threshold_mb and local_loaded

        if preferred == Route.LOCAL and local_viable:
            model = request.model or await self.registry.current_local_model()
            return RoutingDecision(
                route=Route.LOCAL,
                intent=intent,
                model=model,
                reason=f"intent={intent.value} free_ram={free_ram_mb}MB local_loaded=True",
                fallback_route=Route.CLOUD,
            )

        # Default to cloud, always allow local fallback if a model is loaded
        # (even if RAM is below threshold, a loaded model can still serve requests)
        model = request.model or await self.registry.best_cloud_model()
        return RoutingDecision(
            route=Route.CLOUD,
            intent=intent,
            model=model,
            reason=f"intent={intent.value} local_viable={local_viable}",
            fallback_route=Route.LOCAL if local_loaded else None,
        )
