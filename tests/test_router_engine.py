from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ai_gateway.schemas import ChatCompletionRequest, ChatMessage
from ai_gateway.services.router_engine import Intent, Route, RouterEngine


@pytest.fixture
def mock_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.get_system_info.return_value = {"available_ram_mb": 3000, "cpu_cores": 4}
    monitor.is_local_model_loaded.return_value = False
    return monitor


@pytest.fixture
def mock_registry() -> AsyncMock:
    registry = AsyncMock()
    registry.best_cloud_model.return_value = "openrouter/auto"
    registry.best_local_model.return_value = "phi-tiny-moe"
    registry.current_local_model.return_value = "phi-tiny-moe"
    return registry


@pytest.fixture
def engine(settings, mock_monitor, mock_registry) -> RouterEngine:
    return RouterEngine(config=settings, system_monitor=mock_monitor, model_registry=mock_registry)


class TestIntentClassification:
    def test_coding_intent(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(role="user", content="Write a python function to sort a list")]
        assert engine.classify_intent(messages) == Intent.CODING

    def test_analysis_intent(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(role="user", content="Explain how DNS resolution works")]
        assert engine.classify_intent(messages) == Intent.ANALYSIS

    def test_creative_intent(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(role="user", content="Write a blog post about home servers")]
        assert engine.classify_intent(messages) == Intent.CREATIVE

    def test_quick_qa_intent(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(role="user", content="What is Docker?")]
        assert engine.classify_intent(messages) == Intent.QUICK_QA

    def test_translation_intent(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(role="user", content="Translate this to spanish: hello world")]
        assert engine.classify_intent(messages) == Intent.TRANSLATION

    def test_short_message_defaults_quick_qa(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(role="user", content="hello")]
        assert engine.classify_intent(messages) == Intent.QUICK_QA

    def test_long_unknown_defaults_general(self, engine: RouterEngine) -> None:
        messages = [ChatMessage(
            role="user",
            content="I was thinking about the various implications of the current market "
            "situation and how it might affect long term planning for infrastructure",
        )]
        assert engine.classify_intent(messages) == Intent.GENERAL

    def test_empty_messages(self, engine: RouterEngine) -> None:
        assert engine.classify_intent([]) == Intent.GENERAL


class TestRoutingDecisions:
    @pytest.mark.asyncio
    async def test_explicit_cloud_route(self, engine: RouterEngine) -> None:
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            route="cloud",
        )
        decision = await engine.decide(req)
        assert decision.route == Route.CLOUD
        assert decision.reason == "explicit_cloud"

    @pytest.mark.asyncio
    async def test_explicit_local_route(self, engine: RouterEngine) -> None:
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            route="local",
        )
        decision = await engine.decide(req)
        assert decision.route == Route.LOCAL
        assert decision.reason == "explicit_local"

    @pytest.mark.asyncio
    async def test_coding_routes_to_cloud(self, engine: RouterEngine) -> None:
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Debug this python function")],
        )
        decision = await engine.decide(req)
        assert decision.route == Route.CLOUD
        assert decision.intent == Intent.CODING

    @pytest.mark.asyncio
    async def test_quick_qa_routes_local_when_loaded(
        self, engine: RouterEngine, mock_monitor: AsyncMock
    ) -> None:
        mock_monitor.is_local_model_loaded.return_value = True
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What is nginx?")],
        )
        decision = await engine.decide(req)
        assert decision.route == Route.LOCAL
        assert decision.intent == Intent.QUICK_QA
        assert decision.fallback_route == Route.CLOUD

    @pytest.mark.asyncio
    async def test_quick_qa_falls_to_cloud_when_not_loaded(
        self, engine: RouterEngine, mock_monitor: AsyncMock
    ) -> None:
        mock_monitor.is_local_model_loaded.return_value = False
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What is nginx?")],
        )
        decision = await engine.decide(req)
        assert decision.route == Route.CLOUD
        assert decision.fallback_route is None

    @pytest.mark.asyncio
    async def test_low_ram_forces_cloud(
        self, engine: RouterEngine, mock_monitor: AsyncMock
    ) -> None:
        mock_monitor.get_system_info.return_value = {"available_ram_mb": 500}
        mock_monitor.is_local_model_loaded.return_value = True
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What is Docker?")],
        )
        decision = await engine.decide(req)
        assert decision.route == Route.CLOUD
