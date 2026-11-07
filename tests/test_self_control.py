"""inspect_self / adjust_self: GLaDOS reads and changes her own mood and personality."""

import json
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from glados.autonomy.agents.emotion_agent import EmotionAgent
from glados.autonomy.constitution import ConstitutionalState, PromptModifier
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.subagent import SubagentConfig
from glados.core.routing_tree import LOCAL_CATEGORIES
from glados.tools import tool_classes, tool_definitions
from glados.tools.self_control import AdjustSelf, InspectSelf


@pytest.fixture
def emotion(monkeypatch: pytest.MonkeyPatch) -> EmotionAgent:
    memory = Mock()
    memory.get.return_value = None
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock(return_value=memory))
    return EmotionAgent(SubagentConfig("emotion", "Emotion"), LLMConfig("http://test"), slot_store=Mock())


def call(tool_cls: type, config: dict, **args: object) -> dict:
    sink: queue.Queue = queue.Queue()
    tool_cls(sink, config).run("call-1", args)
    return json.loads(sink.get_nowait()["content"])


def test_tools_are_registered_and_routed() -> None:
    names = {t["function"]["name"] for t in tool_definitions}
    assert {"inspect_self", "adjust_self"} <= names
    assert tool_classes["inspect_self"] is InspectSelf and tool_classes["adjust_self"] is AdjustSelf
    assert LOCAL_CATEGORIES["inspect_self"] == LOCAL_CATEGORIES["adjust_self"] == "self"


def test_set_mood_changes_state_and_lasting_moves_baseline(emotion: EmotionAgent) -> None:
    config = {"emotion_agent": lambda: emotion}
    result = call(AdjustSelf, config, action="set_mood", dominance=1.0)
    assert result["mood"]["state"]["dominance"] == 1.0
    assert result["mood"]["baseline"]["dominance"] == 0.0, "temporary by default"
    assert emotion.state.mood_dominance == 1.0

    result = call(AdjustSelf, config, action="set_mood", dominance=5, pleasure=-0.4, lasting=True)
    assert result["mood"]["state"] == {"pleasure": -0.4, "arousal": 0.0, "dominance": 1.0}, "clamped to 1"
    assert emotion.emotion_config.baseline_dominance == 1.0

    assert "error" in call(AdjustSelf, config, action="set_mood")


def test_trait_is_bounded_and_pinned_against_the_observer() -> None:
    state = ConstitutionalState()
    config = {"constitutional_state": state}
    result = call(AdjustSelf, config, action="set_trait", trait="snark_level", value=0.1)
    assert result["value"] == 0.3 and "note" in result, "snark never drops below the character minimum"

    call(AdjustSelf, config, action="set_trait", trait="verbosity", value=0.9)
    assert not state.apply_modifier(PromptModifier("verbosity", 0.2, "observer")), "owner choice wins"
    assert state.active_modifiers["verbosity"].value == 0.9

    assert call(AdjustSelf, config, action="clear_trait", trait="verbosity") == {"trait": "verbosity", "cleared": True}
    assert "error" in call(AdjustSelf, config, action="set_trait", trait="immutable_rules", value=1)


def test_inspect_self_reports_live_setup(emotion: EmotionAgent) -> None:
    engine = SimpleNamespace(
        llm_model="gemma-4-E4B",
        completion_url="http://127.0.0.1:18080/v1/chat/completions",
        language="fr",
        input_mode="audio",
        native_audio=SimpleNamespace(enabled=True),
        started_at=None,
        _hardware_status={"tier": "HIGH"},
        mcp_manager=None,
    )
    emotion.set_mood({"dominance": 0.5})
    report = call(
        InspectSelf,
        {"engine": engine, "emotion_agent": lambda: emotion, "constitutional_state": ConstitutionalState()},
    )
    assert report["system"]["llm_model"] == "gemma-4-E4B" and report["system"]["language"] == "fr"
    assert report["mood"]["now"]["dominance"] == 0.5
    assert report["personality"]["limits"]["snark_level"] == [0.3, 1.0]
    assert "adjust_self" in report["tools"]
