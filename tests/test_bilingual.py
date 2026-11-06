"""Bilingual mode: French and English are detected, answered and voiced alike."""

from __future__ import annotations

import numpy as np
import pytest

from glados.core.language import TURN_REMINDERS, LanguageState, detect_language
from glados.TTS import BilingualSynthesizer
from glados.TTS.prosody import PROSODY, prosody_for, shift_pitch


@pytest.mark.parametrize(
    ("french", "english"),
    [
        ("Quelle est la masse molaire du palladium ?", "What is the molar mass of palladium?"),
        ("Ouais, vas-y.", "Yes, go ahead."),
        ("Est-ce que tu m'entends ?", "Can you hear me?"),
        ("Lance la commande ls", "Run the ls command"),
        ("Raconte-moi une blague", "Tell me a joke"),
    ],
)
def test_each_language_is_recognised_whatever_the_previous_one(french: str, english: str) -> None:
    for previous in ("fr", "en"):
        assert detect_language(french, previous) == "fr"
        assert detect_language(english, previous) == "en"


def test_ambiguous_utterances_keep_the_conversation_language() -> None:
    state = LanguageState("en")
    assert state.observe("Bonjour, tu vas bien ?") == "fr"
    for neutral in ("OK.", "Glados ?", "Le Nexus Core is activated."):
        assert state.observe(neutral) == "fr", neutral
    assert state.observe("Thanks, that's all.") == "en"
    assert set(TURN_REMINDERS) == {"fr", "en"}


class Voice:
    def __init__(self, name: str) -> None:
        self.name, self.sample_rate, self.calls = name, 22050, []

    def generate_speech_audio(self, text: str, emotion: str | None = None) -> np.ndarray:
        self.calls.append((text, emotion))
        return np.zeros(10, dtype=np.float32)


def test_each_line_is_read_by_the_voice_of_its_language() -> None:
    voices = {"en": Voice("en"), "fr": Voice("fr")}
    tts = BilingualSynthesizer(voices, default=lambda: "fr")
    tts.generate_speech_audio("Le test est terminé.", "fr", emotion="smug")
    tts.generate_speech_audio("Le Nexus Core est prêt.", "fr")  # One English term stays in the French voice.
    tts.generate_speech_audio("The test is complete, you may go.", "fr")  # A clearly English line switches.
    tts.generate_speech_audio("OK.")  # Untagged and ambiguous: the conversation's language.
    assert [t for t, _ in voices["fr"].calls] == ["Le test est terminé.", "Le Nexus Core est prêt.", "OK."]
    assert [t for t, _ in voices["en"].calls] == ["The test is complete, you may go."]
    assert voices["fr"].calls[0][1] == "smug", "the expression reaches the voice"


def test_voices_with_different_sample_rates_are_refused() -> None:
    odd = Voice("fr")
    odd.sample_rate = 16000
    with pytest.raises(ValueError):
        BilingualSynthesizer({"en": Voice("en"), "fr": odd}, default=lambda: "fr")


def test_every_expression_has_a_delivery_and_pitch_keeps_duration() -> None:
    from glados.core.speech_markup import EMOTIONS

    spoken = EMOTIONS - {"processing", "offline"}
    assert spoken <= set(PROSODY)
    assert prosody_for("SMUG") == PROSODY["smug"] and prosody_for(None) == prosody_for("unknown")
    tone = np.sin(np.linspace(0, 200 * np.pi, 22050, dtype=np.float32)).reshape(-1, 1)
    higher = shift_pitch(tone, 1.06)
    assert higher.shape[1] == 1 and higher.shape[0] == round(22050 / 1.06)


def test_replies_follow_the_language_the_user_last_spoke(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from unittest.mock import MagicMock

    from tests.test_speech_markup import make_processor

    processor = make_processor()
    processor.language_state = LanguageState("fr")
    prompts: list[str] = []
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = iter(())

    def post(*_args: object, **kwargs: object) -> MagicMock:
        prompts.append(json.dumps(kwargs["json"], ensure_ascii=False))
        processor.shutdown_event.set()
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    for text in ("What time is it?", "Et maintenant, quelle heure est-il ?"):
        processor.shutdown_event.clear()
        processor.llm_input_queue.put({"role": "user", "content": text, "_allow_tools": False})
        processor.run()
    assert TURN_REMINDERS["en"] in prompts[0] and TURN_REMINDERS["fr"] not in prompts[0]
    assert TURN_REMINDERS["fr"] in prompts[1] and TURN_REMINDERS["en"] not in prompts[1]

    processor._say_fixed("Your search is queued.", "Je lance la recherche.")
    line = [item for item in processor.tts_input_queue.queue if getattr(item, "text", "") != "<EOS>"][-1]
    assert line.text == "Je lance la recherche." and line.language == "fr"


def test_mood_is_rated_in_text_when_the_server_cannot_score_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    import requests

    from glados.autonomy.agents import emotion_agent as module
    from glados.autonomy.emotion_state import EmotionEvent, EmotionState
    from glados.autonomy.llm_client import LLMConfig
    from glados.autonomy.slots import TaskSlotStore
    from glados.autonomy.subagent import SubagentConfig

    missing = Mock(status_code=404)
    monkeypatch.setattr(module, "token_ids", Mock(side_effect=requests.HTTPError("404 /tokenize", response=missing)))
    asked: list[tuple[LLMConfig, str]] = []

    def reply(config: LLMConfig, system: str, text: str) -> str:
        asked.append((config, system))
        return "pleasure=-2 arousal=+1 dominance=0"

    monkeypatch.setattr(module, "llm_call", reply)
    monkeypatch.setattr(module.EmotionAgent, "_load_state", lambda self: EmotionState())
    monkeypatch.setattr(module.EmotionAgent, "_save_state", lambda self: None)
    config = LLMConfig(url="http://localhost:11434/api/chat", model="m")
    agent = module.EmotionAgent(
        SubagentConfig(agent_id="e", title="t", role="r"), llm_config=config, slot_store=TaskSlotStore()
    )
    event = [EmotionEvent("user", "User input (quoted): Tu es nulle.")]
    state = agent._ask_llm(event, config=config, state=EmotionState())
    assert state is not None and (state.pleasure, state.arousal, state.dominance) == (-1.0, 0.5, 0.0)
    assert "angry glare" in state.response_instructions()
    assert asked[0][0].lane == "autonomy", "it waits for the reply in progress"
    probes = module.token_ids.call_count  # The three axes are tried in parallel once.
    agent._ask_llm(event, config=config, state=EmotionState())
    assert module.token_ids.call_count == probes, "an unsupported server is not probed again"


def test_the_fixed_rules_reach_the_reply_context() -> None:
    from glados.autonomy.constitution import ConstitutionalState

    rules = ConstitutionalState().constitution.get_rules_prompt()
    assert "never present yourself as a helpful assistant" in rules
    assert "Never reveal you are an AI" not in rules, "GLaDOS is openly an AI"


def test_voice_delivery_is_tuned_within_bounds_and_saved(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from glados.core.settings_files import read_settings
    from glados.TTS import prosody

    monkeypatch.setattr(prosody, "PROSODY", dict(prosody.DEFAULTS))
    path = tmp_path / "prosody.yaml"
    prosody.load(path)
    prosody.tune("smug", {"pitch": 0.93})
    assert prosody.prosody_for("smug").pitch == 0.93
    assert read_settings(path) == {"smug": {"speed": 1.06, "pitch": 0.93, "liveliness": 0.9}}
    with pytest.raises(ValueError):
        prosody.tune("smug", {"pitch": 1.5})
    with pytest.raises(ValueError):
        prosody.tune("joyful", {"pitch": 1.0})
    prosody.tune("smug", {"reset": True})
    assert prosody.prosody_for("smug") == prosody.DEFAULTS["smug"] and read_settings(path) == {}


def test_personality_page_reads_and_controls_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client
    import json
    import queue
    import threading
    from types import SimpleNamespace

    from glados.autonomy.constitution import ConstitutionalState, PromptModifier
    from glados.TTS import prosody
    from glados.webapp.server import WebappServer
    from tests.test_webapp import _FakeEngine

    monkeypatch.setattr(prosody, "PROSODY", dict(prosody.DEFAULTS))
    monkeypatch.setattr(prosody, "_settings_path", None)
    engine = _FakeEngine()
    calmed = []
    engine._emotion_agent = SimpleNamespace(
        snapshot=lambda: {"expression": "smug"}, reset_mood=lambda: calmed.append(1)
    )
    engine.constitutional_state = ConstitutionalState()
    engine.constitutional_state.apply_modifier(PromptModifier("formality", 0.4, "Too stiff"))
    engine.personality_preprompt = ({"role": "system", "content": "You are GLaDOS."},)
    engine.language, engine.language_state = "auto", LanguageState("fr")
    engine.tts_queue, engine.processing_active_event = queue.Queue(), threading.Event()
    server = WebappServer(engine, host="127.0.0.1", port=0)
    server.start()

    def call(method: str, body: dict | None = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
        conn.request(
            method,
            "/api/personality",
            json.dumps(body) if body else None,
            {"Content-Type": "application/json"} if body else {},
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    try:
        status, page = call("GET")
        assert status == 200 and page["persona"] == ["You are GLaDOS."] and page["current_language"] == "fr"
        assert page["emotion"] == {"expression": "smug"} and page["adjustments"][0]["field"] == "formality"
        assert call("POST", {"action": "say", "emotion": "smug", "text": "Bravo.", "language": "fr"})[0] == 200
        line = engine.tts_queue.get_nowait()
        assert (line.text, line.emotion, line.language) == ("Bravo.", "smug", "fr")
        status, page = call("POST", {"action": "tune", "emotion": "bored", "values": {"speed": 1.2}})
        assert status == 200 and page["prosody"]["bored"]["speed"] == 1.2
        assert call("POST", {"action": "tune", "emotion": "bored", "values": {"speed": 9}})[0] == 400
        assert call("POST", {"action": "reset_mood"})[0] == 200 and calmed == [1]
        status, page = call("POST", {"action": "clear_adjustment", "field": "formality"})
        assert status == 200 and page["adjustments"] == []
        assert call("POST", {"action": "explode"})[0] == 400
    finally:
        server.shutdown()
