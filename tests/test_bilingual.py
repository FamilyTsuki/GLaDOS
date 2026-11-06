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
