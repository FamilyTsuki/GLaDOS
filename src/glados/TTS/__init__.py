"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


FRENCH_VOICE = "models/TTS/fr_FR-glados-medium.onnx"


class BilingualSynthesizer:
    """GLaDOS's English and French voices; each line is read by the voice of its language."""

    language = "auto"

    def __init__(self, voices: dict[str, SpeechSynthesizerProtocol], default: Callable[[], str]) -> None:
        rates = {voice.sample_rate for voice in voices.values()}
        if len(rates) != 1:
            raise ValueError(f"Bilingual voices must share a sample rate, got {sorted(rates)}")
        self.voices = voices
        self.sample_rate = rates.pop()
        self._default = default

    def language_of(self, text: str, hint: str | None = None) -> str:
        """The tagged reply language, unless the line itself is clearly in the other one."""
        from ..core.language import detect_language

        fallback = hint if hint in self.voices else self._default()
        # A margin of 2 keeps a French reply quoting one English term in the French voice.
        return detect_language(text, fallback, margin=2)  # type: ignore[arg-type]

    def generate_speech_audio(
        self, text: str, language: str | None = None, emotion: str | None = None
    ) -> NDArray[np.float32]:
        voice = self.voices[self.language_of(text, language)]
        return voice.generate_speech_audio(text, emotion=emotion)  # type: ignore[call-arg]


# Factory function


def get_speech_synthesizer(
    voice: str = "glados",
    language: str = "en",
    current_language: Callable[[], str] = lambda: "en",
) -> SpeechSynthesizerProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.
    Parameters:
        voice (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() == "glados":
        from ..TTS import tts_glados
        from ..utils.resources import resource_path

        if language == "auto":
            return BilingualSynthesizer(
                {
                    "en": tts_glados.SpeechSynthesizer(language="en"),
                    "fr": tts_glados.SpeechSynthesizer(model_path=resource_path(FRENCH_VOICE), language="fr"),
                },
                current_language,
            )
        if language == "fr":
            # The GLaDOS model only knows English and gives French a strong accent: use GLaDOS's
            # French voice, a Piper model trained on her French lines.
            return tts_glados.SpeechSynthesizer(model_path=resource_path(FRENCH_VOICE), language="fr")
        return tts_glados.SpeechSynthesizer(language=language)

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["BilingualSynthesizer", "SpeechSynthesizerProtocol", "get_speech_synthesizer"]
