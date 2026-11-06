"""Emotional prosody for the VITS voices: the same expression sounds alike in English and French.

VITS exposes speed (length_scale) and variability (noise_scale, noise_w) but no pitch control. Pitch is
shifted by synthesizing a longer line, then resampling it back to its intended duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Prosody:
    speed: float = 1.0
    """Multiplies length_scale: above 1 is slower."""
    pitch: float = 1.0
    """Above 1 is higher. Kept within a few percent: further shifts move the formants audibly."""
    liveliness: float = 1.0
    """Multiplies noise_scale and noise_w: above 1 is more varied, below 1 flatter."""


NEUTRAL = Prosody()
PROSODY: dict[str, Prosody] = {
    "neutral": NEUTRAL,
    "smug": Prosody(speed=1.06, pitch=0.97, liveliness=0.9),
    "angry glare": Prosody(speed=0.93, pitch=0.96, liveliness=1.2),
    "suspicious": Prosody(speed=1.08, pitch=0.98, liveliness=0.85),
    "surprised": Prosody(speed=0.94, pitch=1.06, liveliness=1.25),
    "quizzical": Prosody(speed=1.0, pitch=1.03, liveliness=1.1),
    "bored": Prosody(speed=1.15, pitch=0.96, liveliness=0.7),
    "disappointed": Prosody(speed=1.12, pitch=0.95, liveliness=0.75),
}


def prosody_for(emotion: str | None) -> Prosody:
    return PROSODY.get((emotion or "").strip().lower(), NEUTRAL)


def shift_pitch(audio: NDArray[np.float32], factor: float) -> NDArray[np.float32]:
    """Resample audio synthesized ``factor`` times too long back to its length, raising pitch by ``factor``."""
    if factor == 1.0 or audio.size < 2:
        return audio
    flat = audio.reshape(-1)
    target = max(2, round(flat.size / factor))
    positions = np.linspace(0, flat.size - 1, target)
    shifted = np.interp(positions, np.arange(flat.size), flat).astype(np.float32)
    return shifted.reshape(-1, *audio.shape[1:]) if audio.ndim > 1 else shifted
