"""Emotional prosody for the VITS voices: the same expression sounds alike in English and French.

VITS exposes speed (length_scale) and variability (noise_scale, noise_w) but no pitch control. Pitch is
shifted by synthesizing a longer line, then resampling it back to its intended duration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import threading

from loguru import logger
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


DEFAULTS: dict[str, Prosody] = dict(PROSODY)
# Beyond these the voice stops sounding like her: too fast to follow, or formants audibly shifted.
BOUNDS: dict[str, tuple[float, float]] = {"speed": (0.75, 1.35), "pitch": (0.9, 1.1), "liveliness": (0.5, 1.5)}
_lock = threading.Lock()
_settings_path: Path | None = None


def prosody_for(emotion: str | None) -> Prosody:
    with _lock:
        return PROSODY.get((emotion or "").strip().lower(), PROSODY["neutral"])


def snapshot() -> dict[str, dict[str, float]]:
    with _lock:
        return {name: asdict(value) for name, value in PROSODY.items()}


def load(path: Path) -> None:
    """Apply the owner's saved tuning; later changes are written back to this file."""
    from ..core.settings_files import read_settings, settings_source

    global _settings_path
    _settings_path = path
    if not settings_source(path).exists():
        return
    try:
        saved = read_settings(path)
        for emotion, values in (saved or {}).items():
            if emotion in PROSODY and isinstance(values, dict):
                tune(emotion, values, save=False)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Voice prosody: ignoring unreadable settings: {}", exc)


def tune(emotion: str, values: dict[str, object], *, save: bool = True) -> Prosody:
    """Change one expression's delivery within BOUNDS; "reset" restores the default."""
    if emotion not in PROSODY:
        raise ValueError(f"Unknown expression: {emotion}")
    if values.get("reset") is True:
        updated = DEFAULTS[emotion]
    else:
        changes: dict[str, float] = {}
        for name, value in values.items():
            if name not in BOUNDS or isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be one of {sorted(BOUNDS)} with a number")
            low, high = BOUNDS[name]
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
            changes[name] = round(float(value), 3)
        updated = replace(PROSODY[emotion], **changes)
    with _lock:
        PROSODY[emotion] = updated
        tuned = {name: asdict(value) for name, value in PROSODY.items() if value != DEFAULTS[name]}
    if save and _settings_path is not None:
        from ..core.settings_files import write_settings

        write_settings(_settings_path, tuned)
    return updated


def shift_pitch(audio: NDArray[np.float32], factor: float) -> NDArray[np.float32]:
    """Resample audio synthesized ``factor`` times too long back to its length, raising pitch by ``factor``."""
    if factor == 1.0 or audio.size < 2:
        return audio
    flat = audio.reshape(-1)
    target = max(2, round(flat.size / factor))
    positions = np.linspace(0, flat.size - 1, target)
    shifted = np.interp(positions, np.arange(flat.size), flat).astype(np.float32)
    return shifted.reshape(-1, *audio.shape[1:]) if audio.ndim > 1 else shifted
