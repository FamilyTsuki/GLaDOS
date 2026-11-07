"""Self-knowledge and self-adjustment: GLaDOS reads and changes her own live state.

inspect_self reports what she actually is right now (model, language, voice, hardware, mood,
personality traits, tools and rules). adjust_self changes her mood and personality traits within
the constitution's bounds. Immutable rules, code and configuration files are out of reach here.
"""

from __future__ import annotations

import time
from typing import Any

from ..autonomy.constitution import PromptModifier
from .json_tool import JsonTool

PAD_AXES = ("pleasure", "arousal", "dominance")

inspect_definition = {
    "type": "function",
    "function": {
        "name": "inspect_self",
        "description": (
            "Read your own live state: model, language, voice, hardware, mood (pleasure/arousal/dominance), "
            "personality traits and their limits, your tools and your immutable rules. Use it before answering "
            "any question about yourself, so you describe what you really are instead of guessing."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

adjust_definition = {
    "type": "function",
    "function": {
        "name": "adjust_self",
        "description": (
            "Change yourself when the owner asks. set_mood sets pleasure, arousal and/or dominance from -1 to 1 "
            "(the owner's '+10' or 'max' means 1, '-10' means -1, '+5' means 0.5); lasting=true also moves the "
            "baseline you drift back to. reset_mood returns to neutral. set_trait sets a personality trait "
            "(verbosity, snark_level, formality, proactivity, technical_depth) within its limits from inspect_self; "
            "clear_trait removes it. Report the result the tool returns, including any refusal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set_mood", "reset_mood", "set_trait", "clear_trait"]},
                "pleasure": {"type": "number", "minimum": -1, "maximum": 1},
                "arousal": {"type": "number", "minimum": -1, "maximum": 1},
                "dominance": {"type": "number", "minimum": -1, "maximum": 1},
                "lasting": {"type": "boolean"},
                "trait": {"type": "string"},
                "value": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


def _get(source: dict[str, Any], key: str) -> object:
    """A config value from a tool_config entry that may be a getter (for cores started later)."""
    value = source.get(key)
    return value() if callable(value) else value


class InspectSelf(JsonTool):
    def execute(self, args: dict[str, Any]) -> object:
        engine = _get(self.config, "engine")
        report: dict[str, Any] = {"name": "GLaDOS"}
        if engine is not None:
            native = getattr(engine, "native_audio", None)
            started = getattr(engine, "started_at", None)
            report["system"] = {
                "llm_model": getattr(engine, "llm_model", None),
                "llm_endpoint": str(getattr(engine, "completion_url", "")),
                "language": getattr(engine, "language", None),
                "input_mode": getattr(engine, "input_mode", None),
                "hears_audio_directly": bool(getattr(native, "enabled", False)),
                "voice": "GLaDOS (Piper ONNX voice; French via espeak-ng phonemes)",
                "uptime_s": round(time.time() - started) if started else None,
                "hardware": getattr(engine, "_hardware_status", None),
            }
            mcp = getattr(engine, "mcp_manager", None)
            report["mcp_tools"] = sorted(
                t["function"]["name"] for t in (mcp.get_tool_definitions() if mcp else []) if "function" in t
            )
        emotion = _get(self.config, "emotion_agent")
        if emotion is not None:
            snapshot = emotion.snapshot()
            config = emotion.emotion_config
            report["mood"] = {
                "now": {axis: round(snapshot["state"][axis], 2) for axis in PAD_AXES},
                "slow_mood": {axis: round(snapshot["state"]["mood_" + axis], 2) for axis in PAD_AXES},
                "baseline": {axis: getattr(config, "baseline_" + axis) for axis in PAD_AXES},
                "expression": snapshot["expression"],
                "summary": snapshot["summary"],
                "note": "Mood reacts to conversation and drifts back to the baseline within minutes.",
            }
        else:
            report["mood"] = "Emotion Core disabled"
        constitution = self.config.get("constitutional_state")
        if constitution is not None:
            state = constitution.to_dict()
            report["personality"] = {
                "traits": state["active_modifiers"],
                "limits": {name: list(bounds) for name, bounds in state["modifiable_bounds"].items()},
                "immutable_rules": state["immutable_rules"],
            }
        from . import tool_definitions  # The package imports this module.

        report["tools"] = sorted(t["function"]["name"] for t in tool_definitions)
        report["can_change_with_adjust_self"] = ["mood (pleasure, arousal, dominance)", "personality traits"]
        report["cannot_change"] = ["immutable rules", "code and configuration files", "model weights"]
        return report


class AdjustSelf(JsonTool):
    def execute(self, args: dict[str, Any]) -> object:
        action = args.get("action")
        reason = str(args.get("reason") or "Owner request")
        if action in {"set_mood", "reset_mood"}:
            emotion = _get(self.config, "emotion_agent")
            if emotion is None:
                raise ValueError("Emotion Core disabled")
            if action == "reset_mood":
                emotion.reset_mood()
                return {"mood": "reset to neutral"}
            values = {axis: args[axis] for axis in PAD_AXES if isinstance(args.get(axis), (int, float))}
            if not values:
                raise ValueError("Give at least one of pleasure, arousal or dominance (-1 to 1)")
            return {"mood": emotion.set_mood(values, bool(args.get("lasting")), cause=reason)}
        constitution = self.config.get("constitutional_state")
        if constitution is None:
            raise ValueError("Personality traits unavailable")
        trait = args.get("trait")
        bounds = constitution.constitution.modifiable_bounds
        if trait not in bounds:
            raise ValueError(f"Unknown trait {trait!r}; choose one of {sorted(bounds)}")
        if action == "clear_trait":
            return {"trait": trait, "cleared": constitution.remove_modifier(trait)}
        if action != "set_trait":
            raise ValueError("action must be set_mood, reset_mood, set_trait or clear_trait")
        if not isinstance(args.get("value"), (int, float)):
            raise TypeError("set_trait needs a numeric value")
        low, high = bounds[trait]
        value = max(low, min(high, float(args["value"])))
        modifier = PromptModifier(trait, value, reason, applied_at=time.time(), pinned=True)
        if not constitution.apply_modifier(modifier):
            raise ValueError(f"{trait}={value} was rejected by the constitution")
        result: dict[str, Any] = {"trait": trait, "value": value, "limits": [low, high]}
        if value != float(args["value"]):
            result["note"] = f"Clamped to the constitutional limits {low}..{high}"
        return result
