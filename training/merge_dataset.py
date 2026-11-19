"""Merge training/parts/*.jsonl into training/glados_dataset.jsonl with quality filters."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from generate_dataset import TOOLS

HERE = Path(__file__).parent
CHATBOT = re.compile(
    r"si (vous le )?(voulez|souhaitez)|si tu veux|bonne question|great question|n'hésitez|"
    r"do you want me|je peux (aussi )?(vous|t')|let me know|feel free|^(ah|oh)\b|^glados\s*:",
    re.I,
)
TU = re.compile(r"\b(tu|toi|ton|ta|tes|t'|te)\b", re.I)  # GLaDOS must use "vous" in French
FR = re.compile(r"\b(le|la|les|vous|est|une|des|pas|que)\b", re.I)


def reason(ex: dict) -> str | None:
    msgs = ex["messages"]
    final = msgs[-1]["content"]
    prose = re.sub(r"```.*?```", "", final, flags=re.S)
    if CHATBOT.search(prose.strip()):
        return "chatbot"
    if len(FR.findall(prose)) >= 2 and TU.search(prose):
        return "tutoiement"
    if len(prose.split()) > 70:
        return "trop long"
    for m in msgs:
        for c in m.get("tool_calls") or []:
            fn = c.get("function", {})
            if fn.get("name") not in TOOLS:
                return "outil inconnu"
            try:
                json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                return "arguments invalides"
    return None


def main() -> None:
    seen: set[str] = set()
    kept: list[dict] = []
    dropped: Counter[str] = Counter()
    for part in sorted((HERE / "parts").glob("*.jsonl")):
        for line in part.open(encoding="utf-8"):
            ex = json.loads(line)
            q = re.sub(r"\W+", " ", ex["messages"][0]["content"].lower()).strip()
            if q in seen:
                dropped["doublon"] += 1
                continue
            seen.add(q)
            r = reason(ex)
            if r:
                dropped[r] += 1
                continue
            kept.append(ex)
    random.seed(0)
    random.shuffle(kept)
    with (HERE / "glados_dataset.jsonl").open("w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tools = sum(any(m.get("tool_calls") for m in ex["messages"]) for ex in kept)
    print(f"gardés {len(kept)} (dont {tools} avec outils) | rejetés {dict(dropped)}")


if __name__ == "__main__":
    main()
