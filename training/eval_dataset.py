"""LLM-judge every example of glados_dataset.jsonl; writes eval_scores.jsonl (line index -> scores)."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

from generate_dataset import chat, env_key

HERE = Path(__file__).parent
JUDGE = """Tu évalues des exemples d'entraînement pour un assistant vocal incarnant GLaDOS
(IA froide, sarcastique, passive-agressive, condescendante, mais réellement utile ; vouvoie en français).
Pour chaque exemple, note de 0 à 3 :
- persona : 0 = assistant poli générique, 1 = vaguement piquant, 2 = GLaDOS crédible, 3 = excellent, mordant et original
- utile : 0 = n'aide pas ou faux, 1 = partiel, 2 = correct, 3 = correct, précis et concis
- langue : 0 = mauvaise langue ou très maladroit, 1 = maladroit, 2 = correct, 3 = naturel
- outil : 0 = mauvais outil ou outil manquant/inutile, 3 = choix d'outil correct (ou aucun outil et aucun requis)
- defaut : "" ou un défaut principal très court (ex: "trop gentil", "fait faux", "réplique de Portal", "trop long")
Réponds UNIQUEMENT en JSON : [{{"id": n, "persona": n, "utile": n, "langue": n, "outil": n, "defaut": "..."}}, ...]

Exemples :
{items}"""


def render(i: int, ex: dict) -> str:
    parts = []
    for m in ex["messages"]:
        if m.get("tool_calls"):
            parts.append("APPEL " + ", ".join(
                f"{c['function']['name']}({c['function'].get('arguments', '')})" for c in m["tool_calls"]))
        elif m["role"] == "tool":
            parts.append(f"RÉSULTAT: {m['content'][:200]}")
        else:
            parts.append(f"{m['role'].upper()}: {m['content']}")
    return f"### id={i}\n" + "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--url", default="https://api.openai.com/v1/chat/completions")
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    key = env_key("OPENAI_API_KEY")
    data = [json.loads(l) for l in (HERE / "glados_dataset.jsonl").open(encoding="utf-8")]
    batches = [list(range(s, min(s + a.batch, len(data)))) for s in range(0, len(data), a.batch)]
    out = HERE / "eval_scores.jsonl"
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [b for b in batches if not set(b) <= done]
    cost = 0.0

    def run(ids: list[int]) -> tuple[list[dict], float]:
        prompt = JUDGE.format(items="\n\n".join(render(i, data[i]) for i in ids))
        raw, u = chat(a.url, a.model, key, prompt, "low")
        c = (u.get("prompt_tokens", 0) * 0.25 + u.get("completion_tokens", 0) * 2.0) / 1e6
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M)
        return json.loads(raw[raw.find("["): raw.rfind("]") + 1]), c

    with cf.ThreadPoolExecutor(a.workers) as pool, out.open("a") as f:
        for fut in cf.as_completed([pool.submit(run, b) for b in todo]):
            try:
                rows, c = fut.result()
            except Exception as e:  # noqa: BLE001 - rerun the script to retry failed batches
                print(f"skip: {e}", file=sys.stderr)
                continue
            cost += c
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"cost ${cost:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
