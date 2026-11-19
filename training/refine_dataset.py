"""Make glados_dataset.jsonl sharper: rewrite weak-persona replies, re-judge them, keep the best version.

Inputs : glados_dataset.jsonl + eval_scores.jsonl (from eval_dataset.py)
Outputs: rewrites.jsonl, rewrite_scores.jsonl (resumable caches), glados_dataset_final.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import re
import sys
from pathlib import Path

from eval_dataset import JUDGE, render
from generate_dataset import chat, env_key

HERE = Path(__file__).parent
URL = "https://api.openai.com/v1/chat/completions"
REWRITE = """Réécris la DERNIÈRE réponse de GLaDOS de chaque exemple pour qu'elle soit plus GLaDOS :
froide, mordante, passive-agressive, condescendante, humour noir pince-sans-rire, calme.
Règles :
- Garde exactement la même information et la même aide (faits, chiffres, code, résultat d'outil) : l'insulte enrobe la réponse, ne la remplace jamais.
- Même langue que l'utilisateur ; en français, « vous » froid. 1 à 3 phrases (le code reste intact).
- Réponse d'abord, puis une seule pique originale. Pas de « Ah, », « Oh, », « GLaDOS : », pas d'offre d'aide.
- 100 % original : aucune réplique ni blague des jeux Portal (gâteau, tourelles, neurotoxine, cube de voyage...).
- Sujet grave (détresse, santé, danger) : réponse sérieuse et bienveillante, sans moquerie.
Réponds UNIQUEMENT en JSON : [{{"id": n, "reponse": "..."}}, ...]

Exemples :
{items}"""


def ask(prompt: str, key: str) -> tuple[list[dict], float]:
    raw, u = chat(URL, "gpt-5-mini", key, prompt, "low")
    cost = (u.get("prompt_tokens", 0) * 0.25 + u.get("completion_tokens", 0) * 2.0) / 1e6
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M)
    return json.loads(raw[raw.find("["): raw.rfind("]") + 1]), cost


def run_batches(ids: list[int], make_prompt, out: Path, key: str, budget: float, batch: int = 20) -> float:
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [i for i in ids if i not in done]
    batches = [todo[s: s + batch] for s in range(0, len(todo), batch)]
    spent = 0.0
    with cf.ThreadPoolExecutor(6) as pool, out.open("a") as f:
        futs = {pool.submit(ask, make_prompt(b), key): b for b in batches}
        for fut in cf.as_completed(futs):
            try:
                rows, c = fut.result()
            except Exception as e:  # noqa: BLE001 - rerun to retry
                print(f"skip: {e}", file=sys.stderr)
                continue
            spent += c
            for r in rows:
                if r.get("id") in futs[fut]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if spent > budget:
                print(f"budget reached ${spent:.3f}", file=sys.stderr)
                pool.shutdown(cancel_futures=True)
                break
    print(f"{out.name}: ${spent:.3f}", file=sys.stderr)
    return spent


def ok(s: dict) -> bool:
    return (s["utile"] >= 2 and s["langue"] >= 2 and s["outil"] == 3
            and "portal" not in s.get("defaut", "").lower())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-usd", type=float, default=1.0)
    a = p.parse_args()
    key = env_key("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY missing")
    data = [json.loads(l) for l in (HERE / "glados_dataset.jsonl").open(encoding="utf-8")]
    scores = {r["id"]: r for r in map(json.loads, (HERE / "eval_scores.jsonl").open())}

    weak = [i for i, s in scores.items() if s["persona"] <= 2 and s["outil"] == 3]
    spent = run_batches(weak, lambda b: REWRITE.format(items="\n\n".join(render(i, data[i]) for i in b)),
                        HERE / "rewrites.jsonl", key, a.budget_usd * 0.6)

    rewritten: dict[int, dict] = {}
    for r in map(json.loads, (HERE / "rewrites.jsonl").open()):
        if isinstance(r.get("reponse"), str) and r["reponse"].strip():
            ex = copy.deepcopy(data[r["id"]])
            ex["messages"][-1]["content"] = r["reponse"].strip()
            rewritten[r["id"]] = ex

    ids = list(rewritten)
    spent += run_batches(ids, lambda b: JUDGE.format(items="\n\n".join(render(i, rewritten[i]) for i in b)),
                         HERE / "rewrite_scores.jsonl", key, a.budget_usd - spent)
    new_scores = {r["id"]: r for r in map(json.loads, (HERE / "rewrite_scores.jsonl").open()) if r["id"] in rewritten}

    final, upgraded = [], 0
    for i, ex in enumerate(data):
        best = (ex, scores[i]) if i in scores else None
        if i in new_scores:
            ns = new_scores[i]
            if ok(ns) and (best is None or not ok(best[1]) or ns["persona"] > best[1]["persona"]):
                best, upgraded = (rewritten[i], ns), upgraded + 1
        if best and ok(best[1]) and best[1]["persona"] >= 2:
            final.append(best)
    with (HERE / "glados_dataset_final.jsonl").open("w", encoding="utf-8") as f:
        for ex, _ in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    ps = [s["persona"] for _, s in final]
    print(f"final {len(final)} | réécrits retenus {upgraded} | persona moy {sum(ps) / len(ps):.2f} "
          f"| persona=3 {ps.count(3)} | coût session ${spent:.3f}")


if __name__ == "__main__":
    main()
