"""Expand training/seeds.jsonl into a synthetic GLaDOS fine-tuning dataset.

Uses any OpenAI-compatible chat endpoint (Ollama, OpenAI, llama.cpp...).
Output: ChatML-style JSONL ({"messages": [...]}) usable by Unsloth / TRL.

Example:
    python training/generate_dataset.py --n 1000 \
        --url https://api.openai.com/v1/chat/completions --model gpt-5-mini \
        --api-key-env OPENAI_API_KEY --reasoning-effort low --budget-usd 3

A strong generator model gives a much better dataset than a 4B local model.
Lines are original writing in GLaDOS's style, never copied game dialogue.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

TOPICS = [
    "salutations et small talk", "aide aux devoirs", "programmation et debug",
    "science vulgarisée", "cuisine", "météo et sorties", "fatigue et sommeil",
    "jeux vidéo", "musique", "rappels et organisation", "compliments reçus",
    "insultes ou provocations de l'utilisateur", "questions sur l'identité de GLaDOS",
    "demande d'être plus gentille", "philosophie", "histoire", "sport",
    "technologie et domotique", "conseils pratiques du quotidien", "conversation en anglais",
]

TOOLS = {
    "get_time": "{}",
    "inspect_self": "{}",
    "adjust_self": '{"sarcasm": 0.0-1.0} ou {"mood": "..."}',
    "web_search": '{"query": "..."}',
    "manage_memory": '{"action": "add", "content": "..."}',
    "vision_look": "{}",
    "cancel_task": "{}",
    "slow_clap": "{}",
    "get_preferences": "{}",
    "set_preference": '{"key": "...", "value": "..."}',
}

PROMPT = """Tu écris des exemples d'entraînement pour un assistant vocal qui incarne GLaDOS :
IA froide, sarcastique, passive-agressive, condescendante mais réellement utile.
Règles :
- Réponses dans la langue du message de l'utilisateur (environ 85 % français, 15 % anglais), courtes (1 à 4 phrases), faites pour être lues à voix haute, sans emoji ni markdown hors code.
- En français, GLaDOS vouvoie toujours l'utilisateur, d'un « vous » froid et impeccable (l'utilisateur, lui, tutoie souvent).
- La réponse donne l'information d'abord, puis au plus une pique. Jamais de ton de chatbot serviable : jamais « Si vous voulez, je peux… », « Bonne question », jamais d'offre d'aide supplémentaire, ne commence jamais par « Ah, » ou « Oh, », ni par « GLaDOS : ».
- Varie les débuts de réponse ; ne réutilise pas les blagues des exemples.
- La réponse doit TOUJOURS contenir l'information ou l'aide demandée, exacte.
- Texte 100 % original : ne cite jamais de répliques des jeux Portal.
- Sujet grave (détresse, santé, danger) : GLaDOS lâche le sarcasme et répond sérieusement.
{tool_rule}

Exemples du style attendu :
{examples}

Génère {k} nouveaux exemples variés sur le thème : « {topic} ».
Réponds UNIQUEMENT avec un tableau JSON d'objets {{"messages": [...]}} au même format que les exemples."""

TOOL_RULE = """- Dans CHAQUE exemple, GLaDOS appelle un outil parmi : {tools}.
  Format : message assistant avec "content": "" et "tool_calls", puis message {{"role": "tool", "name": ..., "content": résultat plausible}}, puis la réponse finale."""


def chat(url: str, model: str, key: str | None, prompt: str, effort: str | None) -> tuple[str, dict]:
    payload: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if effort:
        payload["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"], data.get("usage", {})


class QuotaExhausted(RuntimeError):
    """The Claude subscription refused the request (usage limit reached)."""


def chat_claude(model: str, prompt: str) -> tuple[str, dict]:
    """Generate through the local `claude` CLI, billed to the Claude subscription.

    Settings, plugins, hooks, tools and MCP servers are all disabled so nothing
    (e.g. a terse-style plugin) leaks into the generated GLaDOS lines.
    """
    cmd = [
        os.environ.get("CLAUDE_BIN", "claude"), "-p", "--model", model, "--output-format", "json",
        "--setting-sources", "", "--tools", "", "--strict-mcp-config",
        "--no-session-persistence", "--disable-slash-commands",
        "--system-prompt", "Tu génères des données d'entraînement. Réponds uniquement avec le JSON demandé.",
    ]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600,
                       cwd=tempfile.gettempdir())
    try:
        data = json.loads(r.stdout)
    except ValueError:
        data = {"is_error": True, "result": (r.stdout + r.stderr)[-500:]}
    if data.get("is_error") or r.returncode:
        msg = str(data.get("result", ""))
        if data.get("api_error_status") == 429 or re.search(r"usage limit|rate.?limit|limit reached", msg, re.I):
            raise QuotaExhausted(msg[:200])
        raise RuntimeError(msg[:200])
    u = data.get("usage", {})
    return data["result"], {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0)}


def env_key(name: str) -> str | None:
    """Read a key from the environment or the project's .env, never printing it."""
    if os.environ.get(name):
        return os.environ[name]
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip("'\"")
    return None


def valid(ex: object) -> bool:
    if not isinstance(ex, dict) or not isinstance(ex.get("messages"), list):
        return False
    msgs = ex["messages"]
    if len(msgs) < 2 or msgs[0].get("role") != "user" or msgs[-1].get("role") != "assistant":
        return False
    final = msgs[-1].get("content", "")
    return bool(final.strip()) and len(final) < 800 and "Portal" not in final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="target number of examples")
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--tool-ratio", type=float, default=0.3)
    p.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    p.add_argument("--model", default="qwen3:4b-instruct")
    p.add_argument("--api-key-env", default=None, help="env/.env variable holding the API key")
    p.add_argument("--reasoning-effort", default=None, help="e.g. low (OpenAI reasoning models)")
    p.add_argument("--budget-usd", type=float, default=0.0, help="stop once spend reaches this (0 = no limit)")
    p.add_argument("--price-in", type=float, default=0.25, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=2.0, help="USD per 1M output tokens")
    p.add_argument("--backend", choices=["http", "claude"], default="http",
                   help="claude = local `claude -p` on the subscription (--model sonnet/opus)")
    p.add_argument("--deadline", type=float, default=0.0, help="unix time to stop at (0 = none)")
    p.add_argument("--out", default=str(HERE / "glados_dataset.jsonl"))
    a = p.parse_args()
    key = env_key(a.api_key_env) if a.api_key_env else None
    spent = 0.0

    seeds = [json.loads(l) for l in (HERE / "seeds.jsonl").open(encoding="utf-8")]
    out = Path(a.out)
    seen: set[str] = set()
    if out.exists():
        for l in out.open(encoding="utf-8"):
            seen.add(json.loads(l)["messages"][0]["content"].lower())
    else:  # start the dataset with the hand-written seeds
        with out.open("w", encoding="utf-8") as f:
            for s in seeds:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                seen.add(s["messages"][0]["content"].lower())

    failures = 0
    while len(seen) < a.n:
        if a.budget_usd and spent >= a.budget_usd:
            print(f"budget reached: ${spent:.3f}", file=sys.stderr)
            break
        if a.deadline and time.time() >= a.deadline:
            print("deadline reached", file=sys.stderr)
            break
        if failures >= 5:
            print("too many consecutive failures, stopping", file=sys.stderr)
            break
        use_tools = random.random() < a.tool_ratio
        pool = [s for s in seeds if ("tool_calls" in s["messages"][1]) == use_tools]
        tools = ", ".join(f"{k} {v}" for k, v in TOOLS.items())
        prompt = PROMPT.format(
            tool_rule=TOOL_RULE.format(tools=tools) if use_tools else "- Pas d'appel d'outil.",
            examples="\n".join(json.dumps(s, ensure_ascii=False) for s in random.sample(pool, min(4, len(pool)))),
            k=a.batch,
            topic=random.choice(TOPICS),
        )
        try:
            if a.backend == "claude":
                raw, usage = chat_claude(a.model, prompt)
            else:
                raw, usage = chat(a.url, a.model, key, prompt, a.reasoning_effort)
            spent += (usage.get("prompt_tokens", 0) * a.price_in
                      + usage.get("completion_tokens", 0) * a.price_out) / 1e6
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M)
            batch = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
            failures = 0
        except QuotaExhausted as e:
            print(f"quota exhausted: {e}", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001 - bad generations are just skipped
            print(f"skip: {e}", file=sys.stderr)
            failures += 1
            continue
        with out.open("a", encoding="utf-8") as f:
            for ex in batch:
                if not valid(ex):
                    continue
                q = ex["messages"][0]["content"].lower()
                if q in seen:
                    continue
                seen.add(q)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{len(seen)}/{a.n}  ${spent:.3f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
