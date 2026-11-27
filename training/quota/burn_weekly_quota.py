#!/usr/bin/env python3
"""Spend leftover weekly Claude quota on new GLaDOS training data.

Run every few minutes by glados-quota-burn.timer. When the weekly (7-day) window
resets within --lead minutes and quota is left, it starts --workers parallel
`generate_dataset.py --backend claude` runs into training/parts/, stops them at
the reset. Afterwards run merge_dataset.py, eval_dataset.py and refine_dataset.py by hand.

Usage data comes from ~/.cache/glados/claude_usage.json, written by statusline.py.
If that snapshot predates the last reset, the weekly reset time is extrapolated
(+7 days) and generation simply stops when Claude reports the limit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "glados"
USAGE = CACHE_DIR / "claude_usage.json"
STATE = CACHE_DIR / "quota_burn_state.json"
WEEK = 7 * 24 * 3600


def to_epoch(v: object) -> float | None:
    if isinstance(v, (int, float)):
        return float(v) / 1000 if v > 1e12 else float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def weekly_window(now: float) -> tuple[float, float | None] | None:
    """Return (next weekly reset, used % or None if the snapshot is stale)."""
    if not USAGE.exists():
        return None
    snap = json.loads(USAGE.read_text())
    week = (snap.get("rate_limits") or {}).get("seven_day") or {}
    reset = to_epoch(week.get("resets_at"))
    if reset is None:
        return None
    used: float | None = week.get("used_percentage")
    while reset <= now:  # snapshot from a previous week: extrapolate
        reset += WEEK
        used = None
    return reset, used


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=float, default=30, help="minutes before the weekly reset to start")
    p.add_argument("--max-used", type=float, default=99, help="skip if weekly usage is already above this %%")
    p.add_argument("--workers", type=int, default=2, help="parallel claude -p generators")
    p.add_argument("--model", default="opus")
    p.add_argument("--batch", type=int, default=8, help="examples per request")
    p.add_argument("--force", action="store_true", help="ignore the time window (manual test)")
    p.add_argument("--minutes", type=float, default=5, help="with --force: how long to run")
    a = p.parse_args()

    now = time.time()
    win = weekly_window(now)
    if a.force:
        deadline = now + a.minutes * 60
        reset_key = f"force-{int(now)}"
    else:
        if win is None:
            log(f"no weekly reset known yet ({USAGE} missing or without seven_day) — open Claude Code once")
            return
        reset, used = win
        left = (reset - now) / 60
        if left > a.lead:
            log(f"weekly reset in {left:.0f} min, waiting (lead {a.lead:.0f} min)")
            return
        if used is not None and used >= a.max_used:
            log(f"weekly quota already at {used:.0f}%, nothing to burn")
            return
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        reset_key = str(int(reset))
        if state.get("done") == reset_key:
            log("already ran for this reset")
            return
        deadline = reset - 30  # stop just before the reset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = (CACHE_DIR / "quota_burn.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another burn is running")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts = TRAINING / "parts"
    parts.mkdir(exist_ok=True)
    log(f"burning until {datetime.fromtimestamp(deadline):%H:%M} with {a.workers}×{a.model}")
    procs = []
    for w in range(a.workers):
        out = parts / f"claude_{stamp}_w{w}.jsonl"
        logf = out.with_suffix(".log").open("w")
        procs.append(subprocess.Popen(
            [sys.executable, str(TRAINING / "generate_dataset.py"), "--backend", "claude",
             "--model", a.model, "--batch", str(a.batch), "--n", "1000000",
             "--deadline", str(deadline), "--out", str(out)],
            stdout=logf, stderr=subprocess.STDOUT, cwd=TRAINING,
        ))
    for pr in procs:
        pr.wait()

    if not a.force:
        STATE.write_text(json.dumps({"done": reset_key, "at": time.time()}))
    new = sum(sum(1 for _ in f.open()) for f in parts.glob(f"claude_{stamp}_w*.jsonl"))
    # No automatic merge: eval_scores.jsonl is keyed by line index, so rebuilding
    # glados_dataset.jsonl must be followed by a fresh eval_dataset.py run.
    log(f"generation done: {new} raw lines (seeds included) in parts/claude_{stamp}_w*.jsonl")


if __name__ == "__main__":
    main()
