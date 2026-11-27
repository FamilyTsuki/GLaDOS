#!/usr/bin/env python3
"""Claude Code statusLine: shows quota usage and caches it for burn_weekly_quota.py.

Claude Code pipes a JSON blob on stdin each time the status line refreshes; its
`rate_limits` field holds the 5-hour and 7-day windows. We save that field to
~/.cache/glados/claude_usage.json so the systemd timer can read it later.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "glados" / "claude_usage.json"


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    limits = data.get("rate_limits") or {}
    if limits:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"saved_at": time.time(), "rate_limits": limits}))
        tmp.replace(CACHE)
    model = (data.get("model") or {}).get("display_name", "")
    parts = [model] if model else []
    for key, label in (("five_hour", "5h"), ("seven_day", "7j")):
        pct = (limits.get(key) or {}).get("used_percentage")
        if pct is not None:
            parts.append(f"{label} {pct:.0f}%")
    print(" | ".join(parts))


if __name__ == "__main__":
    main()
