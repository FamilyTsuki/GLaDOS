"""PreToolUse hook for the coding agents GLaDOS drives (Claude Code and Antigravity).

Standalone (stdlib only, run with ``python3 -I``): it reads the hook payload on stdin and the guard
state file named by argv[1] or $GLADOS_CODE_AGENT_STATE. Commands matching the owner's blacklist and
edits under protected paths are denied, unless that exact command was approved by voice (one use).
Without a state file (the owner's own agy sessions) it prints nothing and changes nothing.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

WRITE_TOOL = re.compile(r"write|edit|replace|create|delete|move|rename", re.IGNORECASE)
COMMAND_KEYS = ("command", "CommandLine", "commandLine", "cmd")
PATH_KEYS = ("file_path", "path", "notebook_path", "TargetFile", "AbsolutePath", "targetFile", "absolutePath")
REASON = (
    "Blocked by GLaDOS: this needs the owner's spoken approval. Do not try another way. Stop now and end your "
    "final message with the line: NEEDS_APPROVAL: {what}"
)


def _consume_approval(state_path: Path, command: str) -> bool:
    """True once for a command the owner approved; the approval is removed."""
    with state_path.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.load(handle)
        approved = state.get("approved", [])
        if command not in approved:
            return False
        approved.remove(command)
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle)
        return True


def _rm_inside(command: str, cwd: str) -> bool:
    """Every rm in ``command`` only removes things inside the project folder (build output, node_modules...)."""
    base = Path(cwd).resolve()
    if base == Path.home().resolve() or len(base.parts) < 4:
        return False  # Never treat the home folder or a shallow system folder as "the project".
    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        return False
    found = False
    for index, word in enumerate(words):
        if word != "rm":
            continue
        found = True
        for arg in words[index + 1 :]:
            if arg in {";", "&&", "||", "|", "&"}:
                break
            if arg.startswith("-"):
                continue
            if any(c in arg for c in "$`*?~{"):
                return False  # Expansions could reach anywhere.
            target = (base / arg).resolve()
            if target == base or base not in target.parents:
                return False
    return found


def _protected(raw: str, roots: list[str]) -> bool:
    path = Path(raw).expanduser().resolve()
    for root in roots:
        base = Path(root).expanduser().resolve()
        if path == base or base in path.parents:
            return True
    return False


def decide(payload: dict[str, Any], state_path: Path) -> str | None:
    """The deny reason, or None when the tool call may run."""
    state = json.loads(state_path.read_text())
    if "toolCall" in payload:  # Antigravity
        name = str(payload["toolCall"].get("name", ""))
        args = payload["toolCall"].get("args") or {}
    else:  # Claude Code
        name = str(payload.get("tool_name", ""))
        args = payload.get("tool_input") or {}
    command = next((str(args[k]) for k in COMMAND_KEYS if isinstance(args.get(k), str)), None)
    cwd = str(payload.get("cwd") or os.getcwd())
    rules = [
        p
        for p in state.get("blacklist", [])
        if not (p == state.get("rm_pattern") and command is not None and _rm_inside(command, cwd))
    ]
    blocked = command is not None and any(re.search(p, command, re.IGNORECASE) for p in rules)
    if command is not None and blocked and not _consume_approval(state_path, command):
        return REASON.format(what=command)
    if WRITE_TOOL.search(name):
        for key in PATH_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value and _protected(value, state.get("protected", [])):
                what = f"{name} {value}"
                if not _consume_approval(state_path, what):
                    return REASON.format(what=what)
    return None


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GLADOS_CODE_AGENT_STATE", "")
    if not source or not Path(source).is_file():
        return  # Not a GLaDOS run: leave the decision to the agent's own settings.
    payload = json.load(sys.stdin)
    agy = "toolCall" in payload
    try:
        reason = decide(payload, Path(source))
    except (OSError, ValueError) as exc:
        reason = f"GLaDOS guard failed ({exc}); stop and report it."
    if agy:
        print(json.dumps({"decision": "deny", "reason": reason} if reason else {"decision": "allow"}))
    elif reason:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
