"""code_agent: GLaDOS drives Claude Code or Antigravity (agy) on a project, by voice.

Each job runs the CLI headless in the project folder as a background task, with full permissions.
A PreToolUse hook (code_guard.py) blocks the owner's blacklisted commands and protected paths: the agent
then stops with ``NEEDS_APPROVAL: <command>``, GLaDOS reads it out, and the owner's spoken confirmation
code lets that exact command run once. A question for the owner ends the agent's turn with
``QUESTION: <question>``; GLaDOS asks it aloud and sends the answer back into the same session.

The owner talks naturally: the most recent project is the current one, so "add a button" or "show me
on my phone" need no session id. ``show`` starts the project on the LAN and opens it on the phone
through KDE Connect (or shows a QR code on the PC screen when the phone is out of reach).
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import unicodedata
import uuid

from ...autonomy.task_manager import TaskResult
from .base import BACKGROUND_MAX_S, RM_PATTERN, _AgentTool, _notify_desktop, redact

STATE_DIR = Path("data/code_agent")
GUARD_SCRIPT = Path(__file__).with_name("code_guard.py").resolve()
AGY_HOOKS = Path("~/.gemini/config/hooks.json").expanduser()
AGY_HOOK_NAME = "glados-guard"
UFW_RULES = Path("/etc/ufw/user.rules")
BACKENDS = ("claude", "agy")
ACTIVE_WINDOW_S = 1800.0  # A project touched in the last 30 min is "the current one".
RESUME_WINDOW_S = 86400.0  # Starting work on the same project within a day continues its session.
PREVIEW_WAIT_S = 90.0
QUESTION = re.compile(r"^\s*QUESTION:\s*(.+)$", re.MULTILINE)
APPROVAL = re.compile(r"^\s*NEEDS_APPROVAL:\s*(.+)$", re.MULTILINE)
RUN = re.compile(r"^\s*RUN:\s*(.+)$", re.MULTILINE)
AGENT_PROMPT = (
    "You are driven by voice: the owner speaks to GLaDOS, a voice assistant on their PC, who relays your messages "
    "and reads your final message aloud. Nobody can answer a prompt while you run, so work autonomously and "
    "verify your work (build, tests) before stopping. Requests come from speech: read them generously and do what "
    "a sensible developer would, choosing reasonable defaults instead of asking. Only if you truly need a decision "
    "only the owner can make, stop and end your final message with exactly one line: QUESTION: <one short "
    "question>. If a tool call is blocked as needing the owner's approval, do not try another way: stop and end "
    "with the line it gives you, NEEDS_APPROVAL: <exact command>. Never start long-running servers yourself in the "
    "background; GLaDOS starts previews. Write the final message in French, 2 to 4 short spoken sentences (no code "
    "blocks, no markdown, no URLs, no long paths): what you did, the result, and what is left."
)
PREVIEW_PROMPT = (
    "The owner wants to see the result {where}. If this project is something that runs as a web page or server, "
    "make sure it works, then do NOT start it: end your final message with exactly one line RUN: <shell command, "
    "run from the project folder, that starts it in the foreground listening on host 0.0.0.0 and on the port in the "
    "PORT environment variable>, for example RUN: npm run dev -- --host 0.0.0.0 --port $PORT, or RUN: python3 -m "
    "http.server $PORT --bind 0.0.0.0. Install missing dependencies first. If it is not a web project (a script, "
    "a CLI, a library), run it or its tests yourself, describe what it shows in 2 or 3 spoken sentences, and end "
    "with the line RUN: none."
)
_lock = threading.Lock()


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def _sessions_path() -> Path:
    return STATE_DIR / "sessions.json"


def load_sessions() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_sessions_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_session(key: str, entry: dict[str, Any]) -> None:
    with _lock:
        sessions = load_sessions()
        sessions[key] = {**sessions.get(key, {}), **entry, "updated": time.time()}
        # Keep the 30 most recent sessions.
        recent = sorted(sessions, key=lambda k: sessions[k].get("updated", 0))[-30:]
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        _sessions_path().write_text(json.dumps({k: sessions[k] for k in recent}, ensure_ascii=False, indent=1))


def has_pending() -> bool:
    """A coding agent waits on the owner (a question or an approval): keep code_agent offered."""
    return any(s.get("pending") for s in load_sessions().values())


def current_session() -> tuple[str, dict[str, Any]] | None:
    """The project the owner is working on: one waiting for them, else the one touched last (30 min)."""
    sessions = load_sessions()
    if not sessions:
        return None
    waiting = [kv for kv in sessions.items() if kv[1].get("pending")]
    key, entry = max(waiting or sessions.items(), key=lambda kv: kv[1].get("updated", 0))
    recent = time.time() - entry.get("updated", 0) < ACTIVE_WINDOW_S
    return (key, entry) if entry.get("pending") or entry.get("running") or recent else None


def is_active() -> bool:
    return current_session() is not None


def context_line() -> str:
    """What GLaDOS should know about the current coding project, for her prompt."""
    found = current_session()
    if found is None:
        return ""
    key, entry = found
    tool = "Claude Code" if entry.get("backend") == "claude" else "Antigravity"
    if entry.get("running"):
        state = f"the agent is working on it now ({int(time.time() - entry.get('started', time.time()))} s)"
        if entry.get("activity"):
            state += f", currently: {entry['activity']}"
    elif pending := entry.get("pending"):
        state = f"the agent waits for the owner: {pending['kind']} - {pending['text'][:200]}"
    else:
        state = "last report: " + (entry.get("last_report") or "none")[:200]
    preview = entry.get("preview") or {}
    if preview.get("url"):
        state += f"; a preview is running at {preview['url']}"
    return (
        f"[Current coding project] {Path(entry.get('project', '')).name} with {tool} (session {key}); {state}. "
        "The owner's follow-ups about it, however they phrase them ('add a dark mode', 'it crashes', 'where is it "
        "at?', 'I want to see it on my phone', 'open it here', 'stop the preview'), go to code_agent: reply for "
        "changes or answers, show to see the result, status for progress. No session id is needed."
    )


def activity(line: str) -> str | None:
    """What a Claude Code stream-json line says the agent is doing, in a few words."""
    try:
        event = json.loads(line)
    except ValueError:
        return None
    if not isinstance(event, dict) or event.get("type") != "assistant":
        return None
    for block in reversed((event.get("message") or {}).get("content") or []):
        if block.get("type") == "tool_use":
            given = block.get("input") or {}
            detail = given.get("description") or given.get("file_path") or given.get("command") or ""
            return redact(f"{block.get('name')}: {Path(detail).name if '/' in detail[:1] else detail}")[:160]
        if block.get("type") == "text" and str(block.get("text", "")).strip():
            return redact(str(block["text"]).strip().splitlines()[0])[:160]
    return None


def lan_ip() -> str:
    """The PC's address on the local network (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def _port_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


class CodeAgent(_AgentTool):
    name = "code_agent"

    # --- tool actions -------------------------------------------------------------------------
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action") or "start"
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "start": self.start,
            "reply": self.reply,
            "show": self.show,
            "stop_preview": self.stop_preview,
            "status": self.status,
        }
        if action not in handlers:
            raise ValueError("action must be start, reply, show, stop_preview or status")
        return handlers[action](args)

    def start(self, args: dict[str, Any]) -> dict[str, Any]:
        task = str(args.get("task") or "").strip()
        if not task:
            raise ValueError("Describe the task for the coding agent")
        backend = args.get("backend") or "claude"
        if backend not in BACKENDS:
            raise ValueError("backend must be claude or agy")
        raw = args.get("project")
        if not (isinstance(raw, str) and raw.strip()) and (found := current_session()):
            raw = found[1]["project"]
        project = self.project_dir(raw)
        # Same project, same tool, recently: continue that session so the agent keeps its context.
        for key, entry in load_sessions().items():
            same = entry.get("project") == str(project) and entry.get("backend") == backend
            if same and entry.get("session") and time.time() - entry.get("updated", 0) < RESUME_WINDOW_S:
                if entry.get("running"):
                    return {"ok": False, "error": "The agent is still working on this project; wait for its report"}
                _save_session(key, {"task": task[:300]})
                return self.launch(key, {**entry, "task": task[:300]}, task)
        key = uuid.uuid4().hex[:6]
        entry = {"backend": backend, "project": str(project), "task": task[:300], "session": None, "pending": None}
        _save_session(key, entry)
        return self.launch(key, entry, task)

    def pick(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        sessions = load_sessions()
        key = str(args.get("session") or "")
        if key in sessions:
            return key, sessions[key]
        if found := current_session():
            return found
        if sessions:  # Older than the active window, but still the last project.
            return max(sessions.items(), key=lambda kv: kv[1].get("updated", 0))
        raise ValueError("No coding project yet: use action=start with a project and a task")

    def reply(self, args: dict[str, Any]) -> dict[str, Any]:
        key, entry = self.pick(args)
        if entry.get("running"):
            return {"ok": False, "error": "The agent is still working on this project; wait for its report"}
        pending = entry.get("pending") or {}
        answer = str(args.get("answer") or args.get("task") or "").strip()
        if args.get("approve"):
            if pending.get("kind") != "approval":
                raise ValueError("Nothing is waiting for approval in that session")
            command = pending["text"]
            if not self.guard.approved():
                return self.guard.refusal(f"the coding agent wants to run: {command}", "voice")
            self.approve(command)
            message = f"The owner approved this exact action; run it now: {command}" + (
                f"\nOwner's note: {answer}" if answer else ""
            )
        elif pending.get("kind") == "approval" and not answer:
            raise ValueError("Approve it (approve=true, with the owner's code) or pass the owner's answer")
        else:
            if not answer:
                raise ValueError("Give the owner's answer or new instructions")
            message = answer
        return self.launch(key, entry, message)

    def show(self, args: dict[str, Any]) -> dict[str, Any]:
        key, entry = self.pick(args)
        if entry.get("running"):
            return {"ok": False, "error": "The agent is still working; show the result once it reports back"}
        target = "pc" if args.get("target") == "pc" else "phone"
        tasks = self.tool_config.get("task_manager")
        if tasks is None:
            raise ValueError("Background tasks are unavailable in this profile")
        task_id = f"task_code_show_{key}_{uuid.uuid4().hex[:4]}"
        cancelled = threading.Event()
        label = f"Preview: {Path(entry['project']).name}"
        tasks.submit(task_id, label, lambda: self.preview(key, target, cancelled), group="agent", cancelled=cancelled)
        where = "on the owner's phone" if target == "phone" else "on this PC"
        return {
            "ok": True,
            "status": "queued",
            "task_id": task_id,
            "instruction": f"Say in one short sentence that you are getting it ready {where}.",
        }

    def stop_preview(self, args: dict[str, Any]) -> dict[str, Any]:
        key, entry = self.pick(args)
        preview = entry.get("preview") or {}
        if not preview.get("pid"):
            return {"ok": True, "stopped": False, "note": "No preview was running"}
        self.kill(preview["pid"], preview.get("port"))
        _save_session(key, {"preview": None})
        return {"ok": True, "stopped": True, "project": Path(entry["project"]).name}

    def status(self, _args: dict[str, Any] | None = None) -> dict[str, Any]:
        sessions = load_sessions()
        recent = sorted(sessions.items(), key=lambda kv: -kv[1].get("updated", 0))[:8]
        now = time.time()
        return {
            "ok": True,
            "sessions": [
                {
                    "session": k,
                    "backend": s.get("backend"),
                    "project": Path(s.get("project", "")).name,
                    "task": s.get("task"),
                    "working_for_s": int(now - s.get("started", now)) if s.get("running") else None,
                    "doing_now": s.get("activity") if s.get("running") else None,
                    "waiting_for_owner": s.get("pending"),
                    "preview": (s.get("preview") or {}).get("url"),
                    "last_report": (s.get("last_report") or "")[:300],
                }
                for k, s in recent
            ],
        }

    # --- projects and guard -------------------------------------------------------------------
    def project_dir(self, raw: object) -> Path:
        root = Path(self.config.code_projects_root).expanduser()
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Name the project (folder name under the projects folder, or a path)")
        text = raw.strip()
        if text.startswith(("/", "~")):
            path = Path(text).expanduser()
        else:
            path = self.match_project(root, text) or root / text
        path = path.resolve()
        home = Path.home().resolve()
        if path == home or home not in path.parents:
            raise ValueError("The project must be a folder inside the home directory, not the home itself")
        if self.guard.protected(path):
            raise ValueError(f"{path} is a protected path")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def match_project(root: Path, spoken: str) -> Path | None:
        """An existing project folder for a spoken name: 'mon projet météo' finds Meteo-App."""
        words = [w for w in re.split(r"\s+", spoken) if _fold(w) not in {"mon", "le", "la", "projet", "project", "my"}]
        wanted = _fold("".join(words)) or _fold(spoken)
        try:
            folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            return None
        exact = [p for p in folders if _fold(p.name) == wanted]
        partial = [p for p in folders if wanted and (wanted in _fold(p.name) or _fold(p.name) in wanted)]
        if exact:
            return exact[0]
        return partial[0] if len(partial) == 1 else None

    def guard_state(self) -> Path:
        path = (STATE_DIR / "guard.json").resolve()
        with _lock:
            try:
                approved = json.loads(path.read_text()).get("approved", [])
            except (OSError, ValueError):
                approved = []
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            state = {
                "blacklist": self.config.code_blacklist,
                "rm_pattern": RM_PATTERN,
                "protected": self.config.protected_paths,
                "approved": approved,
            }
            path.write_text(json.dumps(state))
        return path

    def approve(self, command: str) -> None:
        path = self.guard_state()
        with _lock:
            state = json.loads(path.read_text())
            state["approved"] = [*state.get("approved", [])[-9:], command]
            path.write_text(json.dumps(state))

    def install_agy_hook(self) -> None:
        """agy reads hooks only from hooks.json; the hook is inert unless GLADOS_CODE_AGENT_STATE is set."""
        command = f"{sys.executable} -I {GUARD_SCRIPT}"
        try:
            hooks = json.loads(AGY_HOOKS.read_text()) if AGY_HOOKS.exists() else {}
        except ValueError as exc:
            raise OSError(f"{AGY_HOOKS} is not valid JSON; fix it before using agy") from exc
        wanted = {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": 10}]}]}
        if hooks.get(AGY_HOOK_NAME) != wanted:
            hooks[AGY_HOOK_NAME] = wanted
            AGY_HOOKS.parent.mkdir(parents=True, exist_ok=True)
            AGY_HOOKS.write_text(json.dumps(hooks, indent=2))

    # --- running the coding agent -------------------------------------------------------------
    def argv(self, entry: dict[str, Any], message: str, state: Path) -> tuple[list[str], dict[str, str]]:
        session = entry.get("session")
        if entry["backend"] == "claude":
            hook = f"{sys.executable} -I {GUARD_SCRIPT} {state}"
            settings = {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": hook}]}]}}
            argv = [
                shutil.which("claude") or "claude",
                "-p",
                message,
                "--output-format",
                "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--settings",
                json.dumps(settings),
                "--append-system-prompt",
                AGENT_PROMPT,
            ]
            if self.config.code_claude_model:
                argv += ["--model", self.config.code_claude_model]
            argv += ["--resume", session] if session else ["--session-id", entry["new_session"]]
            return argv, {}
        self.install_agy_hook()
        prompt = message if session else f"{AGENT_PROMPT}\n\nTask from the owner:\n{message}"
        argv = [shutil.which("agy") or "agy", "-p", prompt, "--output-format", "json", "--dangerously-skip-permissions"]
        if session:
            argv += ["--conversation", session]
        return argv, {"GLADOS_CODE_AGENT_STATE": str(state)}

    def launch(self, key: str, entry: dict[str, Any], message: str) -> dict[str, Any]:
        tasks = self.tool_config.get("task_manager")
        if tasks is None:
            raise ValueError("Background tasks are unavailable in this profile")
        task_id = "task_code_" + key + "_" + uuid.uuid4().hex[:4]
        cancelled = threading.Event()
        label = f"{'Claude Code' if entry['backend'] == 'claude' else 'Antigravity'}: {entry['task'][:80]}"
        _save_session(key, {"running": True, "pending": None, "started": time.time()})
        tasks.submit(task_id, label, lambda: self.work(key, message, cancelled), group="agent", cancelled=cancelled)
        return {
            "ok": True,
            "status": "queued",
            "project": Path(entry["project"]).name,
            "task_id": task_id,
            "instruction": "Tell the user in one short sentence that you are on it and will tell them when it is "
            "done. cancel_task with this task_id stops it.",
        }

    def turn(self, key: str, message: str, cancelled: threading.Event) -> tuple[str, dict[str, str] | None, bool]:
        """One coding-agent turn in the session: (final message, what it waits for, ok)."""
        entry = load_sessions()[key]
        if entry["backend"] == "claude" and not entry.get("session"):
            entry["new_session"] = str(uuid.uuid4())
        _save_session(key, {"running": True, "started": entry.get("started") or time.time()})
        try:
            argv, env = self.argv(entry, message, self.guard_state())
            result = self.run_cli(
                argv, Path(entry["project"]), env, cancelled, lambda what: _save_session(key, {"activity": what})
            )
        except OSError as exc:
            result = {"ok": False, "error": str(exc)}
        text, session = self.parse(entry["backend"], result)
        update: dict[str, Any] = {"running": False, "last_report": text[:2000], "activity": None}
        if session or entry.get("new_session"):
            update["session"] = session or entry.get("new_session")
        pending = None
        if found := APPROVAL.search(text):
            pending = {"kind": "approval", "text": found.group(1).strip()}
        elif found := QUESTION.search(text):
            pending = {"kind": "question", "text": found.group(1).strip()}
        update["pending"] = pending
        _save_session(key, update)
        return text, pending, bool(result.get("ok"))

    def work(self, key: str, message: str, cancelled: threading.Event) -> TaskResult:
        text, pending, ok = self.turn(key, message, cancelled)
        if cancelled.is_set():
            return TaskResult("cancelled", "Coding agent cancelled", notify_user=False)
        return self.report(key, text, pending, ok)

    def run_cli(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        cancelled: threading.Event,
        on_activity: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run the CLI; stream-json lines report what the agent is doing as it goes."""
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env={**os.environ, **env},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
            errors="replace",
        )
        out: list[str] = []
        err: list[str] = []

        def read_out() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                out.append(line)
                if on_activity is not None and (what := activity(line)):
                    on_activity(what)

        def read_err() -> None:
            assert process.stderr is not None
            err.extend(process.stderr)

        readers = [threading.Thread(target=f, daemon=True) for f in (read_out, read_err)]
        for reader in readers:
            reader.start()
        deadline = started + min(self.config.code_timeout_s, BACKGROUND_MAX_S)
        error = None
        while process.poll() is None:
            if cancelled.is_set() or time.monotonic() > deadline:
                os.killpg(process.pid, signal.SIGKILL)
                error = "Cancelled" if cancelled.is_set() else "Timed out"
                break
            time.sleep(0.5)
        process.wait()
        for reader in readers:
            reader.join(timeout=5)
        result = {
            "ok": error is None and process.returncode == 0,
            "stdout": "".join(out),
            "stderr": "".join(err)[-2000:],
        }
        if error:
            result["error"] = error
        return result

    @staticmethod
    def parse(backend: str, result: dict[str, Any]) -> tuple[str, str | None]:
        """(final message, session id) from the CLI's JSON output; tolerant of either CLI's shape."""
        raw = str(result.get("stdout") or "").strip()
        data: Any = None
        for candidate in (raw, raw.splitlines()[-1] if raw else ""):
            try:
                data = json.loads(candidate)
                break
            except ValueError:
                continue
        if isinstance(data, list):  # stream-like arrays: take the last result object
            data = next((d for d in reversed(data) if isinstance(d, dict) and "result" in d), data[-1] if data else {})
        if not isinstance(data, dict):
            text = raw or str(result.get("error") or result.get("stderr") or "No output")
            return redact(text)[-3000:], None
        text = next(
            (str(data[k]) for k in ("result", "response", "text", "output", "message") if data.get(k)),
            str(result.get("error") or result.get("stderr") or raw),
        )
        session = next(
            (str(data[k]) for k in ("session_id", "sessionId", "conversationId", "conversation_id") if data.get(k)),
            None,
        )
        return redact(text)[-3000:], session

    def report(self, key: str, text: str, pending: dict[str, str] | None, ok: bool) -> TaskResult:
        entry = load_sessions()[key]
        if pending and pending["kind"] == "approval":
            instruction = (
                f"The coding agent is blocked and needs the owner's approval to run: {pending['text']}. Say what it "
                "wants to do and why in one or two sentences, and ask the owner to say their confirmation code to "
                "allow it. When their next message contains it, call code_agent action=reply approve=true. You do "
                "not know the code: never say, guess or invent it."
            )
            status = "waiting"
        elif pending:
            instruction = (
                f"The coding agent asks the owner: {pending['text']}. Ask them this naturally, in your own words, "
                "then pass their answer with code_agent action=reply answer=<their words>."
            )
            status = "waiting"
        else:
            instruction = (
                "Tell the owner the result in your own words, briefly, like a colleague would. If it is something "
                "they can look at, you may offer to show it on their phone."
            )
            status = "done" if ok else "error"
        _notify_desktop(f"GLaDOS: coding agent {status}", entry["task"][:120])
        return TaskResult(
            status,
            f"Coding agent ({entry['backend']}) {status}: {entry['task'][:100]}",
            notify_user=True,
            report=json.dumps(
                {"project": Path(entry["project"]).name, "report": text, "instruction": instruction},
                ensure_ascii=False,
            ),
            importance=0.9 if pending else 0.8,
            update_priority="important",
        )

    # --- previews -----------------------------------------------------------------------------
    def preview_port(self, key: str) -> int:
        first, last = self.config.code_preview_ports
        taken = {
            (s.get("preview") or {}).get("port") for k, s in load_sessions().items() if k != key and s.get("preview")
        }
        current = (load_sessions().get(key, {}).get("preview") or {}).get("port")
        if current and first <= current <= last:
            return int(current)
        for port in range(first, last + 1):
            if port not in taken and not _port_open(port):
                return port
        raise OSError(f"No free preview port between {first} and {last}")

    def preview(self, key: str, target: str, cancelled: threading.Event) -> TaskResult:
        """Make sure the project runs, start it on the LAN and open it where the owner wants it."""
        entry = load_sessions()[key]
        name = Path(entry["project"]).name
        command = entry.get("run_command")
        where = "on their phone" if target == "phone" else "on this PC"
        note = ""
        for attempt in range(3):
            if cancelled.is_set():
                return TaskResult("cancelled", "Preview cancelled", notify_user=False)
            if not command:
                ask = PREVIEW_PROMPT.format(where=where)
                if note:
                    ask = f"The preview failed: {note}\nFix the cause. " + ask
                text, pending, _ok = self.turn(key, ask, cancelled)
                if pending:
                    return self.report(key, text, pending, True)
                found = RUN.search(text)
                if found is None or found.group(1).strip().casefold() == "none":
                    _save_session(key, {"run_command": None})
                    return self.report(key, RUN.sub("", text).strip(), None, True)
                command = found.group(1).strip()
                _save_session(key, {"run_command": command})
            port = self.preview_port(key)
            started, note = self.start_server(key, command, port)
            if started:
                break
            command = None  # Let the agent look at the error and give a fixed command.
            if attempt == 2:
                return TaskResult(
                    "error",
                    f"Preview of {name} failed",
                    report=json.dumps(
                        {
                            "error": note,
                            "instruction": "Tell the owner briefly that the project does not start, and why.",
                        },
                        ensure_ascii=False,
                    ),
                    update_priority="important",
                )
        local = f"http://localhost:{port}/"
        url = f"http://{lan_ip()}:{port}/"
        _save_session(key, {"preview": {"pid": load_sessions()[key]["preview"]["pid"], "port": port, "url": url}})
        if target == "pc":
            subprocess.Popen(
                ["xdg-open", local], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            delivered = "opened in the browser on the PC"
        else:
            delivered = self.send_to_phone(url, name)
        firewall = self.firewall_fix(port) if target == "phone" else None
        instruction = (
            f"Tell the owner in one or two natural sentences that {name} is {delivered}. Do not read the address aloud."
        )
        if firewall and self.open_terminal(firewall):
            instruction += (
                " But the PC firewall still blocks the phone: you opened a terminal on the PC that opens the preview "
                "ports once and for all; they only need to type their password there, then reload the page on the "
                "phone. Do not read the command aloud."
            )
        elif firewall:
            instruction += (
                " But the PC firewall blocks the phone: tell them to run this once in a terminal (it asks for their "
                f"password), then it works every time: {firewall}"
            )
        return TaskResult(
            "done",
            f"Preview of {name} ready",
            report=json.dumps({"url": url, "delivered": delivered, "instruction": instruction}, ensure_ascii=False),
            importance=0.8,
            update_priority="important",
        )

    def start_server(self, key: str, command: str, port: int) -> tuple[bool, str]:
        """(started, error): run the preview detached, until the port answers."""
        old = (load_sessions()[key].get("preview") or {}).get("pid")
        if old:
            self.kill(old, port)
        if _port_open(port):
            return False, f"Port {port} is already used by another program"
        log_path = STATE_DIR / f"preview-{key}.log"
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                ["bash", "-c", command],
                cwd=load_sessions()[key]["project"],
                env={**os.environ, "PORT": str(port), "HOST": "0.0.0.0", "BROWSER": "none"},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _save_session(key, {"preview": {"pid": process.pid, "port": port}})
        deadline = time.monotonic() + PREVIEW_WAIT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if _port_open(port):
                return True, ""
            time.sleep(0.5)
        else:
            self.kill(process.pid, port)
        tail = redact(log_path.read_text(errors="replace")[-1500:])
        error = f"`{command}` " + ("exited" if process.poll() is not None else "never listened on port " + str(port))
        return False, f"{error}. Output:\n{tail}"

    @staticmethod
    def kill(pid: int, port: int | None = None) -> None:
        """Stop a preview's process group; with ``port``, wait until it is free (SIGKILL after 5 s)."""
        for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            deadline = time.monotonic() + wait
            while port is not None and _port_open(port) and time.monotonic() < deadline:
                time.sleep(0.2)
            if port is None or not _port_open(port):
                return

    def send_to_phone(self, url: str, name: str) -> str:
        """Open the link on the phone through KDE Connect; otherwise show a QR code on the PC screen."""
        cli = shutil.which("kdeconnect-cli")
        if cli:
            try:
                device = self.config.code_phone_device
                if not device:
                    found = subprocess.run(
                        [cli, "-a", "--id-only"], capture_output=True, text=True, timeout=10, check=False
                    )
                    device = next(iter(found.stdout.split()), None)
                if device:
                    sent = subprocess.run(
                        [cli, "-d", device, "--share", url], capture_output=True, timeout=15, check=False
                    )
                    if sent.returncode == 0:
                        return "open on their phone"
            except (OSError, subprocess.TimeoutExpired):
                pass  # Fall back to the QR code.
        if shutil.which("qrencode"):
            image = (STATE_DIR / "preview-qr.png").resolve()
            subprocess.run(["qrencode", "-s", "12", "-m", "2", "-o", str(image), url], check=False, timeout=10)
            viewer = shutil.which("imv") or shutil.which("xdg-open")
            if viewer and image.exists():
                subprocess.Popen(
                    [viewer, str(image)], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                _notify_desktop(f"GLaDOS: {name}", url)
                return "ready: their phone was not reachable, so a QR code to scan is on the PC screen"
        _notify_desktop(f"GLaDOS: {name}", url)
        return f"ready at {url} (their phone was not reachable; the link is in a desktop notification)"

    @staticmethod
    def open_terminal(command: str) -> bool:
        """A terminal on the PC with ``command`` ready: the owner only types their password."""
        launcher = shutil.which("xdg-terminal-exec")
        if launcher is None:
            return False
        script = (
            "echo 'GLaDOS: ouverture des ports de prévisualisation pour ton téléphone (une seule fois).'; "
            f"{command} && echo 'Terminé. Tu peux fermer cette fenêtre.'; read -r -p 'Entrée pour fermer… ' _"
        )
        try:
            subprocess.Popen(
                [launcher, "bash", "-c", script],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
        return True

    def firewall_fix(self, port: int) -> str | None:
        """The one-time ufw command when ufw is on and nothing lets the LAN reach ``port``."""
        try:
            active = subprocess.run(
                ["systemctl", "is-active", "ufw"], capture_output=True, text=True, timeout=5, check=False
            )
            rules = UFW_RULES.read_text()
        except (OSError, subprocess.TimeoutExpired):
            return None
        if active.stdout.strip() != "active":
            return None
        for first, last in re.findall(r"^### tuple ### allow (?:tcp|any) (\d+)(?::(\d+))? ", rules, re.MULTILINE):
            if int(first) <= port <= int(last or first):
                return None
        ports = "{}:{}".format(*self.config.code_preview_ports)
        return " && ".join(
            f"sudo ufw allow from {net} to any port {ports} proto tcp" for net in ("10.0.0.0/8", "192.168.0.0/16")
        )
