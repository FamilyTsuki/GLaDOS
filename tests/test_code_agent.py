import json
from pathlib import Path
import subprocess
import sys

from glados.tools.agent import coder
from glados.tools.agent.base import DEFAULT_BLACKLIST, AgentConfig
from glados.tools.agent.code_guard import decide
from glados.tools.agent.coder import CodeAgent


def _state(tmp_path: Path, approved: list[str] | None = None) -> Path:
    path = tmp_path / "guard.json"
    path.write_text(json.dumps({"blacklist": DEFAULT_BLACKLIST, "protected": ["~/.ssh"], "approved": approved or []}))
    return path


def test_guard_blocks_blacklisted_claude_command(tmp_path: Path) -> None:
    state = _state(tmp_path)
    payload = {"tool_name": "Bash", "tool_input": {"command": "sudo pacman -Syu"}}
    assert "NEEDS_APPROVAL: sudo pacman -Syu" in (decide(payload, state) or "")
    assert decide({"tool_name": "Bash", "tool_input": {"command": "npm test"}}, state) is None


def test_guard_approval_is_single_use(tmp_path: Path) -> None:
    state = _state(tmp_path, approved=["sudo pacman -Syu"])
    payload = {"toolCall": {"name": "run_command", "args": {"CommandLine": "sudo pacman -Syu"}}}
    assert decide(payload, state) is None
    assert decide(payload, state) is not None


def test_guard_blocks_protected_writes(tmp_path: Path) -> None:
    state = _state(tmp_path)
    payload = {"tool_name": "Write", "tool_input": {"file_path": str(Path("~/.ssh/config").expanduser())}}
    assert decide(payload, state) is not None
    assert decide({"tool_name": "Read", "tool_input": {"file_path": "~/.ssh/config"}}, state) is None


def test_guard_script_is_inert_without_state() -> None:
    script = Path(coder.__file__).with_name("code_guard.py")
    run = subprocess.run(
        [sys.executable, "-I", str(script)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "sudo ls"}}),
        capture_output=True,
        text=True,
        env={},
        check=True,
    )
    assert run.stdout == ""


def test_guard_script_claude_deny_format(tmp_path: Path) -> None:
    script = Path(coder.__file__).with_name("code_guard.py")
    run = subprocess.run(
        [sys.executable, "-I", str(script), str(_state(tmp_path))],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm -rf build"}}),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(run.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_parse_extracts_question_and_session() -> None:
    out = {"ok": True, "stdout": json.dumps({"result": "Fait.\nQUESTION: SQLite ou Postgres ?", "session_id": "abc"})}
    text, session = CodeAgent.parse("claude", out)
    assert session == "abc"
    assert coder.QUESTION.search(text).group(1) == "SQLite ou Postgres ?"  # type: ignore[union-attr]


def test_project_dir_stays_in_home(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    tool = CodeAgent(_Sink(), {"agent": AgentConfig(enabled=True, code_projects_root=str(tmp_path / "Projects"))})
    assert tool.project_dir("demo") == (tmp_path / "Projects" / "demo").resolve()
    for bad in ("/etc/x", str(tmp_path)):
        try:
            tool.project_dir(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


class _Sink:
    def put(self, item: dict) -> None:  # type: ignore[type-arg]
        pass

    put_nowait = put


def test_spoken_project_name_finds_existing_folder(tmp_path: Path) -> None:
    (tmp_path / "Meteo-App").mkdir()
    (tmp_path / "GLaDOS").mkdir()
    assert CodeAgent.match_project(tmp_path, "mon projet météo") == tmp_path / "Meteo-App"
    assert CodeAgent.match_project(tmp_path, "glados") == tmp_path / "GLaDOS"
    assert CodeAgent.match_project(tmp_path, "nouveau jeu") is None


def test_current_project_drives_follow_ups(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from glados.core.tool_offer import filter_tools_for_message
    from glados.tools.agent.definitions import agent_definitions

    monkeypatch.setattr(coder, "STATE_DIR", tmp_path)
    offered = {t["function"]["name"] for t in filter_tools_for_message(agent_definitions, "Super, merci")}
    assert "code_agent" not in offered and coder.context_line() == ""
    coder._save_session("abc123", {"backend": "claude", "project": "/x/meteo", "task": "t", "last_report": "ok"})
    offered = {t["function"]["name"] for t in filter_tools_for_message(agent_definitions, "Super, merci")}
    assert "code_agent" in offered
    assert "meteo" in coder.context_line() and "phone" in coder.context_line()
    tool = CodeAgent(_Sink(), {"agent": AgentConfig(enabled=True)})
    assert tool.pick({})[0] == "abc123"


def test_firewall_fix_only_when_port_is_closed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rules = tmp_path / "user.rules"
    rules.write_text("### tuple ### allow tcp 8050 0.0.0.0/0 any 10.0.0.0/8 in\n")
    monkeypatch.setattr(coder, "UFW_RULES", rules)
    monkeypatch.setattr(
        coder.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="active\n", stderr="")
    )
    tool = CodeAgent(_Sink(), {"agent": AgentConfig(enabled=True)})
    assert "8100:8199" in (tool.firewall_fix(8100) or "")
    rules.write_text("### tuple ### allow tcp 8100:8199 0.0.0.0/0 any 10.0.0.0/8 in\n")
    assert tool.firewall_fix(8100) is None


def test_preview_server_starts_and_reports_crashes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(coder, "STATE_DIR", tmp_path)
    (tmp_path / "index.html").write_text("hi")
    coder._save_session("p1", {"backend": "claude", "project": str(tmp_path), "task": "t"})
    tool = CodeAgent(_Sink(), {"agent": AgentConfig(enabled=True)})
    port = tool.preview_port("p1")
    ok, _ = tool.start_server("p1", f"{sys.executable} -m http.server $PORT --bind 0.0.0.0", port)
    try:
        assert ok and coder._port_open(port)
    finally:
        tool.stop_preview({"session": "p1"})
    ok, error = tool.start_server("p1", "echo boom; exit 3", port)
    assert not ok and "boom" in error


def test_activity_from_stream_json() -> None:
    tool_use = {"type": "tool_use", "name": "Bash", "input": {"command": "npm test", "description": "Run the tests"}}
    line = json.dumps({"type": "assistant", "message": {"content": [tool_use]}})
    assert coder.activity(line) == "Bash: Run the tests"
    edit = {"type": "tool_use", "name": "Edit", "input": {"file_path": "/home/x/site/index.html"}}
    assert coder.activity(json.dumps({"type": "assistant", "message": {"content": [edit]}})) == "Edit: index.html"
    assert coder.activity('{"type": "result", "result": "ok"}') is None
    assert coder.activity("not json") is None


def test_code_agents_only_ask_for_important_actions(tmp_path: Path) -> None:
    from glados.tools.agent.base import IMPORTANT_ACTIONS, RM_PATTERN

    project = tmp_path / "proj"
    (project / "build").mkdir(parents=True)
    state = tmp_path / "guard.json"
    state.write_text(json.dumps({"blacklist": IMPORTANT_ACTIONS, "rm_pattern": RM_PATTERN, "approved": []}))

    def bash(command: str) -> str | None:
        return decide({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(project)}, state)

    assert bash("python3 -c 'print(1)' && bash -c 'npm test'") is None
    assert bash("rm -rf build node_modules dist && npm run build") is None
    assert bash("rm -rf ../other") is not None
    assert bash("rm -rf .") is not None
    assert bash("rm -rf $HOME/x") is not None
    assert bash("rm -rf ~/Documents") is not None
    assert bash("sudo pacman -Syu") is not None
