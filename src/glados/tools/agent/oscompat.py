"""The few things the coding agent does differently on Linux and Windows."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
from typing import Any

WINDOWS = os.name == "nt"


def detached() -> dict[str, Any]:
    """Popen arguments for a child that outlives a cancel of its parent's job and can be killed as a tree."""
    if WINDOWS:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def kill_tree(pid: int, force: bool = False) -> bool:
    """Stop a process and its children; False when it was already gone."""
    if WINDOWS:
        argv = ["taskkill", "/PID", str(pid), "/T"] + (["/F"] if force else [])
        done = subprocess.run(argv, capture_output=True, check=False, timeout=15)
        return done.returncode == 0
    try:
        os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def bash() -> str:
    """A bash for the preview commands; on Windows the one from Git for Windows, never WSL's launcher."""
    if not WINDOWS:
        return "bash"
    git = shutil.which("git")
    candidates = [Path(git).resolve().parents[1] / "bin" / "bash.exe"] if git else []
    candidates.append(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise OSError("Install Git for Windows (it brings the bash that Claude Code also needs)")


def command_line(parts: list[str]) -> str:
    """One shell command string from argv, quoted for this OS's shell (hooks run through it)."""
    return subprocess.list2cmdline(parts) if WINDOWS else shlex.join(parts)


def open_path(target: str) -> None:
    """Open a URL or file with the desktop's default application."""
    if WINDOWS:
        os.startfile(target)  # type: ignore[attr-defined]
        return
    opener = shutil.which("imv") if Path(target).suffix == ".png" else None
    subprocess.Popen(
        [opener or "xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )


def kdeconnect_cli() -> str | None:
    found = shutil.which("kdeconnect-cli")
    if found or not WINDOWS:
        return found
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
        for folder in ("KDE Connect", r"Programs\KDE Connect"):
            exe = Path(base) / folder / "bin" / "kdeconnect-cli.exe"
            if exe.is_file():
                return str(exe)
    return None
