"""Keep GLaDOS's state in step between several PCs through a shared folder.

Code travels with git; this covers what git does not: settings, long-term memory, the console's pairing
token and certificate authority (so a paired phone trusts every PC), and the API keys in ``.env``.
The shared folder can be anything both PCs see: Syncthing, a cloud drive, a USB stick.

Each run merges both ways. A file changed on one side only is copied to the other. Changed on both
sides: append-only logs (``.jsonl``) are merged line by line; any other file keeps the newer copy and
the other one is saved next to it as ``<name>.conflict-<machine>``. Nothing is ever deleted.
Per-machine files (logs, audit trail, the console certificate for this PC's addresses, coding sessions
with local paths) stay out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time

REPO = Path(__file__).resolve().parents[2]
BUNDLE = "glados-state"
# (name in the bundle, local folder, relative paths or globs kept in step)
AREAS: tuple[tuple[str, Path, tuple[str, ...]], ...] = (
    (
        "data",
        REPO / "data",
        (
            "*.yaml",
            "*.json",
            "webapp_token",
            "webapp_tls/ca.pem",
            "webapp_tls/ca-key.pem",
            "webapp_tls/ca.srl",
        ),
    ),
    ("repo", REPO, (".env", "training/glados-q4_k_m.gguf")),
    ("home", Path.home() / ".glados", ("**/*",)),
)
BIG_FILE = 50 * 1024 * 1024
SKIP_PARTS = {"browser", "self-improve", "__pycache__", "logs", "code_agent"}
SKIP_SUFFIXES = (".tmp", ".lock")
LOCAL_STATE = REPO / "data" / ".sync-base.json"  # What each file looked like after the last sync, on this PC.
# Rebuilt per machine (keyed by its own LLM backend) or this PC's own bookkeeping.
FOLDER_FILE = REPO / "data" / ".sync-folder"  # This PC's path to the shared folder (differs per PC).
MACHINE_FILES = {LOCAL_STATE.name, "decision_lists.yaml", "agent_audit.jsonl"}


def machine() -> str:
    return socket.gethostname().split(".")[0].lower() or "pc"


def _digest(path: Path) -> str:
    """Content hash; big files (the fine-tuned model) use size and modification time, which copies keep."""
    info = path.stat()
    if info.st_size > BIG_FILE:
        return f"size:{info.st_size}:mtime:{int(info.st_mtime)}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path, patterns: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for pattern in patterns:
        for path in root.glob(pattern):
            rel = path.relative_to(root)
            if not path.is_file() or SKIP_PARTS & set(rel.parts) or path.name.endswith(SKIP_SUFFIXES):
                continue
            if ".conflict-" in path.name or path.name in MACHINE_FILES:
                continue
            found[rel.as_posix()] = path
    return found


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".sync-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _merge_lines(local: Path, remote: Path) -> bytes:
    """Union of two append-only logs: local order first, then the remote lines it lacks."""
    mine = local.read_bytes().splitlines(keepends=True)
    seen = {line.rstrip(b"\r\n") for line in mine}
    extra = [line for line in remote.read_bytes().splitlines(keepends=True) if line.rstrip(b"\r\n") not in seen]
    if mine and not mine[-1].endswith(b"\n"):
        mine[-1] += b"\n"
    return b"".join(mine + extra)


@dataclass
class Report:
    pushed: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = []
        for label, items in (
            ("Sent to the shared folder", self.pushed),
            ("Received", self.pulled),
            ("Merged line by line", self.merged),
            ("Changed on both PCs (older copy kept as .conflict-*)", self.conflicts),
        ):
            if items:
                out.append(f"{label} ({len(items)}): " + ", ".join(items[:12]) + (" …" if len(items) > 12 else ""))
        return out or ["Already in step."]


def sync(
    shared: Path,
    base_path: Path = LOCAL_STATE,
    areas: tuple[tuple[str, Path, tuple[str, ...]], ...] = AREAS,
    name: str | None = None,
) -> Report:
    """Merge this PC's state with the bundle in ``shared`` both ways."""
    name = name or machine()
    bundle = shared / BUNDLE
    try:
        base: dict[str, str] = json.loads(base_path.read_text())
    except (OSError, ValueError):
        base = {}
    report = Report()
    after: dict[str, str] = {}
    for area, root, patterns in areas:
        local = _files(root, patterns)
        remote = _files(bundle / area, ("**/*",))
        for rel in sorted(set(local) | set(remote)):
            key = f"{area}/{rel}"
            here, there = local.get(rel), remote.get(rel)
            mine = _digest(here) if here else None
            theirs = _digest(there) if there else None
            known = base.get(key)
            if mine == theirs:
                pass
            elif theirs is None or (known is not None and theirs == known):
                assert here is not None
                _copy(here, bundle / area / rel)
                report.pushed.append(key)
            elif mine is None or (known is not None and mine == known):
                assert there is not None
                _copy(there, root / rel)
                report.pulled.append(key)
            elif rel.endswith(".jsonl"):
                assert here is not None and there is not None
                merged = _merge_lines(here, there)
                here.write_bytes(merged)
                _copy(here, bundle / area / rel)
                report.merged.append(key)
            else:  # Both changed: the newer copy wins, the other is kept beside it.
                assert here is not None and there is not None
                if here.stat().st_mtime >= there.stat().st_mtime:
                    _copy(there, root / f"{rel}.conflict-shared")
                    _copy(here, bundle / area / rel)
                else:
                    _copy(here, root / f"{rel}.conflict-{name}")
                    _copy(there, root / rel)
                report.conflicts.append(key)
            target = root / rel
            if target.exists():
                after[key] = _digest(target)
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "LAST_SYNC.txt").write_text(f"{name} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(after, indent=0))
    return report


def glados_running(port: int = 8050) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def remembered_folder() -> Path | None:
    """The shared folder for this PC: GLADOS_SYNC_DIR, else the one the last ``glados sync <folder>`` used."""
    raw = os.environ.get("GLADOS_SYNC_DIR")
    if not raw:
        try:
            raw = FOLDER_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    path = Path(raw).expanduser()
    return path if raw and path.is_dir() else None


def main(folder: str | None, force: bool = False) -> int:
    remembered = remembered_folder()
    target = folder or (str(remembered) if remembered else None)
    if not target:
        print("Give the shared folder (glados sync <folder>) or set GLADOS_SYNC_DIR.")
        return 2
    shared = Path(target).expanduser()
    if not shared.is_dir():
        print(f"{shared} does not exist; create it (or point it at your Syncthing / cloud folder) first.")
        return 2
    if glados_running() and not force:
        print("GLaDOS is running and may rewrite what sync brings in. Stop it first, or pass --force.")
        return 1
    for line in sync(shared).lines():
        print(line)
    FOLDER_FILE.write_text(str(shared.resolve()), encoding="utf-8")
    print(f"This PC now syncs with {shared.resolve()} each time GLaDOS starts and stops.")
    return 0
