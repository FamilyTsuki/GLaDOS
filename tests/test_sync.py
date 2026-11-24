import json
import os
from pathlib import Path
import time

from glados import sync as sync_module
from glados.cli import load_env_file
from glados.core.config import machine_overlays


def _pc(root: Path) -> tuple[tuple[tuple[str, Path, tuple[str, ...]], ...], Path]:
    (root / "data").mkdir(parents=True)
    (root / "home").mkdir()
    areas = (("data", root / "data", ("*.yaml", "*.jsonl")), ("home", root / "home", ("**/*",)))
    return areas, root / "base.json"


def _run(pc: tuple[tuple[tuple[str, Path, tuple[str, ...]], ...], Path], shared: Path, name: str) -> sync_module.Report:
    areas, base = pc
    return sync_module.sync(shared, base_path=base, areas=areas, name=name)


def test_two_pcs_converge_without_losing_anything(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop, desktop = _pc(tmp_path / "laptop"), _pc(tmp_path / "desktop")
    (tmp_path / "laptop/data/preferences.yaml").write_text("lang: fr\n")
    (tmp_path / "laptop/home/memory").mkdir()
    (tmp_path / "laptop/home/memory/emotion.jsonl").write_text('{"a": 1}\n')
    assert "data/preferences.yaml" in _run(laptop, shared, "laptop").pushed
    report = _run(desktop, shared, "desktop")
    assert set(report.pulled) == {"data/preferences.yaml", "home/memory/emotion.jsonl"}

    # Each PC adds memories; one changes a setting.
    with (tmp_path / "laptop/home/memory/emotion.jsonl").open("a") as f:
        f.write('{"b": 2}\n')
    with (tmp_path / "desktop/home/memory/emotion.jsonl").open("a") as f:
        f.write('{"c": 3}\n')
    (tmp_path / "desktop/data/preferences.yaml").write_text("lang: en\n")
    _run(desktop, shared, "desktop")
    report = _run(laptop, shared, "laptop")
    assert report.merged == ["home/memory/emotion.jsonl"]
    assert report.pulled == ["data/preferences.yaml"]
    _run(desktop, shared, "desktop")
    for pc in ("laptop", "desktop"):
        lines = (tmp_path / pc / "home/memory/emotion.jsonl").read_text().splitlines()
        assert sorted(lines) == ['{"a": 1}', '{"b": 2}', '{"c": 3}']
        assert (tmp_path / pc / "data/preferences.yaml").read_text() == "lang: en\n"
    assert _run(laptop, shared, "laptop").lines() == ["Already in step."]


def test_both_changed_keeps_newer_and_saves_the_other(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop, desktop = _pc(tmp_path / "laptop"), _pc(tmp_path / "desktop")
    (tmp_path / "laptop/data/manners.yaml").write_text("v: 0\n")
    _run(laptop, shared, "laptop")
    _run(desktop, shared, "desktop")
    (tmp_path / "laptop/data/manners.yaml").write_text("v: laptop\n")
    _run(laptop, shared, "laptop")
    old = time.time() - 100
    (tmp_path / "desktop/data/manners.yaml").write_text("v: desktop\n")
    os.utime(tmp_path / "desktop/data/manners.yaml", (old, old))
    report = _run(desktop, shared, "desktop")
    assert report.conflicts == ["data/manners.yaml"]
    assert (tmp_path / "desktop/data/manners.yaml").read_text() == "v: laptop\n"
    assert (tmp_path / "desktop/data/manners.yaml.conflict-desktop").read_text() == "v: desktop\n"


def test_machine_files_never_travel(tmp_path: Path) -> None:
    (tmp_path / "decision_lists.yaml").write_text("x")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/a.yaml").write_text("x")
    (tmp_path / "ok.yaml").write_text("x")
    assert set(sync_module._files(tmp_path, ("**/*",))) == {"ok.yaml"}


def test_machine_overlay_is_merged_last(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    base = tmp_path / "my.yaml"
    base.write_text("x")
    (tmp_path / "machines").mkdir()
    monkeypatch.setattr(sync_module.socket, "gethostname", lambda: "Gamer-PC.local")
    assert machine_overlays([base]) == []
    (tmp_path / "machines" / "gamer-pc.yaml").write_text("x")
    assert machine_overlays([base]) == [tmp_path / "machines" / "gamer-pc.yaml"]
    monkeypatch.setenv("GLADOS_NO_MACHINE_CONFIG", "1")
    assert machine_overlays([base]) == []


def test_env_file_fills_only_missing_variables(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    env = tmp_path / ".env"
    env.write_text('# keys\nOPENAI_API_KEY="sk-test"\nexport GLADOS_X=1\nALREADY=file\nEMPTY=\n')
    monkeypatch.setenv("ALREADY", "shell")
    for key in ("OPENAI_API_KEY", "GLADOS_X", "EMPTY"):
        monkeypatch.delenv(key, raising=False)
    load_env_file(env)
    assert os.environ["OPENAI_API_KEY"] == "sk-test" and os.environ["GLADOS_X"] == "1"
    assert os.environ["ALREADY"] == "shell" and "EMPTY" not in os.environ


def test_big_files_compare_by_size_and_time(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sync_module, "BIG_FILE", 10)
    big = tmp_path / "model.gguf"
    big.write_bytes(b"0" * 20)
    copy = tmp_path / "copy.gguf"
    sync_module._copy(big, copy)
    assert sync_module._digest(big) == sync_module._digest(copy)
    assert json.dumps(sync_module._digest(big)).startswith('"size:20')
