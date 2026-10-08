"""Tests for session memory and project index."""

from __future__ import annotations

from pathlib import Path

from team.memory import ProjectIndex, SessionMemory


def test_session_add_and_context(tmp_path: Path):
    mem = SessionMemory(max_turns=5, persist_path=tmp_path / "session.json")
    mem.add("user", "hello")
    mem.add("assistant", "hi there", agent="main")
    ctx = mem.context()
    assert "hello" in ctx
    assert "hi there" in ctx
    assert "MAIN" in ctx or "main" in ctx.lower()


def test_session_persistence(tmp_path: Path):
    path = tmp_path / "session.json"
    mem = SessionMemory(persist_path=path)
    mem.add("user", "persist me")
    mem.add_fact("project uses FastAPI")
    mem2 = SessionMemory(persist_path=path)
    assert any("persist me" in t.content for t in mem2.turns)
    assert "project uses FastAPI" in mem2.facts


def test_project_index(tmp_path: Path):
    (tmp_path / "a.py").write_text("x=1", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    idx = ProjectIndex(tmp_path, tmp_path / "index.json")
    msg = idx.rebuild()
    assert "2" in msg or "Indexed" in msg
    hits = idx.query("a.py")
    assert any("a.py" in h for h in hits)
