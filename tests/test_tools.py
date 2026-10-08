"""Basic unit tests for the tool layer (no Ollama required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from team.config import Config
from team.tools import ToolRegistry, parse_tool_calls, strip_tool_blocks


@pytest.fixture
def workspace(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "hello.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def tools(workspace: Path):
    cfg = Config()
    cfg.allow_shell = True
    return ToolRegistry(workspace, cfg)


def test_list_dir(tools: ToolRegistry):
    r = tools.execute("list_dir", {"path": "."})
    assert r.success
    assert "src" in r.output


def test_read_write_roundtrip(tools: ToolRegistry, workspace: Path):
    r = tools.execute("write_file", {"path": "src/new.py", "content": "x = 1\n"})
    assert r.success
    r2 = tools.execute("read_file", {"path": "src/new.py"})
    assert r2.success
    assert "x = 1" in r2.output


def test_syntax_check_ok(tools: ToolRegistry):
    r = tools.execute("syntax_check", {"path": "src/hello.py"})
    assert r.success


def test_syntax_check_bad(tools: ToolRegistry, workspace: Path):
    (workspace / "src" / "bad.py").write_text("def broken(\n", encoding="utf-8")
    r = tools.execute("syntax_check", {"path": "src/bad.py"})
    assert not r.success
    assert "SyntaxError" in r.error


def test_search_files(tools: ToolRegistry):
    r = tools.execute("search_files", {"pattern": "greet", "glob": "*.py"})
    assert r.success
    assert "hello.py" in r.output


def test_run_python(tools: ToolRegistry):
    r = tools.execute("run_python", {"code": "print(2 + 2)"})
    assert r.success
    assert "4" in r.output


def test_parse_tool_calls():
    text = """
Some reasoning.
<tool>
name: read_file
path: src/hello.py
</tool>
More text.
<tool>
name: list_dir
path: .
</tool>
"""
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0][0] == "read_file"
    assert calls[0][1]["path"] == "src/hello.py"
    assert "read_file" not in strip_tool_blocks(text)


def test_path_traversal_blocked(tools: ToolRegistry):
    r = tools.execute("read_file", {"path": "../../etc/passwd"})
    assert not r.success
