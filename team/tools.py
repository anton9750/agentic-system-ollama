"""
Tool registry and implementations for Agent Team.
Agents request tools via a simple XML-like format that the runtime parses.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from team.config import Config


@dataclass
class ToolResult:
    name: str
    success: bool
    output: str
    error: str = ""

    def as_text(self) -> str:
        if self.success:
            return f"[tool:{self.name} OK]\n{self.output}"
        return f"[tool:{self.name} FAILED]\n{self.error or self.output}"


class ToolRegistry:
    """Register and execute tools. All tools are intentionally local / sandboxed."""

    def __init__(self, root: Path, cfg: Config):
        self.root = root.resolve()
        self.cfg = cfg
        self._tools: dict[str, Callable[..., ToolResult]] = {
            "list_dir": self._list_dir,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "run_python": self._run_python,
            "run_shell": self._run_shell,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "list_output": self._list_output,
            "syntax_check": self._syntax_check,
        }

    def available(self) -> list[str]:
        names = list(self._tools.keys())
        if not self.cfg.allow_shell:
            names = [n for n in names if n != "run_shell"]
        return names

    def describe(self) -> str:
        """Human-readable tool catalog injected into agent system prompts."""
        return """You have access to the following tools. To use a tool, emit EXACTLY this format (you may use multiple tools in one response):

<tool>
name: TOOL_NAME
arg1: value
arg2: value
</tool>

Available tools:

1. list_dir
   path: relative path (default ".")
   → lists files and directories

2. read_file
   path: relative file path
   → returns file contents (truncated if huge)

3. write_file
   path: relative file path
   content: the full file content (use this for complete files)
   → writes the file and returns confirmation + syntax check if Python

4. search_files
   pattern: regex or plain text
   path: directory to search (default ".")
   glob: optional file glob e.g. "*.py"
   → returns matching lines with file:line

5. run_python
   code: Python source to execute in a temporary file
   → runs with timeout, returns stdout/stderr/exit code

6. run_shell
   command: shell command (restricted, timeout enforced)
   → executes and returns output (only if enabled)

7. git_status
   → shows git status if inside a git repo

8. git_diff
   → shows unstaged + staged diff

9. list_output
   → lists files previously generated in the output/ directory

10. syntax_check
    path: path to a .py file
    → AST-parses the file and reports syntax errors

Rules for tool use:
- Prefer write_file for complete production files.
- After writing code, you should usually run syntax_check or run_python with a small smoke test.
- Do not invent tool names.
- When you have finished and no longer need tools, give your final answer WITHOUT any <tool> blocks.
"""

    def execute(self, name: str, args: dict[str, str]) -> ToolResult:
        fn = self._tools.get(name)
        if not fn:
            return ToolResult(name, False, "", f"Unknown tool: {name}")
        if name == "run_shell" and not self.cfg.allow_shell:
            return ToolResult(name, False, "", "Shell tool is disabled in config")
        try:
            return fn(**args)
        except TypeError as e:
            return ToolResult(name, False, "", f"Bad arguments for {name}: {e}")
        except Exception as e:
            return ToolResult(name, False, "", f"Tool error: {e}")

    # ── path safety ──────────────────────────────────────────────
    def _safe_path(self, rel: str) -> Path:
        """Resolve path under project root; prevent path traversal."""
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"Path escapes project root: {rel}")
        return p

    # ── tool implementations ─────────────────────────────────────
    def _list_dir(self, path: str = ".") -> ToolResult:
        p = self._safe_path(path)
        if not p.exists():
            return ToolResult("list_dir", False, "", f"Not found: {path}")
        if not p.is_dir():
            return ToolResult("list_dir", False, "", f"Not a directory: {path}")
        entries = []
        for child in sorted(p.iterdir()):
            kind = "dir" if child.is_dir() else "file"
            size = child.stat().st_size if child.is_file() else 0
            entries.append(f"{kind:4} {size:>10}  {child.name}")
        return ToolResult("list_dir", True, "\n".join(entries) or "(empty)")

    def _read_file(self, path: str = "") -> ToolResult:
        if not path:
            return ToolResult("read_file", False, "", "path is required")
        p = self._safe_path(path)
        if not p.exists() or not p.is_file():
            return ToolResult("read_file", False, "", f"File not found: {path}")
        data = p.read_bytes()
        if len(data) > self.cfg.max_file_read_bytes:
            data = data[: self.cfg.max_file_read_bytes]
            truncated = True
        else:
            truncated = False
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        if truncated:
            text += f"\n\n... [truncated at {self.cfg.max_file_read_bytes} bytes]"
        return ToolResult("read_file", True, text)

    def _write_file(self, path: str = "", content: str = "") -> ToolResult:
        if not path:
            return ToolResult("write_file", False, "", "path is required")
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Normalize newlines
        content = content.replace("\r\n", "\n")
        p.write_text(content, encoding="utf-8")
        msg = f"Wrote {p.relative_to(self.root)} ({len(content)} bytes)"
        if path.endswith(".py") and self.cfg.verify_after_write:
            check = self._syntax_check(path=path)
            msg += "\n" + check.as_text()
        return ToolResult("write_file", True, msg)

    def _search_files(
        self,
        pattern: str = "",
        path: str = ".",
        glob: str = "*",
    ) -> ToolResult:
        if not pattern:
            return ToolResult("search_files", False, "", "pattern is required")
        root = self._safe_path(path)
        if not root.exists():
            return ToolResult("search_files", False, "", f"Path not found: {path}")
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult("search_files", False, "", f"Invalid regex: {e}")

        hits: list[str] = []
        for fp in root.rglob(glob):
            if not fp.is_file():
                continue
            if any(part.startswith(".") for part in fp.parts):
                continue  # skip hidden
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    rel = fp.relative_to(self.root)
                    hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(hits) >= self.cfg.max_search_results:
                        hits.append("... (truncated)")
                        return ToolResult("search_files", True, "\n".join(hits))
        return ToolResult("search_files", True, "\n".join(hits) or "(no matches)")

    def _run_python(self, code: str = "") -> ToolResult:
        if not code.strip():
            return ToolResult("run_python", False, "", "code is required")
        tmp = self.root / ".team-memory" / f"_tmp_run_{int(time.time()*1000)}.py"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                ["python3", str(tmp)],
                capture_output=True,
                text=True,
                timeout=self.cfg.python_timeout,
                cwd=str(self.root),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            out = []
            if proc.stdout:
                out.append("STDOUT:\n" + proc.stdout)
            if proc.stderr:
                out.append("STDERR:\n" + proc.stderr)
            out.append(f"EXIT: {proc.returncode}")
            success = proc.returncode == 0
            return ToolResult("run_python", success, "\n".join(out))
        except subprocess.TimeoutExpired:
            return ToolResult("run_python", False, "", f"Timed out after {self.cfg.python_timeout}s")
        except Exception as e:
            return ToolResult("run_python", False, "", str(e))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _run_shell(self, command: str = "") -> ToolResult:
        if not command.strip():
            return ToolResult("run_shell", False, "", "command is required")
        # Very light safety filter
        banned = ["rm -rf /", "mkfs", ":(){", "dd if=", "> /dev/sd"]
        lower = command.lower()
        for b in banned:
            if b in lower:
                return ToolResult("run_shell", False, "", f"Blocked dangerous pattern: {b}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.cfg.shell_timeout,
                cwd=str(self.root),
            )
            out = []
            if proc.stdout:
                out.append(proc.stdout)
            if proc.stderr:
                out.append(proc.stderr)
            out.append(f"[exit {proc.returncode}]")
            return ToolResult("run_shell", proc.returncode == 0, "\n".join(out))
        except subprocess.TimeoutExpired:
            return ToolResult("run_shell", False, "", f"Timed out after {self.cfg.shell_timeout}s")
        except Exception as e:
            return ToolResult("run_shell", False, "", str(e))

    def _git_status(self) -> ToolResult:
        try:
            proc = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.root),
            )
            if proc.returncode != 0:
                return ToolResult("git_status", False, "", proc.stderr or "Not a git repo?")
            return ToolResult("git_status", True, proc.stdout or "(clean)")
        except Exception as e:
            return ToolResult("git_status", False, "", str(e))

    def _git_diff(self) -> ToolResult:
        try:
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(self.root),
            )
            text = proc.stdout or "(no diff)"
            if len(text) > 30_000:
                text = text[:30_000] + "\n... [truncated]"
            return ToolResult("git_diff", True, text)
        except Exception as e:
            return ToolResult("git_diff", False, "", str(e))

    def _list_output(self) -> ToolResult:
        out_dir = self.root / "output"
        if not out_dir.exists():
            return ToolResult("list_output", True, "(output/ empty or missing)")
        files = sorted(out_dir.rglob("*"))
        lines = [str(f.relative_to(self.root)) for f in files if f.is_file()]
        return ToolResult("list_output", True, "\n".join(lines) or "(empty)")

    def _syntax_check(self, path: str = "") -> ToolResult:
        if not path:
            return ToolResult("syntax_check", False, "", "path is required")
        p = self._safe_path(path)
        if not p.exists():
            return ToolResult("syntax_check", False, "", f"File not found: {path}")
        try:
            src = p.read_text(encoding="utf-8")
            ast.parse(src)
            return ToolResult("syntax_check", True, f"Syntax OK: {path}")
        except SyntaxError as e:
            return ToolResult(
                "syntax_check",
                False,
                "",
                f"SyntaxError in {path} line {e.lineno}: {e.msg}",
            )
        except Exception as e:
            return ToolResult("syntax_check", False, "", str(e))


# ── parser for agent tool calls ──────────────────────────────────

TOOL_BLOCK_RE = re.compile(
    r"<tool>\s*(.*?)\s*</tool>",
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
    """Extract tool calls from agent response text."""
    calls = []
    for match in TOOL_BLOCK_RE.finditer(text):
        body = match.group(1)
        args: dict[str, str] = {}
        name = ""
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if key == "name":
                    name = val
                else:
                    args[key] = val
        if name:
            # Support multi-line content: if content was split, rejoin remaining
            if "content" in args and args["content"] == "" and "\n" in body:
                # fallback: take everything after "content:"
                idx = body.lower().find("content:")
                if idx >= 0:
                    args["content"] = body[idx + len("content:") :].strip()
            calls.append((name, args))
    return calls


def strip_tool_blocks(text: str) -> str:
    return TOOL_BLOCK_RE.sub("", text).strip()
