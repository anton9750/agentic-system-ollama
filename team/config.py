"""Configuration for Agent Team v1.0 – local Ollama multi-agent system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # Ollama
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    default_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder"))
    request_timeout: int = 300  # seconds

    # Runtime behaviour
    max_tool_rounds: int = 12
    max_session_turns: int = 30
    temperature_default: float = 0.25
    enable_tools: bool = True
    auto_save_code: bool = True
    verify_after_write: bool = True  # try to syntax-check Python after write

    # Paths (resolved relative to project root at runtime)
    output_dir: str = "output"
    memory_dir: str = ".team-memory"
    skills_dir: str = "skills"
    workflows_dir: str = "workspaces"

    # Safety
    allow_shell: bool = True
    shell_timeout: int = 30
    python_timeout: int = 20
    max_file_read_bytes: int = 200_000
    max_search_results: int = 40

    # Display
    color: bool = True
    verbose: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "ollama_host": self.ollama_host,
            "default_model": self.default_model,
            "max_tool_rounds": self.max_tool_rounds,
            "enable_tools": self.enable_tools,
            "allow_shell": self.allow_shell,
        }


def load_config(root: Path | None = None) -> Config:
    """Load config, optionally overriding from a simple JSON file."""
    cfg = Config()
    if root is None:
        root = Path.cwd()
    cfg_path = root / "team-config.json"
    if cfg_path.exists():
        try:
            import json
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception:
            pass
    return cfg
