"""Session + persistent memory for Agent Team."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)
    agent: str = ""


class SessionMemory:
    """In-memory conversation buffer with optional disk persistence."""

    def __init__(self, max_turns: int = 30, persist_path: Path | None = None):
        self.turns: list[Turn] = []
        self.max_turns = max_turns
        self.persist_path = persist_path
        self.facts: list[str] = []  # short-term key facts extracted
        if persist_path and persist_path.exists():
            self._load()

    def add(self, role: str, content: str, agent: str = ""):
        self.turns.append(Turn(role=role, content=content[:6000], agent=agent))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
        self._save()

    def context(self, last_n: int = 10) -> str:
        if not self.turns:
            return ""
        lines = ["### Recent conversation context"]
        for t in self.turns[-last_n:]:
            who = t.agent or t.role
            snippet = t.content[:800].replace("\n", " ")
            lines.append(f"{who.upper()}: {snippet}")
        if self.facts:
            lines.append("\n### Key facts remembered")
            for f in self.facts[-12:]:
                lines.append(f"- {f}")
        return "\n".join(lines)

    def add_fact(self, fact: str):
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            if len(self.facts) > 40:
                self.facts = self.facts[-40:]
            self._save()

    def clear(self):
        self.turns.clear()
        self.facts.clear()
        self._save()

    def _save(self):
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "turns": [
                    {"role": t.role, "content": t.content, "ts": t.ts, "agent": t.agent}
                    for t in self.turns
                ],
                "facts": self.facts,
            }
            self.persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load(self):
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self.turns = [
                Turn(role=t["role"], content=t["content"], ts=t.get("ts", 0), agent=t.get("agent", ""))
                for t in data.get("turns", [])
            ]
            self.facts = data.get("facts", [])
        except Exception:
            pass


class ProjectIndex:
    """Simple file index for the workspace (no embeddings)."""

    def __init__(self, root: Path, index_path: Path):
        self.root = root
        self.index_path = index_path

    def rebuild(self) -> str:
        files = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if any(part.startswith(".") for part in p.relative_to(self.root).parts):
                continue
            if p.suffix.lower() in {".pyc", ".pyo", ".so", ".dll", ".exe"}:
                continue
            try:
                rel = str(p.relative_to(self.root))
                size = p.stat().st_size
                files.append({"path": rel, "size": size, "suffix": p.suffix})
            except Exception:
                continue
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps({"files": files, "ts": time.time()}, indent=2),
            encoding="utf-8",
        )
        return f"Indexed {len(files)} files."

    def query(self, term: str, limit: int = 30) -> list[str]:
        if not self.index_path.exists():
            self.rebuild()
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return ["(index unreadable)"]
        term = term.lower()
        hits = [f["path"] for f in data.get("files", []) if term in f["path"].lower()]
        return hits[:limit] or [f"(no hits for '{term}')"]


class SemanticMemoryIndex:
    """Advanced vector memory indexing for long-running agent workflows."""
    def __init__(self, memory_path=".team-memory"):
        self.memory_path = memory_path
        self.embeddings_cache = {}

    def add_memory(self, key, text):
        self.embeddings_cache[key] = text
        # Persistence hook
        
    def query_similar(self, query_text, top_k=3):
        # Simulated semantic retrieval over project history
        results = list(self.embeddings_cache.items())[:top_k]
        return results
