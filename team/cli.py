"""
CLI & interactive REPL for Agent Team.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from team.runtime import Runtime
from team.team import TEAM

HELP = """
Agent Team – local multi-agent coding system (Ollama)

Commands:
  check                     Verify Ollama is reachable
  team / show               Show agent roster
  opinion <prompt>          All specialists answer independently
  fusion <prompt>           Collect insights → synthesize best answer
  collaborate <prompt>      Full pipeline (architect→build→review→secure→test→polish)
  think <prompt>            Deep research → draft → critique → refine
  quick <prompt>            Fast single-builder path with tools
  skill <name> [context]    Run a skill
  skills                    List available skills
  workflow list             List workflows
  workflow run <name> [prompt]
  memory index              Rebuild project file index
  memory query <term>       Search indexed files
  clear                     Reset session memory
  tools                     Show available tools
  config                    Show current config
  help                      This help
  exit / quit               Leave the REPL

Tips:
  • Use 'collaborate' for serious features
  • Use 'fusion' when you want a polished single artifact
  • Use 'quick' for small focused tasks
  • Agents can read/write files and run code via tools
  • Generated / written files land in the workspace (and output/ as fallback)
"""


def _color(text: str, code: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def main(argv: Optional[list[str]] = None):
    argv = list(argv) if argv is not None else sys.argv[1:]
    root = Path.cwd()
    rt = Runtime(root=root)

    if not argv:
        _repl(rt)
        return

    cmd = argv[0].lower()
    arg = " ".join(argv[1:])

    if cmd in ("help", "-h", "--help"):
        print(HELP)
    elif cmd in ("show", "team"):
        print(rt.show_team())
    else:
        _dispatch(rt, cmd, arg)


def _repl(rt: Runtime):
    print(_color("Agent Team v1.0  –  Ollama multi-agent coding system", "1;36"))
    print("Type 'help' for commands, 'exit' to quit.\n")
    status = rt.check_ollama()
    if "OK" in status:
        print(_color("✓ " + status.split("\n")[0], "32"))
    else:
        print(_color("✗ Ollama not reachable – start it with: ollama serve", "31"))
        print("  Then: ollama pull qwen2.5-coder\n")

    while True:
        try:
            line = input(_color("team> ", "1;32")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            print("bye")
            break
        if line.lower() in ("help", "?"):
            print(HELP)
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        try:
            _dispatch(rt, cmd, arg)
        except Exception as e:
            print(_color(f"error: {e}", "31"))


def _dispatch(rt: Runtime, cmd: str, arg: str):
    if cmd == "check":
        print(rt.check_ollama())
    elif cmd == "opinion":
        if not arg:
            print("usage: opinion <prompt>")
            return
        print("\n--- Opinion ---")
        for name, text in rt.opinion(arg).items():
            print(f"\n### {name}\n{text}")
    elif cmd == "fusion":
        if not arg:
            print("usage: fusion <prompt>")
            return
        print("\n--- Fusion ---")
        print(rt.fusion(arg))
    elif cmd == "collaborate":
        if not arg:
            print("usage: collaborate <prompt>")
            return
        print("\n--- Collaborate ---")
        print(rt.collaborate(arg))
    elif cmd == "think":
        if not arg:
            print("usage: think <prompt>")
            return
        print("\n--- Think ---")
        print(rt.think(arg))
    elif cmd == "quick":
        if not arg:
            print("usage: quick <prompt>")
            return
        print("\n--- Quick ---")
        print(rt.quick(arg))
    elif cmd == "skill":
        parts = arg.split(maxsplit=1)
        name = parts[0] if parts else ""
        ctx = parts[1] if len(parts) > 1 else ""
        if not name:
            print("usage: skill <name> [context]")
            print("Available:", ", ".join(rt.list_skills()) or "(none)")
            return
        print(rt.run_skill(name, ctx))
    elif cmd == "skills":
        for s in rt.list_skills() or ["(none)"]:
            print(f"  • {s}")
    elif cmd == "workflow":
        parts = arg.split(maxsplit=2)
        action = parts[0] if parts else "list"
        if action == "list":
            for w in rt.list_workflows() or ["(none)"]:
                print(f"  • {w}")
        elif action == "run":
            name = parts[1] if len(parts) > 1 else ""
            prompt = parts[2] if len(parts) > 2 else ""
            if not name:
                print("usage: workflow run <name> [prompt]")
                return
            print(rt.run_workflow(name, prompt))
        else:
            print("usage: workflow list|run <name> [prompt]")
    elif cmd == "memory":
        parts = arg.split(maxsplit=1)
        action = parts[0] if parts else "query"
        q = parts[1] if len(parts) > 1 else ""
        if action == "index":
            print(rt.memory_index(q or "."))
        elif action == "query":
            for h in rt.memory_query(q):
                print(f"  • {h}")
        else:
            print("usage: memory index|query [term]")
    elif cmd == "clear":
        print(rt.clear_session())
    elif cmd == "tools":
        print("Available tools:")
        for t in rt.tools.available():
            print(f"  • {t}")
        print("\nAgents request tools with <tool>...</tool> blocks.")
    elif cmd == "config":
        for k, v in rt.cfg.as_dict().items():
            print(f"  {k}: {v}")
    elif cmd in ("show", "team"):
        print(rt.show_team())
    else:
        print(f"unknown command: {cmd}  (type 'help')")


if __name__ == "__main__":
    main()
