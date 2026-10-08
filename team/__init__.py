"""Agent Team – intelligent multi-agent coding system powered by local Ollama."""

__version__ = "1.0.0"

from team.cli import main
from team.runtime import Runtime
from team.team import TEAM, get_agent, get_team

__all__ = ["main", "Runtime", "TEAM", "get_agent", "get_team", "__version__"]
