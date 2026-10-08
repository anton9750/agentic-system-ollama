"""Tests for agent roster."""

from team.team import TEAM, get_agent, Role, agents_by_role


def test_team_not_empty():
    assert len(TEAM) >= 8


def test_get_agent():
    a = get_agent("main")
    assert a is not None
    assert a.role == Role.BUILDER
    assert a.use_tools is True


def test_agents_by_role():
    builders = agents_by_role(Role.BUILDER)
    assert any(a.name == "main" for a in builders)
    assert any(a.name == "secondary" for a in builders)


def test_unique_names():
    names = [a.name for a in TEAM]
    assert len(names) == len(set(names))
