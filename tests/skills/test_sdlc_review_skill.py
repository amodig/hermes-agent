"""Behavioural contracts for the bundled SDLC review skill.

`skills/AGENTS.md` requires a per-skill suite at this path. This one checks the
two contracts the skill itself cannot state: which runtime environment offers it,
and that every Kanban transition it instructs a reviewer to call actually exists.
Prose wording is deliberately not asserted here — the typed review transitions are
covered by the lifecycle conformance suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.skill_utils import parse_frontmatter, skill_matches_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "devops" / "sdlc-review" / "SKILL.md"
# The reviewer's route for each card contract, cited as `tool(...)` in the skill.
DOCUMENTED_TRANSITIONS = (
    "kanban_request_changes",
    "kanban_complete",
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_is_offered_to_kanban_workers(
    skill_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frontmatter, _body = parse_frontmatter(skill_text)
    assert frontmatter["name"] == "sdlc-review"
    assert frontmatter["version"]
    assert frontmatter["environments"] == ["kanban"]

    # A dispatcher-spawned worker owns HERMES_KANBAN_TASK: only then is the skill
    # offered, so a reviewer is never handed review guidance outside the lane.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0123456789abcdef")
    assert skill_matches_environment(frontmatter) is True


def test_documented_transitions_are_registered_kanban_tools(skill_text: str) -> None:
    from tools.registry import registry

    import tools.kanban_tools  # noqa: F401 - registers the kanban tools

    cited = set(re.findall(r"`(kanban_[a-z_]+)\(", skill_text))
    assert set(DOCUMENTED_TRANSITIONS) <= cited

    unregistered = sorted(
        name
        for name in cited
        if (entry := registry.get_entry(name)) is None or entry.toolset != "kanban"
    )
    assert unregistered == [], f"skill documents unknown kanban tools: {unregistered}"
